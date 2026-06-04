"""Dense per-pixel hierarchical material segmentation model.

Architecture:
  ResNet50 (features_only) → FPN → dense feature map
  ResNet50 C5 pool → GNN → refined prototype embeddings (once per image)
  einsum(dense_feat, prototypes) → [B, num_nodes, H/4, W/4]
"""

import networkx as nx
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected

from gnn_classifier.hgnn import GraphBackbone
from taxonomy.tree import get_edge_index, get_hierarchy_mask, taxa_to_indices


class FPN(nn.Module):
    """Top-down Feature Pyramid Network over ResNet50 C2..C5. Returns P2 only."""

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.smooths = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

    def forward(self, features):
        # features: [C2, C3, C4, C5] from timm features_only backbone
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        # Top-down pathway: start from C5, merge down to C2
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        # Return P2 (finest level, index 0) after smoothing
        return self.smooths[0](laterals[0])


class DenseMaterialSegmentor(nn.Module):
    """
    Dense per-pixel material segmentation with hierarchical taxonomy output.

    forward(images) -> [B, num_nodes, H/4, W/4]
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        cnn_backbone: str = "resnet50",
        pretrained: bool = True,
        fpn_channels: int = 256,
        proto_dim: int = 256,
        gnn_kwargs: dict = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if gnn_kwargs is None:
            gnn_kwargs = {
                "input_dim": 256,
                "hidden_dim": 256,
                "output_dim": 256,
                "num_layers": 2,
                "num_heads": 4,
            }

        self.num_nodes = graph.number_of_nodes()
        self.proto_dim = proto_dim

        # Backbone: ResNet50, returns [C2, C3, C4, C5]
        self.backbone = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        backbone_channels = self.backbone.feature_info.channels()  # [256, 512, 1024, 2048]
        c5_channels = backbone_channels[-1]

        # FPN over C2..C5 → P2
        self.fpn = FPN(in_channels=backbone_channels, out_channels=fpn_channels)

        # Project C5 global pool → GNN input space
        gnn_input_dim = gnn_kwargs["input_dim"]
        self.img_proj = nn.Linear(c5_channels, gnn_input_dim)

        # GNN + prototypes (same as HGNN)
        self.prototypes = nn.Embedding(self.num_nodes, gnn_input_dim)
        self.gnn = GraphBackbone(**gnn_kwargs)

        gnn_output_dim = gnn_kwargs["output_dim"]

        # Project dense P2 features → prototype dimension
        self.dense_proj = nn.Sequential(
            nn.Conv2d(fpn_channels, proto_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proto_dim, gnn_output_dim, 1),
        )

        self.dropout = nn.Dropout(p=dropout)

        self._init_graph(graph)

        nn.init.normal_(self.prototypes.weight, std=0.02)

    def _init_graph(self, graph: nx.DiGraph):
        hierarchy_mask = get_hierarchy_mask(graph)

        edge_index = to_undirected(
            torch.tensor(get_edge_index(graph), dtype=torch.long).t().contiguous()
        )
        context_to_labels = torch.stack(
            [
                torch.zeros(graph.number_of_nodes(), dtype=torch.long),
                torch.arange(1, graph.number_of_nodes() + 1, dtype=torch.long),
            ],
            dim=0,
        )
        edge_index_global_context = torch.cat([edge_index + 1, context_to_labels], dim=1)

        self.register_buffer("edge_index_global_context", edge_index_global_context)

    def _run_gnn(self, img_emb: torch.Tensor) -> torch.Tensor:
        """
        Run GNN once per image to get refined prototype embeddings.

        img_emb: [B, gnn_input_dim]
        returns: [B, num_nodes, gnn_output_dim]
        """
        B = img_emb.size(0)
        data_list = []
        for i in range(B):
            x = torch.cat([img_emb[i].unsqueeze(0), self.prototypes.weight], dim=0)
            data_list.append(Data(x=x, edge_index=self.edge_index_global_context))
        batched = Batch.from_data_list(data_list)

        # batch=None skips global_mean_pool → returns [B*(1+num_nodes), gnn_output_dim]
        node_feats = self.gnn(batched.x, batched.edge_index, batch=None)
        # Reshape and drop context node (index 0 per graph)
        node_feats = node_feats.view(B, 1 + self.num_nodes, -1)
        proto_emb = self.dropout(node_feats[:, 1:, :])  # [B, num_nodes, gnn_output_dim]
        return proto_emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: [B, num_nodes, H/4, W/4]
        """
        feats = self.backbone(images)          # [C2, C3, C4, C5]

        # Branch 1: GNN prototype refinement (global context)
        c5_pool = feats[-1].mean(dim=[2, 3])   # [B, 2048]
        img_emb = self.img_proj(c5_pool)       # [B, gnn_input_dim]
        proto_emb = self._run_gnn(img_emb)     # [B, num_nodes, gnn_output_dim]

        # Branch 2: Spatial FPN features
        p2 = self.fpn(feats)                   # [B, fpn_channels, H/4, W/4]
        dense = self.dense_proj(p2)            # [B, gnn_output_dim, H/4, W/4]

        # Pixel-prototype dot product
        logits = torch.einsum("bdhw,bnd->bnhw", dense, proto_emb)
        return logits
