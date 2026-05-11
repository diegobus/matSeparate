from typing import Dict

import networkx as nx
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_undirected
from typing import Tuple, Union

from taxonomy.tree import get_edge_index, get_hierarchy_mask, taxa_to_indices


class Permute(nn.Module):
    def __init__(self, *dims):
        """
        Args:
            dims: a permutation of the input tensor's dimensions.
                  e.g. (0,2,3,1) to go from [B,C,H,W] → [B,H,W,C]
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    def __init__(self, dim: Union[int, Tuple[int], None] = None):
        """
        Args:
            dim:
              - an int to squeeze that specific dimension if it's size 1,
              - a tuple of ints to squeeze multiple dims (applied in order),
              - or None to squeeze all size-1 dims.
        """
        super().__init__()
        if isinstance(dim, tuple):
            self.dims = dim
        else:
            self.dims = (dim,) if dim is not None else ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dims:
            return x.squeeze()
        for d in sorted(self.dims, reverse=True):
            x = x.squeeze(d)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        finetune: bool = False,
        in_channels: int = 3,
        is_transformer: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.is_transformer = is_transformer

        backbone: nn.Module = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            # num_classes=0,
            # global_pool="",
            # features_only=True
        )
        classifier_input_dim = (
            backbone.get_classifier().in_features
        )  # Get the input size of the classifier

        # done for distrubted training, and cannot just be initialized in timm.create_model
        # afaict since we are also changing the pooling payer
        self.cnn = backbone

        if hasattr(self.cnn, "fc"):
            delattr(self.cnn, "fc")
        elif hasattr(self.cnn, "head"):
            delattr(self.cnn, "head")

        if hasattr(self.cnn, "fc_norm"):
            delattr(self.cnn, "fc_norm")

        if self.is_transformer:
            self.pooling = nn.Sequential(
                Permute(0, 2, 1), nn.AdaptiveAvgPool1d(output_size=1), Squeeze(-1)
            )
        else:
            self.pooling = timm.layers.SelectAdaptivePool2d(
                pool_type="avg",
                flatten=nn.Flatten(start_dim=1, end_dim=-1),
            )

        self.classifier = nn.Linear(classifier_input_dim, output_dim)

        if finetune:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs) -> torch.Tensor:
        features = self.cnn.forward_features(images)
        features = self.pooling(features)
        logits = self.classifier(features)
        return logits


class GraphBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 1,
        skip_connection: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.dropout = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList(
            [
                GATConv(
                    input_dim if i == 0 else hidden_dim * num_heads,
                    hidden_dim if i < num_layers - 1 else output_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    concat=True if i < num_layers - 1 else False,
                    dropout=dropout,
                    residual=skip_connection,
                )
                for i in range(num_layers)
            ]
        )
        # Add BatchNorm layers for all but the final GNN layer.
        # self.bns = nn.ModuleList(
        #     [nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers - 1)]
        # )

    def forward(self, node_features, edge_index, batch=None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            node_features = conv(node_features, edge_index)

            if i < len(self.convs) - 1:
                # node_features = self.bns[i](node_features)
                node_features = F.leaky_relu(node_features)
                node_features = self.dropout(node_features)

        if batch is not None:
            node_features = global_mean_pool(node_features, batch)

        return node_features.squeeze()


class HGNN(nn.Module):
    def __init__(
        self,
        graph: nx.DiGraph,
        path_predict: bool = False,
        dropout_prob: float = 0.0,
        cnn_kwargs: Dict = {},
        gnn_kwargs: Dict = {},
    ):
        super().__init__()
        self.path_predict = path_predict
        self.num_nodes = graph.number_of_nodes()

        self.cnn = ImageEncoder(**cnn_kwargs)
        self.gnn = GraphBackbone(**gnn_kwargs)
        self.projection = nn.Linear(self.cnn.output_dim, self.gnn.input_dim)
        self.prototypes = nn.Embedding(self.num_nodes, self.gnn.input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.gnn.output_dim, self.num_nodes)

        self._init_graph(graph)

    def _init_graph(self, graph: nx.DiGraph):
        adjacency_matrix = torch.tensor(
            nx.adjacency_matrix(graph).toarray(),
            dtype=torch.float32,
        )
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
        edge_index_ancestors = (
            torch.tensor(
                taxa_to_indices(
                    nx.from_numpy_array(
                        hierarchy_mask,
                        create_using=nx.DiGraph,
                        nodelist=graph.nodes,
                    ).edges,
                    graph,
                ),
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )
        edge_index_descendants = (
            torch.tensor(
                taxa_to_indices(
                    nx.from_numpy_array(
                        hierarchy_mask.T,
                        create_using=nx.DiGraph,
                        nodelist=graph.nodes,
                    ).edges,
                    graph,
                ),
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        hierarchy_mask = torch.tensor(hierarchy_mask, dtype=torch.float32)
        edge_index_global_context = torch.cat(
            [edge_index + 1, context_to_labels], dim=1
        )
        edge_index_global_context_anc_desc = torch.cat(
            [
                edge_index + 1,
                edge_index_ancestors + 1,
                edge_index_descendants + 1,
                context_to_labels,
            ],
            dim=1,
        )

        self.register_buffer("hierarchy_mask", hierarchy_mask)
        self.register_buffer("adjacency_matrix", adjacency_matrix)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_index_ancestors", edge_index_ancestors)
        self.register_buffer("edge_index_descendants", edge_index_descendants)
        self.register_buffer("edge_index_global_context", edge_index_global_context)
        self.register_buffer(
            "edge_index_global_context_anc_desc", edge_index_global_context_anc_desc
        )

    def extract_features(self, images: torch.Tensor):
        image_features = self.projection(self.cnn(images))
        return image_features

    def forward(self, images: torch.Tensor):
        if self.training:
            return self._forward(images)
        else:
            if self.path_predict:
                return self.predict(images)
            else:
                return self._forward(images)

    def _forward(self, images: torch.Tensor):
        batch_size = images.size(0)
        image_features = self.extract_features(images)
        data_list = []
        for i in range(batch_size):
            label_embeddings = self.prototypes.weight
            x_subgraph = torch.cat(
                [image_features[i].unsqueeze(0), label_embeddings], dim=0
            )
            data_list.append(
                Data(
                    x=x_subgraph,
                    edge_index=self.edge_index_global_context,
                )
            )
        data_batch = Batch.from_data_list(data_list)

        node_embeddings = self.gnn(
            data_batch.x, data_batch.edge_index, batch=data_batch.batch
        )
        node_embeddings = self.dropout(node_embeddings)
        node_logits = self.classifier(node_embeddings)

        return node_logits

    def enable_dropout(self):
        for module in self.gnn.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.inference_mode()
    def predict(self, images):
        logits = self._forward(images)
        probabilities = F.sigmoid(logits)
        return self._get_best_path(probabilities)

    @torch.inference_mode()
    def _get_best_path(
        self,
        probabilities: torch.Tensor,
        uncertainties: torch.Tensor = None,
        start_node: int = 0,
        prob_threshold: float = 0.0,
        unc_penalty: float = 0.0,
    ):
        batch_size = probabilities.size(0)
        paths = torch.zeros(
            batch_size, self.num_nodes, dtype=torch.float32, device=probabilities.device
        )
        current_nodes = torch.full(
            (batch_size,),
            fill_value=start_node,
            device=probabilities.device,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=probabilities.device)

        i = 0
        while active.any() and i <= probabilities.shape[1]:
            # Mark current nodes in the paths
            paths[
                torch.arange(batch_size, device=probabilities.device)[active],
                current_nodes[active],
            ] = 1

            # Create one-hot encoding of current nodes
            current_node_mask = torch.zeros(
                batch_size, self.num_nodes, device=probabilities.device
            )
            current_node_mask[
                torch.arange(batch_size, device=probabilities.device), current_nodes
            ] = 1

            # Use adjacency matrix to find neighbors
            neighbor_scores = current_node_mask @ self.adjacency_matrix
            neighbor_probs = neighbor_scores * probabilities

            # Check uncertainties
            if uncertainties is not None:
                # Select the most probable neighbors
                neighbor_uncs = neighbor_scores * uncertainties
                next_nodes = torch.argmax(
                    torch.clamp(neighbor_probs - unc_penalty * neighbor_uncs, min=0),
                    dim=1,
                )  # Shape: [batch_size]
            else:
                # Select the most probable neighbors
                next_nodes = torch.argmax(neighbor_probs, dim=1)  # Shape: [batch_size]

            # Update
            next_probs = torch.max(neighbor_probs, dim=1).values  # Shape: [batch_size]
            active = (neighbor_probs.sum(dim=1) > 0) & (next_probs >= prob_threshold)
            current_nodes = next_nodes
            i += 1

        return paths

    @torch.inference_mode()
    def _get_best_path_beam(
        self,
        probabilities: torch.Tensor,
        uncertainties: torch.Tensor = None,
        start_node: int = 0,
        prob_threshold: float = 0.0,
        unc_penalty: float = 0.0,
        beam_width: int = 3,
        max_depth: int = None,
    ):
        batch_size = probabilities.size(0)
        num_nodes = self.num_nodes

        if max_depth is None:
            max_depth = num_nodes

        best_paths = [None] * batch_size

        for b in range(batch_size):
            prob_vec = probabilities[b]  # shape: (num_nodes,)
            unc_vec = uncertainties[b] if uncertainties is not None else None
            beam = [{"path": [start_node], "score": 0.0, "current_node": start_node}]

            depth = 0
            while depth < max_depth:
                new_beam = []
                for candidate in beam:
                    current_node = candidate["current_node"]
                    neighbor_mask = self.adjacency_matrix[
                        current_node
                    ]  # shape: (num_nodes,)
                    for next_node in range(num_nodes):
                        if neighbor_mask[next_node] > 0:  # valid neighbor
                            prob_val = prob_vec[next_node].item()
                            if prob_val < prob_threshold:
                                continue
                            if unc_vec is not None:
                                unc_val = unc_vec[next_node].item()
                                increment = prob_val - unc_penalty * unc_val
                            else:
                                increment = prob_val

                            new_candidate = {
                                "path": candidate["path"] + [next_node],
                                "score": candidate["score"] + increment,
                                "current_node": next_node,
                            }
                            new_beam.append(new_candidate)
                if not new_beam:
                    break
                new_beam.sort(key=lambda x: x["score"], reverse=True)
                beam = new_beam[:beam_width]
                depth += 1

            best_candidate = (
                max(beam, key=lambda x: x["score"]) if beam else {"path": [start_node]}
            )
            best_paths[b] = best_candidate["path"]

        paths_tensor = torch.zeros(
            batch_size, num_nodes, dtype=torch.float32, device=probabilities.device
        )
        for b, path in enumerate(best_paths):
            for node in path:
                paths_tensor[b, node] = 1

        return paths_tensor
