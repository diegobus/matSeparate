import numpy as np
import torch

from hgnn import HGNN
from loss import greedy_loss
from taxonomy.tree import (all_paths_subgraph, get_hierarchy_levels,
                           get_taxonomy, level_complete_tree, taxa_to_indices,
                           taxa_to_onehot, write_network_text_with_color)

# knobs
TAXONOMY = get_taxonomy()
COMPLETE_GRAPH = False
EXAMPLE_BACKBONES = [
    "resnet18",
    "resnet18.fb_swsl_ig1b_ft_in1k",
    "resnet50",
    "resnet50.fb_swsl_ig1b_ft_in1k",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_xl.in21k_ft_in1k",
    "convnext_xlarge.fb_in22k_ft_in1k_384",
    "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
    "densenet161.tv_in1k",
    "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
]
GRAYSCALE_IMAGES = False
IS_TRANSFORMER = False



# hierarchy data loading
labels_with_data = ["iron", "moss", "granite", "dirt", "fur"]
TAXONOMY = (
    all_paths_subgraph(TAXONOMY, "root", labels_with_data)
    if not COMPLETE_GRAPH
    else level_complete_tree(
        all_paths_subgraph(TAXONOMY, "root", labels_with_data),
        root="root",
        directed=True,
    )
)
TAXONOMY_LEVELS = get_hierarchy_levels(TAXONOMY)
NUM_TAXONOMY_LEVELS = len(TAXONOMY_LEVELS)
HIERARCHY_LEVELS = [
    taxa_to_indices(TAXONOMY_LEVELS[level], TAXONOMY)
    for level in sorted(TAXONOMY_LEVELS.keys())
]
print(
    "Hierarchical Tree:\n" + write_network_text_with_color(TAXONOMY, labels_with_data)
)


# model
model = HGNN(
    graph=TAXONOMY,
    cnn_kwargs={
        "backbone": EXAMPLE_BACKBONES[0],
        "pretrained": True,
        "finetune": True,
        "output_dim": 1024,
        "in_channels": 1 if GRAYSCALE_IMAGES else 3,
        "is_transformer": IS_TRANSFORMER,
    },
    gnn_kwargs={
        "input_dim": 1024,
        "hidden_dim": 512,
        "output_dim": 256,
        "num_layers": 2,
        "skip_connection": True,
    },
)
if IS_TRANSFORMER and hasattr(model.cnn.cnn, "fc_norm"):
    delattr(model.cnn.cnn, "fc_norm")


# dummy data
img = torch.rand(5, 3, 128, 128)
gt = np.array(labels_with_data)
labels_multihot = torch.tensor(
    taxa_to_onehot(
        gt,
        TAXONOMY,
        maxlevel=NUM_TAXONOMY_LEVELS,  
        full_path=True,
    ),
    dtype=torch.float32
)

out = model(img)
loss = greedy_loss(out, labels_multihot, HIERARCHY_LEVELS, mode="combined")
print(loss)



#####
# Optimizer and scheduler setup
#####
# INIT_LR = 1e-3
# EPS_LR = 1e-8

# # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
# cnn_params = model.cnn.parameters()
# classifier_params = model.classifier.parameters()
# muon_params = [p for p in cnn_params if p.ndim >= 2]
# adamw_params = [p for p in cnn_params if p.ndim < 2]
# adamw_params.extend(classifier_params)
# gnn_params = model.gnn.parameters()
# projection_params = model.projection.parameters()
# prototypes_params = model.prototypes.parameters()
# muon_params.extend([p for p in gnn_params if p.ndim >= 2])
# adamw_params.extend([p for p in gnn_params if p.ndim < 2])
# adamw_params.extend(projection_params)
# adamw_params.extend(prototypes_params)

# # Create the optimizer and scheduler
# optimizer_muon = Muon(
#     muon_params,
#     lr=INIT_LR,
#     momentum=0.95,
#     adamw_params=adamw_params,
#     adamw_lr=INIT_LR,
#     adamw_betas=(0.90, 0.95),
#     adamw_eps=EPS_LR,
# )
# optimizers = [optimizer_muon]
# num_steps = len(train_loader) * epochs
# schedulers = [
#     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=num_steps)
#     for opt in optimizers
# ]