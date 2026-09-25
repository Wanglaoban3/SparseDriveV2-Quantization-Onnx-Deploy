from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class SparseDriveConfig:
    """Global SparseDrive config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    # ================ model ================ #
    # basic setting
    d_model: int = 256
    d_ffn: int = 1024
    num_head: int = 8
    dropout: float = 0.0

    # vision backbone & neck
    image_architecture: str = "resnet34"
    bkb_path: str = "ckpt/resnet34.bin"    
    use_grid_mask: bool = True
    with_img_neck: bool = True
    num_levels: int = 4

    # vocabulary
    path_anchor: str = "ckpt/kmeans/path_1024.npy"
    velocity_anchor: str = "ckpt/kmeans/velocity_256.npy"
    trajectory_anchor: str = "ckpt/kmeans/trajectory_1024_256.npz"

    mode_path: int = 1024
    len_path: int = 50
    path_interval: float = 1.0
    mode_vel: int = 256
    len_vel_seq: int = 8
    vel_time_interval: float = 0.5

    ## decoder
    decoder_num_layers: int = 2

    path_filter_num: List[int] = (128, 20)
    velocity_filter_num: List[int] = (64, 10)

    path_sigmas: float = 4.0
    velocity_sigmas: float = 4.0
    trajectory_sigmas: float = 4.0

    # deformable
    fix_height: List[float] = (0., -0.25, -0.5, 0.25, 0.5)
    num_learnable_pts: int = 2

    # metric supervision
    dataset_version: str = "v2"
    metrics: List[str] = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
                          "time_to_collision_within_bound", "ego_progress", "lane_keeping", "history_comfort")
    metric_loss_weight: float = 5.0

    # ================ reward-aligned fine-tuning (stage 0 / stage 1) ================ #
    # Master switch: requires an IL checkpoint, freezes most of the network and adds
    # rule-scorer-reward losses (see rl_finetune.py). Requires metric caches (token_path).
    rl_finetune: bool = False
    # Which parameters stay trainable: "scorer_heads" (traj_mlp + metric_heads of the last
    # decoder layer) or "traj_branch" (additionally the t_* stack of the last layer).
    train_scope: str = "scorer_heads"
    # Score used to pick the final trajectory at inference: "metric" (composed metric logits,
    # unchanged default) or "traj" (traj_scores, which stage 0 aligns to the reward).
    selection_score: str = "metric"
    # Stage 0a: CE(traj_scores, softmax(-tau * (1 - EPDMS))).
    reward_ce_weight: float = 1.0
    reward_tau: float = 10.0
    # Stage 0b: BCE(composed metric score, EPDMS) — trains the deployed selection path.
    composition_loss_weight: float = 1.0
    # Stage 1: group-relative policy gradient; 0 disables. KL anchor uses a frozen copy of the
    # scoring branch taken at fine-tuning start (the IL policy).
    grpo_weight: float = 0.0
    grpo_num_samples: int = 0  # group size K; 0 = all final candidates (exact, no sampling)
    grpo_kl_weight: float = 0.5
    keep_il_loss: bool = True  # keep the imitation CE on traj_scores as anchor
    # token_path mapping, for caches whose directory names differ from the navtrain default
    data_cache_prefix: str = "data_cache_navtrain"
    metric_cache_prefix: str = "metric_cache_navtrain"

    # ================ data process ================ #
    cams: List[str] = ("cam_l0", "cam_f0", "cam_r0")
    resize_lim: List[float] = (512/1920, 512/1920)
    final_dim: List[float] = (256, 512)
    bot_pct_lim: List[float] = (0.0, 0.0)
    rot_lim: List[float] = (-0., 0.)
    H: int = 1080
    W: int = 1920
    rand_flip: bool = False
    rot3d_range: List[float] = (-0., 0.)
    photo_metric_distortion: bool = True
    img_mean: List[float] = (123.675, 116.28, 103.53)
    img_std: List[float] = (58.395, 57.12, 57.375)
    to_bgr: bool = False

