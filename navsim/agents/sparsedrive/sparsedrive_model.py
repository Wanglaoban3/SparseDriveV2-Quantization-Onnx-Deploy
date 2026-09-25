from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.common.enums import StateSE2Index

from .sparsedrive_config import SparseDriveConfig
from .sparsedrive_backbone import SparseBackbone
from .custom_decoder import CustomTransformerDecoder


class SparseDriveModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: SparseDriveConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._config = config
        self._backbone = SparseBackbone(config)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.d_model)
        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.d_ffn,
            d_model=config.d_model,
            config=config,
        )

    def apply_rl_finetune(self):
        """Freeze everything except the scoring modules selected by config.train_scope.

        Call AFTER loading the IL checkpoint. Frozen submodules are kept in eval mode by the
        train() override below, so BatchNorm running stats stay fixed and dropout inside frozen
        branches is disabled (deterministic features for the scorer heads to fit).
        """
        cfg = self._config
        if not cfg.rl_finetune:
            return
        if cfg.train_scope not in ("scorer_heads", "traj_branch"):
            raise ValueError(f"Unknown train_scope: {cfg.train_scope}")

        last = f"_trajectory_head.decoder.layers.{cfg.decoder_num_layers - 1}."
        if cfg.train_scope == "scorer_heads":
            trainable_prefixes = (last + "traj_mlp", last + "metric_heads")
        else:
            trainable_prefixes = (last + "t_", last + "traj_mlp", last + "metric_heads")

        for param in self.parameters():
            param.requires_grad_(False)
        for name, param in self.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad_(True)

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        if self._config.rl_finetune and mode:
            # pl.Trainer calls train() every epoch; re-apply eval to frozen submodules so
            # dropout inside them stays off and BN stats stay fixed
            last = f"_trajectory_head.decoder.layers.{self._config.decoder_num_layers - 1}."
            if self._config.train_scope == "scorer_heads":
                trainable_prefixes = (last + "traj_mlp", last + "metric_heads")
            else:
                trainable_prefixes = (last + "t_", last + "traj_mlp", last + "metric_heads")
            for name, module in self.named_modules():
                if name and not name.startswith(trainable_prefixes):
                    module.eval()
        return self

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]
        status_encoding = self._status_encoding(status_feature)

        imgs = camera_feature["imgs"]
        feature_maps = self._backbone(imgs)
        camera_feature["feature_maps"] = feature_maps

        output = {}
        trajectory, loss_dict = self._trajectory_head(camera_feature, status_encoding, targets)
        output.update(trajectory)

        if self.training:
            loss_dict["loss"] = sum(loss_dict.values())
        
        return output, loss_dict


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, config: SparseDriveConfig = None):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self.path_vocab = nn.Parameter(
            torch.from_numpy(np.load(config.path_anchor)).float(),
            requires_grad=False
        )
        self.vel_vocab = nn.Parameter(
            torch.from_numpy(np.load(config.velocity_anchor)).float(),
            requires_grad=False
        )
        trajectory_data = np.load(config.trajectory_anchor)
        self.traj_vocab = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory"]).float(),
            requires_grad=False
        )
        self.traj_mask = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory_mask"]).float(),
            requires_grad=False
        )

        self.path_pos_embed = nn.Sequential(
            nn.Linear(config.len_path * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.vel_pos_embed = nn.Sequential(
            nn.Linear(config.len_vel_seq, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        self.decoder = CustomTransformerDecoder(num_poses, d_model, d_ffn, config)

    def forward(self, camera_feature, status_encoding, targets) -> Dict[str, torch.Tensor]:
        B = status_encoding.shape[0]

        path_vocab = self.path_vocab.data[None].repeat(B, 1, 1, 1)
        vel_vocab = self.vel_vocab.data[None].repeat(B, 1, 1)
        traj_vocab = self.traj_vocab.data[None].repeat(B, 1, 1, 1, 1)
        traj_mask = self.traj_mask.data[None].repeat(B, 1, 1, 1)

        path_embed = self.path_pos_embed(path_vocab.flatten(-2, -1))
        vel_embed = self.vel_pos_embed(vel_vocab)

        decoder_outputs = self.decoder(
            (path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask),
            (camera_feature, status_encoding, targets),
        )

        return decoder_outputs