"""Reward-aligned fine-tuning (stage 0 / stage 1) for the vocab scorer.

Stage 0 (supervised reward distillation, not RL):
    - reward_ce_loss:   CE(traj_scores, softmax(-tau * (1 - EPDMS))) — the imitation target of
                        traj_scores is replaced by (or anchored to) the rule-scorer ranking.
    - composition_loss: pulls the deployed selection score (composed from metric logits) toward
                        the official EPDMS of each candidate.

Stage 1 (offline policy gradient / GRPO):
    - grpo_loss:        treats softmax(traj_scores) as the policy over the final candidate set,
                        computes group-normalized advantages from the rule-scorer reward and adds
                        an optional KL anchor to the frozen IL policy (ref scores).

All rewards come from the cached rule scorer (get_pdm_score_v1/v2), which reports the official
composed score per candidate under the "pdm_score" key — the environment itself is the
NAVSIM metric cache, no simulator interaction is involved.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def rewards_from_sub_scores(sub_scores, device: torch.device) -> torch.Tensor:
    """Official EPDMS per candidate from the rule scorer output.

    :param sub_scores: list over the batch of dicts metric name -> (G,) float array,
                       as returned by get_pdm_score_v1/v2; the composed score is "pdm_score".
    :return: (B, G) float32 tensor of rewards in [0, 1].
    """
    reward = np.stack([np.asarray(ss["pdm_score"], dtype=np.float32) for ss in sub_scores])
    return torch.from_numpy(reward).to(device, non_blocking=True)


def reward_ce_loss(traj_scores: torch.Tensor, reward: torch.Tensor, tau: float) -> torch.Tensor:
    """Stage 0: distill the rule-scorer ranking into traj_scores.

    Soft label = softmax(-tau * (1 - reward)): tau controls how peaked the target is —
    tau -> 0 keeps only the relative order, tau -> inf approaches a one-hot argmax target.
    """
    logp = F.log_softmax(traj_scores, dim=-1)
    with torch.no_grad():
        soft = F.softmax(-tau * (1.0 - reward), dim=-1)
    return -(soft * logp).sum(-1).mean()


def composition_loss(metric_logit: Dict[str, torch.Tensor], reward: torch.Tensor, dataset_version: str) -> torch.Tensor:
    """Stage 0: align the deployed selection score with the official reward.

    Recomputes the same composed score used at inference (product of hard-gate sigmoids times
    the weighted sum of soft-metric sigmoids, normalized to [0, 1]) and regresses it to the
    rule-scorer EPDMS. Differentiable w.r.t. the metric heads, so the actual inference path
    is trained directly.
    """
    sig = torch.sigmoid
    if dataset_version == "v1":
        gates = sig(metric_logit["no_at_fault_collisions"]) * sig(metric_logit["drivable_area_compliance"])
        weighted = (
            5.0 * sig(metric_logit["time_to_collision_within_bound"])
            + 5.0 * sig(metric_logit["ego_progress"])
            + 2.0 * sig(metric_logit["comfort"])
        )
        total_weight = 12.0
    else:
        gates = (
            sig(metric_logit["no_at_fault_collisions"])
            * sig(metric_logit["drivable_area_compliance"])
            * sig(metric_logit["driving_direction_compliance"])
            * sig(metric_logit["traffic_light_compliance"])
        )
        weighted = (
            5.0 * sig(metric_logit["time_to_collision_within_bound"])
            + 5.0 * sig(metric_logit["ego_progress"])
            + 2.0 * sig(metric_logit["lane_keeping"])
            + 2.0 * sig(metric_logit["history_comfort"])
        )
        total_weight = 14.0
    composed = (gates * weighted / total_weight).clamp(1e-6, 1.0 - 1e-6)
    # BCE via logits: F.binary_cross_entropy is unsafe under AMP autocast (16-mixed training)
    return F.binary_cross_entropy_with_logits(torch.logit(composed), reward)


def grpo_loss(
    traj_scores: torch.Tensor,
    reward: torch.Tensor,
    num_samples: int = 0,
    ref_scores: Optional[torch.Tensor] = None,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """Stage 1: group-relative policy gradient over the vocab policy pi = softmax(traj_scores).

    :param traj_scores: (B, G) logits of the current policy.
    :param reward: (B, G) rule-scorer EPDMS of the same candidates.
    :param num_samples: group size K; 0 uses all G candidates (no sampling, exact expectation).
                        With 0 < K < G the group is sampled from the current policy — sampling
                        is the exploration mechanism and bounds the reward-scoring cost.
    :param ref_scores: (B, G) logits of the frozen reference (IL) policy for the KL anchor.
    :param kl_weight: beta of the KL(pi || pi_ref) anchor; prevents drift toward reward hacking
                      and keeps decent-but-suppressed candidates alive.
    """
    logp = F.log_softmax(traj_scores, dim=-1)
    if num_samples and num_samples < traj_scores.shape[1]:
        with torch.no_grad():
            idx = torch.multinomial(logp.detach().exp(), num_samples, replacement=True)
        logp = logp.gather(1, idx)
        reward = reward.gather(1, idx)
    with torch.no_grad():
        mean = reward.mean(dim=1, keepdim=True)
        std = reward.std(dim=1, keepdim=True).clamp_min(1e-6)
        advantage = (reward - mean) / std
    loss = -(advantage * logp).sum(-1).mean() / logp.shape[1]
    if ref_scores is not None and kl_weight > 0:
        ref_logp = F.log_softmax(ref_scores, dim=-1)
        kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
        loss = loss + kl_weight * kl
    return loss
