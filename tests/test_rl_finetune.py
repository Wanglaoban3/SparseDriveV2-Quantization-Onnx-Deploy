import torch

from navsim.agents.sparsedrive.rl_finetune import (
    composition_loss,
    grpo_loss,
    reward_ce_loss,
    rewards_from_sub_scores,
)

torch.manual_seed(0)
B, G = 2, 200

# fake rule-scorer output: per-candidate EPDMS in [0,1], some crashed (0), some clean
rewards_np = [torch.rand(G).numpy() * 0.3 + 0.7 for _ in range(B)]
rewards_np[0][:50] = 0.0
sub_scores = [{"pdm_score": r.astype("float16")} for r in rewards_np]
reward = rewards_from_sub_scores(sub_scores, torch.device("cpu"))
assert reward.shape == (B, G) and reward.dtype == torch.float32
assert reward[0, 0] == 0.0 and reward[0, 50] > 0.7

# --- reward_ce_loss: target peaked on high-reward candidates; gradient flows ---
logits = torch.zeros(B, G, requires_grad=True)
loss = reward_ce_loss(logits, reward, tau=10.0)
loss.backward()
with torch.no_grad():
    soft = torch.softmax(-10.0 * (1.0 - reward), dim=-1)
    # controlled rewards: one candidate at 1.0, rest at 0.8 -> target ratio e^2 ~ 7.4x
    ctrl_reward = torch.zeros(1, G)
    ctrl_reward[0, 0] = 1.0
    ctrl_reward[0, 1:] = 0.8
    ctrl_soft = torch.softmax(-10.0 * (1.0 - ctrl_reward), dim=-1)
    assert ctrl_soft[0, 0] > ctrl_soft[0, 1] * 3
    # log-softmax gradient of CE with soft targets = softmax(logits) - target; at zero logits = (1/G - target)
    grad = logits.grad
    expected = torch.softmax(logits.detach(), -1) - soft
    assert torch.allclose(grad, expected / B, atol=1e-6)
print(f"reward_ce_loss ok: loss={loss.item():.4f}, target_max={soft.max().item():.4f}")

# --- grpo_loss: full-group == exact expectation; zero KL when ref == current ---
logits = (torch.randn(B, G) * 0.1).requires_grad_(True)
ref = logits.detach().clone()
loss_pg = grpo_loss(logits, reward, num_samples=0, ref_scores=ref, kl_weight=0.5)
assert torch.allclose(loss_pg, grpo_loss(logits, reward, num_samples=G, ref_scores=ref, kl_weight=0.5))
loss_pg.backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all()

# advantages are zero-mean per scene -> full-group PG gradient ~ covariance term, must be finite
# zero-advantage case (all rewards equal) gives exactly zero PG loss
flat_reward = torch.full((B, G), 0.8)
loss_flat = grpo_loss(logits.detach().requires_grad_(True), flat_reward, num_samples=0, ref_scores=ref, kl_weight=0.5)
assert loss_flat.item() >= 0.0 and loss_flat.item() < 1e-4, loss_flat.item()  # only KL(pi||ref)>0 remains... ref==logits -> 0
assert abs(loss_flat.item()) < 1e-6
print(f"grpo_loss ok: pg={loss_pg.item():.6f}, flat-case={loss_flat.item():.2e}")

# KL anchoring actually pulls: current near-one-hot vs uniform ref -> KL(one-hot||uniform) = log(G)
one_hot_logits = torch.full((B, G), -10.0)
one_hot_logits[:, 0] = 10.0
loss_far = grpo_loss(one_hot_logits, reward, num_samples=0, ref_scores=torch.zeros(B, G), kl_weight=0.5)
assert loss_far.item() > 2.0  # 0.5 * log(200) ~ 2.65
print(f"grpo KL anchor ok: KL term={loss_far.item():.4f} (log(G)/2={0.5*torch.log(torch.tensor(float(G))).item():.4f})")

# --- composition_loss: mirrors the deployed v2 formula ---
metric_logit = {m: torch.zeros(B, G, requires_grad=True) for m in [
    "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "traffic_light_compliance", "time_to_collision_within_bound", "ego_progress",
    "lane_keeping", "history_comfort"]}
loss_c = composition_loss(metric_logit, reward, "v2")
loss_c.backward()
sig = torch.sigmoid
expected_composed = (
    sig(metric_logit["no_at_fault_collisions"]) * sig(metric_logit["drivable_area_compliance"])
    * sig(metric_logit["driving_direction_compliance"]) * sig(metric_logit["traffic_light_compliance"])
) * (
    5 * sig(metric_logit["time_to_collision_within_bound"]) + 5 * sig(metric_logit["ego_progress"])
    + 2 * sig(metric_logit["lane_keeping"]) + 2 * sig(metric_logit["history_comfort"])
) / 14.0
manual = torch.nn.functional.binary_cross_entropy(expected_composed.clamp(1e-6, 1 - 1e-6), reward)
assert torch.allclose(loss_c, manual)
for m in metric_logit:
    assert metric_logit[m].grad is not None
print(f"composition_loss ok: loss={loss_c.item():.4f} (all-zero logits, gates=1 -> BCE vs mean reward {reward.mean().item():.3f})")

print("ALL RL FINETUNE NUMERICAL CHECKS PASSED")
