export HYDRA_FULL_ERROR=1

# Reward-aligned fine-tuning (stage 0/1) on top of the IL checkpoint.
# Stage 0: reward-CE on traj_scores + BCE(composed metric score, EPDMS)  -> reward_ce_weight / composition_loss_weight
# Stage 1: GRPO policy gradient with KL anchor to the IL policy          -> grpo_weight / grpo_kl_weight
# Both stages use the official EPDMS of every final candidate from the metric caches;
# run the metric caching for navtrain first (scripts/cache/run_metric_caching_navtrain_v1.sh).

config=default_training
agent=sparsedrive_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name $config \
    agent=$agent \
    experiment_name=sparsedrive_rl_s0s1 \
    train_test_split=navtrain \
    use_cache_without_dataset=True  \
    force_cache_computation=False \
    cache_path=exp/data_cache_navtrain \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=16 \
    dataloader.params.prefetch_factor=4 \
    trainer.params.max_epochs=2 \
    agent.lr=0.00001 \
    agent.checkpoint_path=ckpt/sparsedrive_navsimv1_92p2.ckpt \
    +agent.config.dataset_version=v1 \
    +agent.config.metrics=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","time_to_collision_within_bound","comfort","ego_progress"] \
    +agent.config.velocity_filter_num=[64,20] \
    +agent.config.rl_finetune=true \
    +agent.config.train_scope=scorer_heads \
    +agent.config.reward_ce_weight=1.0 \
    +agent.config.reward_tau=10.0 \
    +agent.config.composition_loss_weight=1.0 \
    +agent.config.grpo_weight=1.0 \
    +agent.config.grpo_num_samples=0 \
    +agent.config.grpo_kl_weight=0.5
