export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=ckpt/sparsedrive_navsimv2.ckpt
CACHE_PATH=exp/metric_cache_navhard
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navhard_fast.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=sparsedrive_agent \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=sparsedrive_agent \
    metric_cache_path=$CACHE_PATH \
    +test_cache_path=${NAVSIM_EXP_ROOT}/data_cache_navhard/ \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    dataloader.params.batch_size=8
