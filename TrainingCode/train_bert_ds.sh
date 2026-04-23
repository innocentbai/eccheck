deepspeed --bind_cores_to_rank train_bert_ds.py --checkpoint_dir experiment_deepspeed $@
# --bind_cores_to_rank: 启用 CPU 核心绑定 ，确保每个进程的线程在固定的核心上运行（减少资源争用）