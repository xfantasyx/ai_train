"""训练配置"""

def get_train_config(batch_size: int = 8, epochs: int = 3):
    """获取训练配置"""
    
    config = {
        # 数据配置
        "train_data_path": "./data/train",
        "val_data_path": "./data/val",
        "test_data_path": "./data/test",
        "max_length": 2048,
        
        # 训练参数
        "batch_size": batch_size,
        "num_workers": 4,
        "epochs": epochs,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        
        # 优化器配置
        "optimizer_type": "adamw",
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        
        # 学习率调度器
        "scheduler_type": "cosine",
        "use_layer_lr_decay": True,
        "lr_layer_decay_rate": 0.9,
        
        # 混合精度训练
        "use_amp": True,
        
        # 梯度累积
        "gradient_accumulation_steps": 4,
        
        # 检查点保存
        "checkpoint_dir": "./checkpoints",
        "save_every": 1,
        
        # 日志配置
        "use_wandb": True,
        "wandb_project": "domain-llm",
        "wandb_run_name": "domain-llm-training",
        
        # 冻结层
        "freeze_layers": False,  # 或指定要冻结的层数/列表
        
        # 评估配置
        "eval_steps": 500,
        "eval_batch_size": 16,
        
        # 生成配置
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.7,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "max_output_length": 1024
    }
    
    return config 