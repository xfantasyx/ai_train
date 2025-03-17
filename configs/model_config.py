"""模型配置"""

def get_model_config(model_size: str = "base"):
    """获取模型配置"""
    
    # 基础配置
    base_config = {
        # 模型基本信息
        "model_name_or_path": "baichuan-inc/Baichuan2-7B-Base",  # 或其他适合的基础模型
        "tokenizer_name_or_path": "baichuan-inc/Baichuan2-7B-Base",
        
        # 是否从预训练模型初始化
        "from_pretrained": True,
        
        # 模型结构配置
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
        
        # 词汇表配置
        "vocab_size": 125696,
        
        # 特殊token
        "special_tokens": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<程序>", "<代码>", "<解释>"]
        }
    }
    
    # 根据模型大小调整配置
    if model_size == "small":
        small_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 5504
        }
        base_config.update(small_config)
    elif model_size == "large":
        large_config = {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "intermediate_size": 13696
        }
        base_config.update(large_config)
    
    return base_config 