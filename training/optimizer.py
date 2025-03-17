import torch
from torch.optim import AdamW, Optimizer
from typing import Dict, Any, List, Optional

def create_optimizer(model_params, config: Dict[str, Any]) -> Optimizer:
    """创建优化器"""
    optimizer_type = config.get('optimizer_type', 'adamw').lower()
    
    if optimizer_type == 'adamw':
        return AdamW(
            model_params,
            lr=config.get('learning_rate', 5e-5),
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            model_params,
            lr=config.get('learning_rate', 5e-5),
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0)
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model_params,
            lr=config.get('learning_rate', 0.01),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_param_groups(model, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """创建参数组，可以为不同层设置不同的学习率"""
    no_decay = ['bias', 'LayerNorm.weight']
    
    # 基础学习率
    base_lr = config.get('learning_rate', 5e-5)
    
    # 学习率衰减因子（从底层到顶层）
    lr_decay = config.get('lr_layer_decay_rate', 0.9)
    
    # 获取所有层
    layers = [(name, param) for name, param in model.named_parameters()]
    
    # 按层分组
    groups = []
    
    # 如果使用层衰减
    if config.get('use_layer_lr_decay', False):
        num_layers = model.config.num_hidden_layers
        
        for layer_idx in range(num_layers + 1):  # +1 for embeddings
            # 计算该层的学习率
            layer_lr = base_lr * (lr_decay ** (num_layers - layer_idx))
            
            # 该层的参数（区分权重衰减和非权重衰减）
            if layer_idx == 0:
                # Embeddings
                layer_params_decay = [
                    p for n, p in layers 
                    if 'embed' in n and not any(nd in n for nd in no_decay) and p.requires_grad
                ]
                layer_params_no_decay = [
                    p for n, p in layers 
                    if 'embed' in n and any(nd in n for nd in no_decay) and p.requires_grad
                ]
            else:
                # Transformer层
                layer_params_decay = [
                    p for n, p in layers 
                    if f'layer.{layer_idx-1}' in n and not any(nd in n for nd in no_decay) and p.requires_grad
                ]
                layer_params_no_decay = [
                    p for n, p in layers 
                    if f'layer.{layer_idx-1}' in n and any(nd in n for nd in no_decay) and p.requires_grad
                ]
            
            # 添加到组
            if layer_params_decay:
                groups.append({
                    'params': layer_params_decay,
                    'lr': layer_lr,
                    'weight_decay': config.get('weight_decay', 0.01)
                })
            
            if layer_params_no_decay:
                groups.append({
                    'params': layer_params_no_decay,
                    'lr': layer_lr,
                    'weight_decay': 0.0
                })
    else:
        # 简单分组（只区分权重衰减和非权重衰减）
        groups = [
            {
                'params': [p for n, p in layers if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in layers if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
    
    return groups 