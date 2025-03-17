from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional

def create_scheduler(optimizer: Optimizer, config: Dict[str, Any], total_steps: Optional[int] = None):
    """创建学习率调度器"""
    scheduler_type = config.get('scheduler_type', 'cosine').lower()
    
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.get('warmup_steps', 0),
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.get('warmup_steps', 0),
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine_annealing':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get('t_max', total_steps),
            eta_min=config.get('min_lr', 0)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_reduce_factor', 0.1),
            patience=config.get('lr_patience', 10),
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建线性学习率调度器，带预热
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    创建余弦学习率调度器，带预热
    """
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch) 