import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os

def plot_training_history(history: Dict[str, List[float]], output_path: Optional[str] = None):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='训练损失')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制困惑度
    plt.subplot(1, 2, 2)
    if 'train_loss' in history:
        train_perplexity = [np.exp(loss) for loss in history['train_loss']]
        plt.plot(train_perplexity, label='训练困惑度')
    if 'val_loss' in history:
        val_perplexity = [np.exp(loss) for loss in history['val_loss']]
        plt.plot(val_perplexity, label='验证困惑度')
    plt.title('困惑度曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"图表已保存到 {output_path}")
    
    plt.show()

def plot_attention_weights(attention_weights: np.ndarray, tokens: List[str], output_path: Optional[str] = None):
    """绘制注意力权重热图"""
    plt.figure(figsize=(10, 8))
    
    plt.imshow(attention_weights, cmap='viridis')
    
    # 设置刻度标签
    plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    plt.yticks(np.arange(len(tokens)), tokens)
    
    plt.colorbar()
    plt.title('注意力权重')
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"热图已保存到 {output_path}")
    
    plt.show()

def plot_metrics_comparison(metrics_list: List[Dict[str, float]], 
                           model_names: List[str], 
                           metric_names: List[str],
                           output_path: Optional[str] = None):
    """绘制多个模型的指标比较"""
    n_metrics = len(metric_names)
    n_models = len(model_names)
    
    plt.figure(figsize=(12, 4 * n_metrics))
    
    for i, metric in enumerate(metric_names):
        plt.subplot(n_metrics, 1, i+1)
        
        values = [metrics.get(metric, 0) for metrics in metrics_list]
        
        plt.bar(model_names, values)
        plt.title(f'{metric} 比较')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱状图上显示数值
        for j, v in enumerate(values):
            plt.text(j, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"比较图表已保存到 {output_path}")
    
    plt.show() 