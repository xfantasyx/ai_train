import os
import sys
import argparse
import json
import logging
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_model_config
from config.train_config import get_train_config
from model.architecture import DomainLLM
from model.tokenizer import DomainTokenizer
from data.dataloader import create_dataloaders
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_param_groups
from training.scheduler import create_scheduler
from utils.logging import setup_logging
from utils.visualization import plot_training_history

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练领域LLM")
    
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                        help="模型大小: small, base, large")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    
    parser.add_argument("--train_data", type=str, required=True,
                        help="训练数据路径")
    
    parser.add_argument("--val_data", type=str, required=True,
                        help="验证数据路径")
    
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志目录")
    
    parser.add_argument("--use_amp", action="store_true",
                        help="是否使用混合精度训练")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用Weights & Biases记录训练过程")
    
    parser.add_argument("--wandb_project", type=str, default="domain-llm",
                        help="Weights & Biases项目名称")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("训练参数:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    model_config = get_model_config(args.model_size)
    train_config = get_train_config(args.batch_size, args.epochs)
    
    # 更新配置
    train_config.update({
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'learning_rate': args.learning_rate,
        'use_amp': args.use_amp,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints')
    })
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
    
    # 初始化分词器
    logger.info("初始化分词器...")
    tokenizer = DomainTokenizer(model_config)
    
    # 初始化模型
    logger.info("初始化模型...")
    model = DomainLLM(model_config)
    
    # 加载检查点（如果提供）
    if args.checkpoint:
        logger.info(f"从检查点加载: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    dataloaders = create_dataloaders(train_config, tokenizer.tokenizer)
    
    # 创建参数组
    param_groups = create_param_groups(model, train_config)
    
    # 创建优化器
    optimizer = create_optimizer(param_groups, train_config)
    
    # 计算总步数
    total_steps = len(dataloaders['train']) * train_config['epochs'] // train_config['gradient_accumulation_steps']
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, train_config, total_steps)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        config=train_config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 训练模型
    logger.info("开始训练...")
    history = trainer.train(train_config['epochs'])
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config
    }, final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 绘制训练历史
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main() 