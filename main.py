import os
import argparse
import torch
from typing import Dict, Any

from config.model_config import get_model_config
from config.train_config import get_train_config
from model.architecture import DomainLLM
from model.tokenizer import DomainTokenizer
from data.dataloader import create_dataloaders
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_param_groups
from training.scheduler import create_scheduler
from evaluation.evaluator import Evaluator
from inference.engine import InferenceEngine
from inference.serving import ModelServer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="领域LLM训练与推理")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "infer", "serve"],
                        help="运行模式: train, eval, infer, serve")
    
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                        help="模型大小: small, base, large")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    model_config = get_model_config(args.model_size)
    train_config = get_train_config(args.batch_size, args.epochs)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分词器
    print("初始化分词器...")
    tokenizer = DomainTokenizer(model_config)
    
    # 初始化模型
    print("初始化模型...")
    model = DomainLLM(model_config)
    
    # 加载检查点（如果提供）
    if args.checkpoint:
        print(f"从检查点加载: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 根据模式执行不同操作
    if args.mode == "train":
        # 创建数据加载器
        print("创建数据加载器...")
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
        print("开始训练...")
        trainer.train(train_config['epochs'])
        
    elif args.mode == "eval":
        # 创建数据加载器
        print("创建数据加载器...")
        dataloaders = create_dataloaders(train_config, tokenizer.tokenizer)
        
        # 创建评估器
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer.tokenizer,
            config=train_config,
            device=device
        )
        
        # 评估模型
        print("开始评估...")
        results = evaluator.evaluate_all(dataloaders['val'])
        
        # 保存结果
        output_path = os.path.join(args.output_dir, "eval_results.json")
        evaluator.save_results(results, output_path)
        
        # 打印结果
        print("\n评估结果:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
        
    elif args.mode == "infer":
        # 创建推理引擎
        inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer.tokenizer,
            config=train_config,
            device=device
        )
        
        # 进入交互模式
        inference_engine.interactive_mode()
        
    elif args.mode == "serve":
        # 创建推理引擎
        inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer.tokenizer,
            config=train_config,
            device=device
        )
        
        # 创建服务器
        server = ModelServer(inference_engine, train_config)
        
        # 启动服务
        print(f"启动服务，端口: {args.port}")
        server.start(port=args.port)

if __name__ == "__main__":
    main() 