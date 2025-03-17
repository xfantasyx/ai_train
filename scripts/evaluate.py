import os
import sys
import argparse
import json
import logging
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import DomainLLM
from model.tokenizer import DomainTokenizer
from data.dataloader import create_dataloaders
from evaluation.evaluator import Evaluator
from utils.logging import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估领域LLM")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    
    parser.add_argument("--test_data", type=str, required=True,
                        help="测试数据路径")
    
    parser.add_argument("--output_dir", type=str, default="./evaluation",
                        help="输出目录")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="评估批次大小")
    
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志目录")
    
    parser.add_argument("--generate", action="store_true",
                        help="是否评估生成能力")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"eval_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("评估参数:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 获取配置
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        # 尝试从同目录加载配置
        config_path = os.path.join(os.path.dirname(args.model_path), 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
        else:
            raise ValueError("未找到模型配置")
    
    # 初始化分词器
    logger.info("初始化分词器...")
    tokenizer = DomainTokenizer(model_config)
    
    # 初始化模型
    logger.info("初始化模型...")
    model = DomainLLM(model_config)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 评估配置
    eval_config = {
        'batch_size': args.batch_size,
        'test_data_path': args.test_data,
        'max_length': model_config.get('max_position_embeddings', 2048),
        'num_workers': 4
    }
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    test_dataset = create_dataloaders({
        'batch_size': args.batch_size,
        'test_data_path': args.test_data,
        'max_length': eval_config['max_length'],
        'num_workers': eval_config['num_workers']
    }, tokenizer.tokenizer)['val']  # 使用验证集加载器加载测试数据
    
    # 加载测试数据（用于生成评估）
    test_data = None
    if args.generate:
        logger.info("加载测试数据用于生成评估...")
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 限制样本数量以加快评估
        test_data = test_data[:100]  # 只使用前100个样本
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer.tokenizer,
        config=eval_config,
        device=device
    )
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluator.evaluate_all(test_loader=test_dataset, test_data=test_data)
    
    # 保存结果
    results_path = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    evaluator.save_results(results, results_path)
    
    # 打印结果
    logger.info("评估结果:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value}")
    
    logger.info("评估完成!")

if __name__ == "__main__":
    main() 