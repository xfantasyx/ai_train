import os
import sys
import argparse
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.pipeline import DataPipeline
from utils.logging import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="处理数据集")
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入数据路径")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    
    parser.add_argument("--config_path", type=str, default=None,
                        help="配置文件路径")
    
    parser.add_argument("--check_quality", action="store_true",
                        help="是否检查数据质量")
    
    parser.add_argument("--filter_low_quality", action="store_true",
                        help="是否过滤低质量样本")
    
    parser.add_argument("--clean_data", action="store_true",
                        help="是否清洗数据")
    
    parser.add_argument("--convert_format", action="store_true",
                        help="是否转换数据格式")
    
    parser.add_argument("--target_format", type=str, default="instruction",
                        choices=["instruction", "chat"],
                        help="目标数据格式")
    
    parser.add_argument("--augment_data", action="store_true",
                        help="是否进行数据增强")
    
    parser.add_argument("--split_data", action="store_true",
                        help="是否分割数据集")
    
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    
    parser.add_argument("--save_jsonl", action="store_true",
                        help="是否保存为JSONL格式")
    
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"process_data_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("数据处理参数:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # 加载配置
    if args.config_path:
        logger.info(f"从配置文件加载: {args.config_path}")
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 使用命令行参数创建配置
        config = {
            'check_quality': args.check_quality,
            'filter_low_quality': args.filter_low_quality,
            'clean_data': args.clean_data,
            'convert_format': args.convert_format,
            'target_format': args.target_format,
            'augment_data': args.augment_data,
            'split_data': args.split_data,
            'save_jsonl': args.save_jsonl,
            
            # 分割配置
            'splitter_config': {
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'random_seed': 42
            },
            
            # 质量检查配置
            'quality_config': {
                'min_length': 10,
                'max_length': 2048,
                'min_instruction_length': 5,
                'min_response_length': 10
            },
            
            # 清洗配置
            'cleaner_config': {
                'remove_html': True,
                'remove_urls': True,
                'normalize_punctuation': True,
                'normalize_whitespace': True
            }
        }
    
    # 创建数据处理流水线
    pipeline = DataPipeline(config)
    
    # 处理数据
    logger.info(f"开始处理数据: {args.input_path}")
    result = pipeline.process(args.input_path, args.output_dir)
    
    # 打印结果
    logger.info("处理结果:")
    for key, value in result.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("数据处理完成!")

if __name__ == "__main__":
    main() 