import os
import sys
import argparse
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collection.data_collector import DataCollector
from utils.logging import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="收集数据")
    
    parser.add_argument("--config_path", type=str, required=True,
                        help="配置文件路径")
    
    parser.add_argument("--output_dir", type=str, default="./collected_data",
                        help="输出目录")
    
    parser.add_argument("--max_workers", type=int, default=4,
                        help="最大工作线程数")
    
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
    log_file = os.path.join(args.log_dir, f"collect_data_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("数据收集参数:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # 加载配置
    logger.info(f"从配置文件加载: {args.config_path}")
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新配置
    config['output_dir'] = args.output_dir
    config['max_workers'] = args.max_workers
    
    # 创建数据收集器
    collector = DataCollector(config)
    
    # 收集数据
    logger.info("开始收集数据")
    results = collector.collect()
    
    # 打印结果
    logger.info("收集结果:")
    for source, count in results.items():
        logger.info(f"  {source}: {count} 条数据")
    
    logger.info("数据收集完成!")

if __name__ == "__main__":
    main() 