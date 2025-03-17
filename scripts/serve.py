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
from inference.engine import InferenceEngine
from inference.serving import ModelServer
from utils.logging import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="部署领域LLM服务")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务主机地址")
    
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口")
    
    parser.add_argument("--max_length", type=int, default=1024,
                        help="生成的最大长度")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成的温度")
    
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="生成的top-p值")
    
    parser.add_argument("--top_k", type=int, default=50,
                        help="生成的top-k值")
    
    parser.add_argument("--num_beams", type=int, default=1,
                        help="束搜索的束数")
    
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
    log_file = os.path.join(args.log_dir, f"serve_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("服务参数:")
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
    
    # 推理配置
    infer_config = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'num_beams': args.num_beams,
        'do_sample': args.temperature > 0
    }
    
    # 创建推理引擎
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer.tokenizer,
        config=infer_config,
        device=device
    )
    
    # 创建服务器
    server = ModelServer(inference_engine, infer_config)
    
    # 启动服务
    logger.info(f"启动服务，地址: {args.host}:{args.port}")
    server.start(host=args.host, port=args.port)

if __name__ == "__main__":
    main() 