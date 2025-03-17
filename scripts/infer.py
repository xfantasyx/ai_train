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
from utils.logging import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用领域LLM进行推理")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    
    parser.add_argument("--interactive", action="store_true",
                        help="是否进入交互模式")
    
    parser.add_argument("--input_file", type=str, default=None,
                        help="输入文件路径（每行一个提示）")
    
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")
    
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
    log_file = os.path.join(args.log_dir, f"infer_{timestamp}.log")
    logger = setup_logging(log_file=log_file)
    
    # 记录参数
    logger.info("推理参数:")
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
    
    # 根据模式执行不同操作
    if args.interactive:
        # 交互模式
        logger.info("进入交互模式")
        inference_engine.interactive_mode()
    elif args.input_file:
        # 批处理模式
        logger.info(f"从文件加载提示: {args.input_file}")
        
        # 加载提示
        prompts = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        
        logger.info(f"加载了 {len(prompts)} 个提示")
        
        # 生成回复
        logger.info("开始生成回复...")
        responses = inference_engine.batch_generate(prompts)
        
        # 保存结果
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for prompt, response in zip(prompts, responses):
                    f.write(f"提示: {prompt}\n")
                    f.write(f"回复: {response}\n")
                    f.write("-" * 80 + "\n")
            
            logger.info(f"结果已保存到 {args.output_file}")
        else:
            # 打印结果
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n[{i+1}/{len(prompts)}]")
                print(f"提示: {prompt}")
                print(f"回复: {response}")
                print("-" * 80)
    else:
        # 没有指定模式，进入交互模式
        logger.info("未指定模式，进入交互模式")
        inference_engine.interactive_mode()

if __name__ == "__main__":
    main() 