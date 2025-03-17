import torch
import time
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到设备
        self.model.to(self.device)
        self.model.eval()
        
        # 设置生成参数
        self.gen_kwargs = {
            'max_length': config.get('max_length', 1024),
            'min_length': config.get('min_length', 1),
            'do_sample': config.get('do_sample', True),
            'top_p': config.get('top_p', 0.9),
            'top_k': config.get('top_k', 50),
            'temperature': config.get('temperature', 0.7),
            'num_beams': config.get('num_beams', 1),
            'no_repeat_ngram_size': config.get('no_repeat_ngram_size', 3),
            'early_stopping': config.get('early_stopping', True)
        }
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        # 合并生成参数
        generation_kwargs = {**self.gen_kwargs, **kwargs}
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_input_length', 512)
        ).to(self.device)
        
        # 记录开始时间
        start_time = time.time()
        
        # 生成输出
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs
            )
        
        # 计算生成时间
        generation_time = time.time() - start_time
        
        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 移除提示部分（如果需要）
        if self.config.get('remove_prompt', False):
            output_text = output_text[len(prompt):].strip()
            
        logger.info(f"生成时间: {generation_time:.2f}秒")
        
        return output_text
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成文本"""
        results = []
        
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            
        return results
    
    def interactive_mode(self):
        """交互模式"""
        print("进入交互模式，输入'exit'退出")
        
        while True:
            prompt = input("\n请输入提示: ")
            
            if prompt.lower() in ['exit', 'quit', '退出']:
                break
                
            try:
                response = self.generate(prompt)
                print(f"\n回答: {response}")
            except Exception as e:
                print(f"生成出错: {e}") 