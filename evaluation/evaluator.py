import torch
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from .metrics import Metrics

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = Metrics()
        
        # 将模型移动到设备
        self.model.to(self.device)
        
    def evaluate_loss(self, dataloader) -> float:
        """评估模型损失"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 更新统计信息
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
                
        avg_loss = total_loss / total_samples
        perplexity = self.metrics.perplexity(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def evaluate_generation(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估模型生成能力"""
        self.model.eval()
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="Generating"):
            # 准备输入
            if 'question' in sample:
                input_text = f"问题: {sample['question']}"
                reference = sample.get('answer', '')
            elif 'instruction' in sample:
                input_text = f"指令: {sample['instruction']}"
                reference = sample.get('response', '')
            elif 'code' in sample:
                input_text = f"代码:\n{sample['code']}"
                reference = sample.get('explanation', '')
            else:
                continue
                
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get('max_input_length', 512)
            ).to(self.device)
            
            # 生成输出
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=self.config.get('max_output_length', 512),
                    num_beams=self.config.get('num_beams', 4),
                    no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', 3),
                    early_stopping=True
                )
            
            # 解码输出
            prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 保存预测和参考
            predictions.append(prediction)
            references.append(reference)
        
        # 计算指标
        results = {}
        
        # BLEU
        results['bleu'] = self.metrics.bleu(
            predictions, 
            [[ref] for ref in references]
        )
        
        # ROUGE
        rouge_scores = self.metrics.rouge(predictions, references)
        results['rouge-1'] = rouge_scores['rouge-1']['f']
        results['rouge-2'] = rouge_scores['rouge-2']['f']
        results['rouge-l'] = rouge_scores['rouge-l']['f']
        
        # 精确匹配
        results['exact_match'] = self.metrics.exact_match(predictions, references)
        
        return results
    
    def evaluate_all(self, test_loader, test_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """综合评估"""
        # 评估损失和困惑度
        loss_metrics = self.evaluate_loss(test_loader)
        
        results = {
            'loss': loss_metrics['loss'],
            'perplexity': loss_metrics['perplexity']
        }
        
        # 如果提供了测试数据，评估生成能力
        if test_data:
            gen_metrics = self.evaluate_generation(test_data)
            results.update(gen_metrics)
            
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"评估结果已保存到 {output_path}") 