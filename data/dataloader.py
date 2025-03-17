import json
import os
from typing import Dict, List, Iterator, Any
import torch
from torch.utils.data import Dataset, DataLoader

class DomainDataset(Dataset):
    """领域数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, is_training: bool = True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.samples = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        if os.path.isdir(self.data_path):
            # 处理目录中的多个文件
            samples = []
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.data_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        samples.extend(json.load(f))
            return samples
        else:
            # 处理单个文件
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 构建输入文本
        if 'question' in sample and 'answer' in sample:
            # 问答格式
            input_text = f"问题: {sample['question']}\n回答: {sample['answer']}"
        elif 'instruction' in sample and 'response' in sample:
            # 指令格式
            input_text = f"指令: {sample['instruction']}\n响应: {sample['response']}"
        elif 'code' in sample and 'explanation' in sample:
            # 代码解释格式
            input_text = f"代码:\n{sample['code']}\n解释: {sample['explanation']}"
        else:
            # 纯文本格式
            input_text = sample.get('text', '')
        
        # 编码文本
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移除批次维度
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        # 对于自回归训练，标签就是输入ID
        if self.is_training:
            item['labels'] = item['input_ids'].clone()
            
        return item

def create_dataloaders(config: Dict[str, Any], tokenizer) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    train_dataset = DomainDataset(
        data_path=config['train_data_path'],
        tokenizer=tokenizer,
        max_length=config['max_length'],
        is_training=True
    )
    
    val_dataset = DomainDataset(
        data_path=config['val_data_path'],
        tokenizer=tokenizer,
        max_length=config['max_length'],
        is_training=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    } 