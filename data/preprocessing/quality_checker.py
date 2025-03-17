import re
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Tuple, Set
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class QualityChecker:
    """数据质量检查工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_length = config.get('min_length', 10)
        self.max_length = config.get('max_length', 2048)
        self.min_instruction_length = config.get('min_instruction_length', 5)
        self.min_response_length = config.get('min_response_length', 10)
        
    def check_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查整个数据集的质量"""
        results = {
            'total_samples': len(data),
            'valid_samples': 0,
            'invalid_samples': 0,
            'issues': {
                'too_short': 0,
                'too_long': 0,
                'empty_fields': 0,
                'duplicate_samples': 0,
                'low_quality_content': 0
            },
            'field_stats': {},
            'length_stats': {}
        }
        
        # 检查重复
        seen_contents = set()
        duplicates = 0
        
        # 字段统计
        field_counter = Counter()
        
        # 长度统计
        lengths = {
            'instruction': [],
            'response': [],
            'question': [],
            'answer': [],
            'code': [],
            'explanation': []
        }
        
        for sample in tqdm(data, desc="检查数据质量"):
            is_valid = True
            
            # 统计字段
            for field in sample.keys():
                field_counter[field] += 1
            
            # 检查内容
            content_key = self._get_content_key(sample)
            if content_key in seen_contents:
                results['issues']['duplicate_samples'] += 1
                duplicates += 1
                is_valid = False
            else:
                seen_contents.add(content_key)
            
            # 检查字段长度
            for field in ['instruction', 'response', 'question', 'answer', 'code', 'explanation']:
                if field in sample and sample[field]:
                    field_length = len(sample[field])
                    lengths[field].append(field_length)
                    
                    # 检查是否太短
                    if field in ['instruction', 'question'] and field_length < self.min_instruction_length:
                        results['issues']['too_short'] += 1
                        is_valid = False
                    elif field in ['response', 'answer', 'explanation'] and field_length < self.min_response_length:
                        results['issues']['too_short'] += 1
                        is_valid = False
                    
                    # 检查是否太长
                    if field_length > self.max_length:
                        results['issues']['too_long'] += 1
                        is_valid = False
            
            # 检查空字段
            if self._has_empty_required_fields(sample):
                results['issues']['empty_fields'] += 1
                is_valid = False
            
            # 检查低质量内容
            if self._is_low_quality(sample):
                results['issues']['low_quality_content'] += 1
                is_valid = False
            
            # 更新统计
            if is_valid:
                results['valid_samples'] += 1
            else:
                results['invalid_samples'] += 1
        
        # 计算字段统计
        results['field_stats'] = {field: count for field, count in field_counter.items()}
        
        # 计算长度统计
        for field, values in lengths.items():
            if values:
                results['length_stats'][field] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values)
                }
        
        # 计算重复率
        results['duplicate_rate'] = duplicates / len(data) if data else 0
        
        return results
    
    def _get_content_key(self, sample: Dict[str, Any]) -> str:
        """获取样本的内容键，用于检测重复"""
        key_parts = []
        
        for field in ['instruction', 'response', 'question', 'answer', 'code', 'explanation']:
            if field in sample and sample[field]:
                # 规范化文本以减少假阳性
                normalized = re.sub(r'\s+', ' ', sample[field].strip().lower())
                key_parts.append(f"{field}:{normalized}")
        
        return '|'.join(key_parts)
    
    def _has_empty_required_fields(self, sample: Dict[str, Any]) -> bool:
        """检查是否有空的必填字段"""
        # 检查问答对
        if ('question' in sample and not sample['question']) or ('answer' in sample and not sample['answer']):
            return True
        
        # 检查指令对
        if ('instruction' in sample and not sample['instruction']) or ('response' in sample and not sample['response']):
            return True
        
        # 检查代码解释对
        if ('code' in sample and not sample['code']) or ('explanation' in sample and not sample['explanation']):
            return True
        
        return False
    
    def _is_low_quality(self, sample: Dict[str, Any]) -> bool:
        """检查是否是低质量内容"""
        # 检查重复文本
        for field in ['instruction', 'response', 'question', 'answer', 'explanation']:
            if field in sample and sample[field]:
                text = sample[field]
                
                # 检查重复段落
                paragraphs = text.split('\n')
                if len(paragraphs) > 3:
                    paragraph_set = set(paragraphs)
                    if len(paragraph_set) < len(paragraphs) * 0.7:  # 超过30%的段落重复
                        return True
                
                # 检查重复句子
                sentences = re.split(r'[.!?]+', text)
                if len(sentences) > 5:
                    sentence_set = set(s.strip() for s in sentences if s.strip())
                    if len(sentence_set) < len(sentences) * 0.7:  # 超过30%的句子重复
                        return True
        
        return False
    
    def filter_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤低质量样本"""
        filtered_data = []
        
        for sample in tqdm(data, desc="过滤数据"):
            # 检查内容长度
            is_valid = True
            
            # 检查字段长度
            for field in ['instruction', 'response', 'question', 'answer', 'code', 'explanation']:
                if field in sample and sample[field]:
                    field_length = len(sample[field])
                    
                    # 检查是否太短
                    if field in ['instruction', 'question'] and field_length < self.min_instruction_length:
                        is_valid = False
                        break
                    elif field in ['response', 'answer', 'explanation'] and field_length < self.min_response_length:
                        is_valid = False
                        break
                    
                    # 检查是否太长
                    if field_length > self.max_length:
                        is_valid = False
                        break
            
            # 检查空字段
            if self._has_empty_required_fields(sample):
                is_valid = False
            
            # 检查低质量内容
            if self._is_low_quality(sample):
                is_valid = False
            
            if is_valid:
                filtered_data.append(sample)
        
        logger.info(f"过滤前样本数: {len(data)}, 过滤后样本数: {len(filtered_data)}")
        
        return filtered_data
    
    def save_report(self, results: Dict[str, Any], output_path: str):
        """保存质量报告"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"质量报告已保存到 {output_path}") 