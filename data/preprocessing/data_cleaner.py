import re
import json
import os
from typing import Dict, List, Any, Optional, Union, Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
            
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除HTML标签
        if self.config.get('remove_html', True):
            text = re.sub(r'<[^>]+>', '', text)
            
        # 移除URL
        if self.config.get('remove_urls', True):
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
            
        # 移除特殊字符
        if self.config.get('remove_special_chars', False):
            text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', ' ', text)
            
        # 规范化标点
        if self.config.get('normalize_punctuation', True):
            text = re.sub(r'[\,\.\!\?]+(?=[\,\.\!\?])', '', text)
            
        # 移除多余空白（再次）
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_code(self, code: str) -> str:
        """清洗代码"""
        if not code:
            return ""
            
        # 移除注释（可选）
        if self.config.get('remove_code_comments', False):
            # 移除C风格注释
            code = re.sub(r'/\*[\s\S]*?\*/|//.*', '', code)
            # 移除Python风格注释
            code = re.sub(r'#.*', '', code)
            # 移除多行字符串（可能是文档字符串）
            code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', code)
            
        # 规范化空白
        if self.config.get('normalize_whitespace', True):
            # 将制表符替换为空格
            code = code.replace('\t', '    ')
            # 移除行尾空白
            code = re.sub(r' +$', '', code, flags=re.MULTILINE)
            # 最多允许两个连续空行
            code = re.sub(r'\n{3,}', '\n\n', code)
            
        return code.strip()
    
    def clean_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """清洗单个样本"""
        cleaned_sample = {}
        
        for key, value in sample.items():
            if key in ['question', 'instruction', 'response', 'answer', 'explanation']:
                cleaned_sample[key] = self.clean_text(value)
            elif key in ['code', 'function', 'program']:
                cleaned_sample[key] = self.clean_code(value)
            else:
                cleaned_sample[key] = value
                
        return cleaned_sample
    
    def clean_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗整个数据集"""
        cleaned_data = []
        
        for sample in tqdm(data, desc="清洗数据"):
            cleaned_sample = self.clean_sample(sample)
            
            # 过滤掉空样本
            if self.config.get('filter_empty', True):
                if all(not value for key, value in cleaned_sample.items() 
                       if key in ['question', 'instruction', 'response', 'answer', 'explanation', 'code']):
                    continue
                    
            cleaned_data.append(cleaned_sample)
            
        logger.info(f"清洗前样本数: {len(data)}, 清洗后样本数: {len(cleaned_data)}")
        
        return cleaned_data
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """处理单个文件"""
        # 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 清洗数据
        cleaned_data = self.clean_dataset(data)
        
        # 保存清洗后的数据
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                
        return cleaned_data
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """处理目录中的所有文件"""
        result = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                logger.info(f"处理文件: {input_path}")
                cleaned_data = self.process_file(input_path, output_path)
                
                result[filename] = cleaned_data
                
        return result 