import re
import unicodedata
from typing import List, Dict, Any

class TextProcessor:
    """文本预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def normalize(self, text: str) -> str:
        """标准化文本"""
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_code(self, code: str) -> str:
        """清理代码文本"""
        # 移除注释（可选，取决于任务）
        if self.config.get('remove_comments', False):
            # 简单的注释移除逻辑，实际应用中可能需要更复杂的解析器
            code = re.sub(r'//.*?$|/\*.*?\*/|\'\'\'.*?\'\'\'|""".*?"""', '', code, flags=re.DOTALL|re.MULTILINE)
        return code.strip()
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个样本"""
        if 'question' in sample:
            sample['question'] = self.normalize(sample['question'])
        
        if 'answer' in sample:
            sample['answer'] = self.normalize(sample['answer'])
            
        if 'code' in sample:
            sample['code'] = self.clean_code(sample['code'])
            
        return sample
    
    def process_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理样本"""
        return [self.process_sample(sample) for sample in samples] 