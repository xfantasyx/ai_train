import random
import copy
import re
import nltk
from typing import Dict, List, Any, Optional, Union, Callable
from tqdm import tqdm
import logging

# 下载NLTK资源
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import wordnet
except:
    pass

logger = logging.getLogger(__name__)

class DataAugmenter:
    """数据增强工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.techniques = {
            'synonym_replacement': self.synonym_replacement,
            'random_deletion': self.random_deletion,
            'random_swap': self.random_swap,
            'back_translation': self.back_translation,
            'code_comment_variation': self.code_comment_variation,
            'variable_name_change': self.variable_name_change
        }
        
    def augment_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强整个数据集"""
        augmented_data = copy.deepcopy(data)
        
        # 获取要应用的增强技术
        techniques = self.config.get('techniques', ['synonym_replacement'])
        
        # 获取每个样本的增强次数
        augment_times = self.config.get('augment_times', 1)
        
        # 获取要增强的字段
        fields_to_augment = self.config.get('fields_to_augment', ['question', 'instruction'])
        
        # 对每个样本进行增强
        for sample in tqdm(data, desc="增强数据"):
            for _ in range(augment_times):
                # 随机选择一种增强技术
                technique = random.choice(techniques)
                
                # 创建新样本
                new_sample = copy.deepcopy(sample)
                
                # 对指定字段应用增强
                for field in fields_to_augment:
                    if field in sample and sample[field]:
                        new_sample[field] = self.techniques[technique](sample[field])
                
                # 添加到增强数据集
                augmented_data.append(new_sample)
        
        logger.info(f"原始样本数: {len(data)}, 增强后样本数: {len(augmented_data)}")
        
        return augmented_data
    
    def synonym_replacement(self, text: str) -> str:
        """同义词替换"""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        n = max(1, int(len(words) * self.config.get('synonym_replace_ratio', 0.1)))
        random_word_indices = random.sample(range(len(words)), min(n, len(words)))
        
        for idx in random_word_indices:
            word = words[idx]
            synonyms = []
            
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        synonyms.append(synonym)
            
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str) -> str:
        """随机删除单词"""
        words = nltk.word_tokenize(text)
        
        # 如果文本太短，不进行删除
        if len(words) <= 5:
            return text
            
        p = self.config.get('random_delete_ratio', 0.1)
        new_words = []
        
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        # 确保至少保留一个单词
        if not new_words:
            new_words = [random.choice(words)]
            
        return ' '.join(new_words)
    
    def random_swap(self, text: str) -> str:
        """随机交换单词位置"""
        words = nltk.word_tokenize(text)
        
        # 如果文本太短，不进行交换
        if len(words) <= 1:
            return text
            
        new_words = words.copy()
        n = max(1, int(len(words) * self.config.get('random_swap_ratio', 0.1)))
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)
    
    def back_translation(self, text: str) -> str:
        """回译增强（需要外部翻译API）"""
        # 这里只是一个示例，实际实现需要调用翻译API
        # 例如可以使用Google Translate API或其他翻译服务
        
        # 模拟回译效果
        words = nltk.word_tokenize(text)
        new_words = []
        
        for word in words:
            # 随机决定是否替换为同义词
            if random.random() < 0.2 and len(word) > 3:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != word:
                            synonyms.append(synonym)
                
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)
    
    def code_comment_variation(self, code: str) -> str:
        """代码注释变体"""
        # 查找代码中的注释
        c_style_comments = re.findall(r'/\*[\s\S]*?\*/|//.*', code)
        python_comments = re.findall(r'#.*', code)
        
        new_code = code
        
        # 修改C风格注释
        for comment in c_style_comments:
            if comment.startswith('//'):
                # 单行注释
                new_comment = f"// {self.random_deletion(comment[2:])}"
            else:
                # 多行注释
                content = comment[2:-2].strip()
                new_content = self.random_deletion(content)
                new_comment = f"/* {new_content} */"
                
            new_code = new_code.replace(comment, new_comment)
            
        # 修改Python风格注释
        for comment in python_comments:
            new_comment = f"# {self.random_deletion(comment[1:])}"
            new_code = new_code.replace(comment, new_comment)
            
        return new_code
    
    def variable_name_change(self, code: str) -> str:
        """变量名变化"""
        # 这是一个简化版本，实际实现可能需要使用代码解析器
        
        # 查找可能的变量名
        # 这里使用简单的正则表达式，可能不够准确
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        # 排除的关键字
        keywords = ['if', 'else', 'for', 'while', 'return', 'function', 'def', 'class',
                   'int', 'float', 'str', 'bool', 'True', 'False', 'None', 'import',
                   'from', 'as', 'try', 'except', 'finally', 'raise', 'with']
        
        # 查找所有可能的变量名
        potential_vars = set(re.findall(var_pattern, code))
        variables = [var for var in potential_vars if var not in keywords and len(var) > 1]
        
        # 如果没有找到变量，返回原始代码
        if not variables:
            return code
            
        # 随机选择一个变量进行替换
        var_to_replace = random.choice(variables)
        
        # 生成新的变量名
        prefixes = ['new', 'my', 'temp', 'modified']
        new_var = f"{random.choice(prefixes)}_{var_to_replace}"
        
        # 替换变量名（使用单词边界确保只替换完整的变量名）
        new_code = re.sub(r'\b' + var_to_replace + r'\b', new_var, code)
        
        return new_code 