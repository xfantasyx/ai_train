from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional

class DomainTokenizer:
    """领域特定的分词器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 加载预训练分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer_name_or_path'],
            trust_remote_code=True
        )
        
        # 如果需要添加特殊token
        if 'special_tokens' in config:
            self.add_special_tokens(config['special_tokens'])
            
    def add_special_tokens(self, special_tokens: Dict[str, List[str]]):
        """添加特殊token"""
        special_tokens_dict = {}
        
        if 'additional_special_tokens' in special_tokens:
            special_tokens_dict['additional_special_tokens'] = special_tokens['additional_special_tokens']
            
        # 添加其他类型的特殊token
        for key in ['bos_token', 'eos_token', 'pad_token', 'sep_token']:
            if key in special_tokens:
                special_tokens_dict[key] = special_tokens[key]
                
        # 更新分词器
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"添加了 {num_added} 个特殊token")
        
    def train_new_tokenizer(self, corpus_files: List[str], vocab_size: int = 50000):
        """训练新的分词器（如果需要从头训练）"""
        from tokenizers import ByteLevelBPETokenizer
        
        # 初始化分词器
        tokenizer = ByteLevelBPETokenizer()
        
        # 在语料库上训练
        tokenizer.train(
            files=corpus_files,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        )
        
        # 保存分词器
        tokenizer.save_model(self.config['tokenizer_save_path'])
        
        # 加载新训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['tokenizer_save_path'],
            trust_remote_code=True
        )
        
    def __call__(self, *args, **kwargs):
        """调用底层分词器"""
        return self.tokenizer(*args, **kwargs) 