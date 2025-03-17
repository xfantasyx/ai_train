import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# 下载NLTK资源
try:
    nltk.download('punkt')
except:
    pass

class Metrics:
    """评估指标计算"""
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """计算困惑度"""
        return torch.exp(torch.tensor(loss)).item()
    
    @staticmethod
    def accuracy(predictions: List[int], references: List[int]) -> float:
        """计算准确率"""
        return accuracy_score(references, predictions)
    
    @staticmethod
    def precision_recall_f1(predictions: List[int], references: List[int]) -> Dict[str, float]:
        """计算精确率、召回率和F1分数"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average='weighted'
        )
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def bleu(predictions: List[str], references: List[List[str]]) -> float:
        """计算BLEU分数"""
        if not predictions or not references:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            # 分词
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(r.lower()) for r in ref]
            
            # 计算BLEU
            try:
                score = sentence_bleu(ref_tokens, pred_tokens)
                scores.append(score)
            except:
                scores.append(0.0)
                
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def rouge(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """计算ROUGE分数"""
        if not predictions or not references:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
        
        rouge = Rouge()
        try:
            scores = rouge.get_scores(predictions, references, avg=True)
            return scores
        except:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """计算精确匹配率"""
        if not predictions or not references:
            return 0.0
        
        matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return matches / len(predictions) 