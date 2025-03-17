import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from typing import Dict, Any, Optional, Union, Tuple

class DomainLLM(nn.Module):
    """领域特定的大语言模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        if config.get('from_pretrained', False):
            # 从预训练模型初始化
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model_name_or_path'],
                trust_remote_code=True
            )
        else:
            # 从头开始训练
            model_config = AutoConfig.from_pretrained(config['model_name_or_path'])
            # 可以根据需要修改配置
            if 'hidden_size' in config:
                model_config.hidden_size = config['hidden_size']
            if 'num_hidden_layers' in config:
                model_config.num_hidden_layers = config['num_hidden_layers']
            if 'num_attention_heads' in config:
                model_config.num_attention_heads = config['num_attention_heads']
                
            self.model = AutoModelForCausalLM.from_config(model_config)
            
        # 如果需要冻结某些层
        if config.get('freeze_layers', False):
            self._freeze_layers(config['freeze_layers'])
            
    def _freeze_layers(self, layers_to_freeze: Union[int, list]):
        """冻结指定层"""
        if isinstance(layers_to_freeze, int):
            # 冻结前n层
            for param in self.model.transformer.h[:layers_to_freeze].parameters():
                param.requires_grad = False
        elif isinstance(layers_to_freeze, list):
            # 冻结指定层
            for layer_idx in layers_to_freeze:
                for param in self.model.transformer.h[layer_idx].parameters():
                    param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, **kwargs):
        """生成文本"""
        return self.model.generate(**kwargs) 