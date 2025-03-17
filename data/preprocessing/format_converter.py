import json
import csv
import os
import re
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class FormatConverter:
    """数据格式转换工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def convert_to_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """转换为JSONL格式"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"已将{len(data)}条数据转换为JSONL格式并保存到 {output_path}")
    
    def convert_to_json(self, data: List[Dict[str, Any]], output_path: str):
        """转换为JSON格式"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已将{len(data)}条数据转换为JSON格式并保存到 {output_path}")
    
    def convert_to_csv(self, data: List[Dict[str, Any]], output_path: str):
        """转换为CSV格式"""
        if not data:
            logger.warning("没有数据可转换")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取所有字段
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = list(fieldnames)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
        logger.info(f"已将{len(data)}条数据转换为CSV格式并保存到 {output_path}")
    
    def convert_to_instruction_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换为指令格式"""
        instruction_data = []
        
        for item in data:
            # 问答格式转指令格式
            if 'question' in item and 'answer' in item:
                instruction_data.append({
                    'instruction': item['question'],
                    'response': item['answer'],
                    'input': item.get('context', ''),
                    'id': item.get('id', '')
                })
            # 代码解释格式转指令格式
            elif 'code' in item and 'explanation' in item:
                instruction_data.append({
                    'instruction': f"解释以下代码:\n{item['code']}",
                    'response': item['explanation'],
                    'input': '',
                    'id': item.get('id', '')
                })
            # 已经是指令格式
            elif 'instruction' in item and 'response' in item:
                instruction_data.append(item)
                
        logger.info(f"已将{len(data)}条数据转换为指令格式，得到{len(instruction_data)}条数据")
        
        return instruction_data
    
    def convert_to_chat_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换为对话格式"""
        chat_data = []
        
        for item in data:
            messages = []
            
            # 问答格式转对话格式
            if 'question' in item and 'answer' in item:
                messages.append({"role": "user", "content": item['question']})
                messages.append({"role": "assistant", "content": item['answer']})
            # 指令格式转对话格式
            elif 'instruction' in item and 'response' in item:
                content = item['instruction']
                if 'input' in item and item['input']:
                    content += f"\n\n{item['input']}"
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": item['response']})
            # 代码解释格式转对话格式
            elif 'code' in item and 'explanation' in item:
                messages.append({"role": "user", "content": f"解释以下代码:\n{item['code']}"})
                messages.append({"role": "assistant", "content": item['explanation']})
            # 已经是对话格式
            elif 'messages' in item:
                messages = item['messages']
                
            if messages:
                chat_data.append({
                    "messages": messages,
                    "id": item.get('id', '')
                })
                
        logger.info(f"已将{len(data)}条数据转换为对话格式，得到{len(chat_data)}条数据")
        
        return chat_data
    
    def load_data(self, input_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        data = []
        
        if input_path.endswith('.json'):
            # 加载JSON文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif input_path.endswith('.jsonl'):
            # 加载JSONL文件
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        elif input_path.endswith('.csv'):
            # 加载CSV文件
            with open(input_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))
        else:
            logger.error(f"不支持的文件格式: {input_path}")
            
        return data
    
    def convert_file(self, input_path: str, output_path: str, output_format: str, data_format: Optional[str] = None):
        """转换文件格式"""
        # 加载数据
        data = self.load_data(input_path)
        
        # 转换数据格式
        if data_format == 'instruction':
            data = self.convert_to_instruction_format(data)
        elif data_format == 'chat':
            data = self.convert_to_chat_format(data)
            
        # 保存为指定格式
        if output_format == 'json':
            self.convert_to_json(data, output_path)
        elif output_format == 'jsonl':
            self.convert_to_jsonl(data, output_path)
        elif output_format == 'csv':
            self.convert_to_csv(data, output_path)
        else:
            logger.error(f"不支持的输出格式: {output_format}") 