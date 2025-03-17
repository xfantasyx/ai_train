import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from tqdm import tqdm

from .data_cleaner import DataCleaner
from .format_converter import FormatConverter
from .dataset_splitter import DatasetSplitter
from .quality_checker import QualityChecker
from ..augmentation.data_augmenter import DataAugmenter

logger = logging.getLogger(__name__)

class DataPipeline:
    """数据处理流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各个组件
        self.cleaner = DataCleaner(config.get('cleaner_config', {}))
        self.converter = FormatConverter(config.get('converter_config', {}))
        self.splitter = DatasetSplitter(config.get('splitter_config', {}))
        self.quality_checker = QualityChecker(config.get('quality_config', {}))
        
        # 如果需要数据增强
        if config.get('use_augmentation', False):
            self.augmenter = DataAugmenter(config.get('augmenter_config', {}))
        else:
            self.augmenter = None
    
    def process(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """处理数据集"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 加载数据
        logger.info(f"加载数据: {input_path}")
        data = self.converter.load_data(input_path)
        logger.info(f"加载了 {len(data)} 条数据")
        
        # 步骤2: 检查数据质量
        if self.config.get('check_quality', True):
            logger.info("检查数据质量")
            quality_results = self.quality_checker.check_dataset(data)
            
            # 保存质量报告
            quality_report_path = os.path.join(output_dir, 'quality_report.json')
            self.quality_checker.save_report(quality_results, quality_report_path)
            
            # 过滤低质量样本
            if self.config.get('filter_low_quality', True):
                logger.info("过滤低质量样本")
                data = self.quality_checker.filter_dataset(data)
        
        # 步骤3: 清洗数据
        if self.config.get('clean_data', True):
            logger.info("清洗数据")
            data = self.cleaner.clean_dataset(data)
        
        # 步骤4: 转换数据格式
        if self.config.get('convert_format', True):
            target_format = self.config.get('target_format', 'instruction')
            logger.info(f"转换数据格式为: {target_format}")
            
            if target_format == 'instruction':
                data = self.converter.convert_to_instruction_format(data)
            elif target_format == 'chat':
                data = self.converter.convert_to_chat_format(data)
        
        # 步骤5: 数据增强
        if self.augmenter and self.config.get('augment_data', False):
            logger.info("进行数据增强")
            data = self.augmenter.augment_dataset(data)
        
        # 步骤6: 分割数据集
        if self.config.get('split_data', True):
            logger.info("分割数据集")
            
            # 是否使用分层抽样
            if self.config.get('use_stratified_split', False):
                stratify_field = self.config.get('stratify_field', 'category')
                splits = self.splitter.stratified_split(data, stratify_field)
            else:
                splits = self.splitter.split_dataset(data)
            
            # 保存分割后的数据集
            for split_name, split_data in splits.items():
                split_path = os.path.join(output_dir, f"{split_name}.json")
                
                with open(split_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"已将{split_name}集保存到 {split_path}")
                
            # 转换为其他格式（如果需要）
            if self.config.get('save_jsonl', False):
                for split_name, split_data in splits.items():
                    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
                    self.converter.convert_to_jsonl(split_data, jsonl_path)
            
            result = {
                'splits': {name: len(data) for name, data in splits.items()},
                'total_processed': len(data)
            }
        else:
            # 不分割，直接保存
            output_path = os.path.join(output_dir, 'processed_data.json')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已将处理后的数据保存到 {output_path}")
            
            # 转换为其他格式（如果需要）
            if self.config.get('save_jsonl', False):
                jsonl_path = os.path.join(output_dir, 'processed_data.jsonl')
                self.converter.convert_to_jsonl(data, jsonl_path)
            
            result = {
                'total_processed': len(data)
            }
        
        return result 