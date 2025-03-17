import os
import json
import requests
import time
import logging
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import concurrent.futures

logger = logging.getLogger(__name__)

class DataCollector:
    """数据收集工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = config.get('sources', [])
        self.output_dir = config.get('output_dir', './collected_data')
        self.max_workers = config.get('max_workers', 4)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def collect(self) -> Dict[str, Any]:
        """从所有配置的源收集数据"""
        results = {}
        
        for source in self.sources:
            source_type = source.get('type', '')
            source_name = source.get('name', source_type)
            
            logger.info(f"从源 {source_name} 收集数据")
            
            try:
                if source_type == 'api':
                    data = self._collect_from_api(source)
                elif source_type == 'file':
                    data = self._collect_from_file(source)
                elif source_type == 'github':
                    data = self._collect_from_github(source)
                elif source_type == 'web':
                    data = self._collect_from_web(source)
                else:
                    logger.warning(f"不支持的源类型: {source_type}")
                    continue
                
                # 保存收集到的数据
                output_path = os.path.join(self.output_dir, f"{source_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                results[source_name] = len(data)
                logger.info(f"从 {source_name} 收集了 {len(data)} 条数据")
                
            except Exception as e:
                logger.error(f"从 {source_name} 收集数据时出错: {e}")
                results[source_name] = 0
        
        return results
    
    def _collect_from_api(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从API收集数据"""
        url = source.get('url', '')
        headers = source.get('headers', {})
        params = source.get('params', {})
        method = source.get('method', 'GET')
        max_items = source.get('max_items', 1000)
        
        data = []
        page = 1
        
        with tqdm(total=max_items, desc=f"从API收集数据") as pbar:
            while len(data) < max_items:
                # 添加分页参数
                current_params = params.copy()
                current_params['page'] = page
                current_params['limit'] = min(100, max_items - len(data))
                
                # 发送请求
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, params=current_params)
                else:
                    response = requests.post(url, headers=headers, json=current_params)
                
                # 检查响应
                if response.status_code != 200:
                    logger.warning(f"API请求失败: {response.status_code} - {response.text}")
                    break
                
                # 解析响应
                try:
                    response_data = response.json()
                    
                    # 提取数据项
                    items_key = source.get('items_key', 'data')
                    if items_key in response_data:
                        items = response_data[items_key]
                    else:
                        items = response_data
                    
                    # 如果返回的是字典而不是列表，转换为列表
                    if isinstance(items, dict):
                        items = [items]
                    
                    # 添加到结果
                    data.extend(items)
                    pbar.update(len(items))
                    
                    # 检查是否有更多数据
                    has_more_key = source.get('has_more_key', 'has_more')
                    if has_more_key in response_data and not response_data[has_more_key]:
                        break
                    
                    # 增加页码
                    page += 1
                    
                    # 添加延迟以避免请求过快
                    time.sleep(source.get('delay', 1))
                    
                except Exception as e:
                    logger.error(f"解析API响应时出错: {e}")
                    break
        
        return data[:max_items]
    
    def _collect_from_file(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从文件收集数据"""
        file_path = source.get('path', '')
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            # 根据文件扩展名加载数据
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.endswith('.jsonl'):
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []
            
            # 如果数据不是列表，转换为列表
            if not isinstance(data, list):
                data = [data]
            
            return data
            
        except Exception as e:
            logger.error(f"从文件加载数据时出错: {e}")
            return []
    
    def _collect_from_github(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从GitHub收集数据"""
        # 这里只是一个简单的实现，实际应用中可能需要更复杂的逻辑
        repo = source.get('repo', '')
        path = source.get('path', '')
        token = source.get('token', '')
        
        if not repo:
            logger.error("未指定GitHub仓库")
            return []
        
        # 构建API URL
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        
        # 设置请求头
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        
        if token:
            headers['Authorization'] = f"token {token}"
        
        try:
            # 发送请求
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"GitHub API请求失败: {response.status_code} - {response.text}")
                return []
            
            # 解析响应
            contents = response.json()
            
            # 如果是目录，获取所有文件
            if isinstance(contents, list):
                files = [item for item in contents if item['type'] == 'file']
            else:
                files = [contents]
            
            # 收集数据
            data = []
            
            for file in tqdm(files, desc="从GitHub收集文件"):
                # 获取文件内容
                file_url = file['download_url']
                file_response = requests.get(file_url)
                
                if file_response.status_code != 200:
                    logger.warning(f"获取文件失败: {file_url}")
                    continue
                
                # 解析文件内容
                file_content = file_response.text
                
                # 根据文件类型处理
                if file['name'].endswith('.json'):
                    try:
                        file_data = json.loads(file_content)
                        
                        # 如果是列表，扩展数据
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
                    except:
                        logger.warning(f"解析JSON文件失败: {file['name']}")
                elif file['name'].endswith('.py'):
                    # 提取Python文件中的文档字符串和注释
                    data.append({
                        'file_name': file['name'],
                        'content': file_content,
                        'type': 'code',
                        'language': 'python'
                    })
                
                # 添加延迟以避免请求过快
                time.sleep(source.get('delay', 0.5))
            
            return data
            
        except Exception as e:
            logger.error(f"从GitHub收集数据时出错: {e}")
            return []
    
    def _collect_from_web(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从网页收集数据"""
        # 这里需要使用网页抓取库，如BeautifulSoup或Scrapy
        urls = source.get('urls', [])
        selectors = source.get('selectors', {})
        
        if not urls:
            logger.error("未指定URL")
            return []
        
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("未安装BeautifulSoup，无法从网页收集数据")
            return []
        
        data = []
        
        # 并行抓取
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self._scrape_url, url, selectors, source): url for url in urls}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), desc="从网页收集数据"):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        data.extend(result)
                except Exception as e:
                    logger.error(f"抓取URL时出错 {url}: {e}")
        
        return data
    
    def _scrape_url(self, url: str, selectors: Dict[str, str], source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """抓取单个URL"""
        try:
            # 发送请求
            headers = source.get('headers', {'User-Agent': 'Mozilla/5.0'})
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"请求失败: {url} - {response.status_code}")
                return []
            
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取数据
            items = []
            
            # 如果有容器选择器，先找到所有容器
            container_selector = selectors.get('container', '')
            if container_selector:
                containers = soup.select(container_selector)
            else:
                # 如果没有容器选择器，将整个页面作为一个容器
                containers = [soup]
            
            # 从每个容器中提取数据
            for container in containers:
                item = {}
                
                # 提取各个字段
                for field, selector in selectors.items():
                    if field == 'container':
                        continue
                    
                    elements = container.select(selector)
                    if elements:
                        # 根据字段类型处理
                        if field.endswith('_text'):
                            item[field[:-5]] = elements[0].get_text().strip()
                        elif field.endswith('_html'):
                            item[field[:-5]] = str(elements[0])
                        elif field.endswith('_attr'):
                            # 格式: field_attr_name
                            attr_name = field.split('_')[-1]
                            item[field.rsplit('_', 2)[0]] = elements[0].get(attr_name, '')
                        else:
                            item[field] = elements[0].get_text().strip()
                
                # 添加URL
                item['source_url'] = url
                
                # 如果提取到了数据，添加到结果
                if len(item) > 1:  # 至少有一个字段（除了source_url）
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"抓取URL时出错 {url}: {e}")
            return [] 