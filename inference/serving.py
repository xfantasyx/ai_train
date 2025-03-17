import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import torch

class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None

class BatchGenerationRequest(BaseModel):
    """批量生成请求模型"""
    prompts: List[str]
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None

class ModelServer:
    """模型服务器"""
    
    def __init__(self, inference_engine, config: Dict[str, Any]):
        self.inference_engine = inference_engine
        self.config = config
        self.app = FastAPI(title="领域LLM服务")
        
        # 注册路由
        self._register_routes()
        
    def _register_routes(self):
        """注册API路由"""
        @self.app.get("/")
        async def root():
            return {"message": "领域LLM服务已启动"}
        
        @self.app.post("/generate")
        async def generate(request: GenerationRequest):
            try:
                # 提取生成参数
                gen_kwargs = {}
                for param in ['max_length', 'temperature', 'top_p', 'top_k', 'num_beams', 'do_sample']:
                    value = getattr(request, param)
                    if value is not None:
                        gen_kwargs[param] = value
                
                # 生成文本
                response = self.inference_engine.generate(request.prompt, **gen_kwargs)
                
                return {
                    "prompt": request.prompt,
                    "response": response
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_generate")
        async def batch_generate(request: BatchGenerationRequest):
            try:
                # 提取生成参数
                gen_kwargs = {}
                for param in ['max_length', 'temperature', 'top_p', 'top_k', 'num_beams', 'do_sample']:
                    value = getattr(request, param)
                    if value is not None:
                        gen_kwargs[param] = value
                
                # 批量生成文本
                responses = self.inference_engine.batch_generate(request.prompts, **gen_kwargs)
                
                return {
                    "prompts": request.prompts,
                    "responses": responses
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
    
    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务器"""
        uvicorn.run(self.app, host=host, port=port) 