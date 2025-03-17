import os
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import wandb

class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 设置混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) else None
        
        # 设置梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # 设置检查点保存
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置日志
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'domain-llm'),
                name=config.get('wandb_run_name', f'run-{time.strftime("%Y%m%d-%H%M%S")}'),
                config=config
            )
    
    def train(self, epochs: int) -> Dict[str, List[float]]:
        """训练模型"""
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # 训练一个epoch
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)
            
            # 验证
            if self.val_loader is not None:
                val_loss = self._validate()
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.config.get('save_every', 1) == 0:
                self._save_checkpoint(epoch + 1)
                
            # 记录到wandb
            if self.use_wandb:
                log_dict = {'epoch': epoch + 1, 'train_loss': train_loss}
                if self.val_loader is not None:
                    log_dict['val_loss'] = val_loss
                wandb.log(log_dict)
                
        return history
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 混合精度训练
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.get('max_grad_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['max_grad_norm']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.get('max_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['max_grad_norm']
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
            
            # 更新统计信息
            batch_size = batch['input_ids'].size(0)
            total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
        return total_loss / total_samples
    
    def _validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating")
            
            for batch in progress_bar:
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 更新统计信息
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
                
        return total_loss / total_samples
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint.get('epoch', 0) 