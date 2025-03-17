# 垂直领域LLM训练框架

这是一个用于从头开始训练垂直领域大语言模型的框架，特别适用于程序开发等专业领域。

## 功能特点

- 完整的数据处理流水线：收集、清洗、增强、转换和分割
- 灵活的模型架构：支持从头训练或基于预训练模型微调
- 高效的训练流程：支持混合精度训练、梯度累积和分布式训练
- 全面的评估指标：困惑度、BLEU、ROUGE等多种评估方法
- 便捷的推理接口：交互式命令行和REST API服务

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（用于GPU训练）

### 自动环境设置（推荐）

我们提供了自动环境设置脚本，支持Windows、Linux和MacOS平台。您可以选择使用Miniconda或Anaconda作为环境管理工具。

#### 使用Miniconda（推荐，占用空间小）

```bash
# 添加执行权限
chmod +x setup_miniconda_env.sh

# 运行安装脚本
./setup_miniconda_env.sh
```

#### 使用Anaconda（包含更多预装包）

```bash
# 添加执行权限
chmod +x setup_anaconda_env.sh

# 运行安装脚本
./setup_anaconda_env.sh
```

脚本功能：
- 自动检测操作系统（Windows、Linux、MacOS）
- 检查并安装Conda（如果未安装）
- 创建名为"domain-llm-env"的Conda环境
- 安装所有必要的依赖，包括PyTorch（支持CPU或GPU版本）
- 创建项目目录结构
- 导出完整环境配置
- 处理网络问题，提供手动下载选项
- 分批安装依赖，避免网络超时问题

#### 脚本特性

- **自动检测系统**：自动识别Windows、Linux和MacOS系统，以及x86_64和ARM架构
- **智能错误处理**：提供多种下载方式和错误恢复机制
- **网络优化**：增加下载超时设置，支持断点续传
- **用户友好**：提供详细的安装指导和进度提示
- **灵活安装**：支持自动下载或手动下载安装程序
- **环境管理**：检测已存在的环境，提供重建选项

#### Windows用户注意事项

Windows用户可以在Git Bash、WSL或PowerShell中运行此脚本。如果使用PowerShell，请确保已启用脚本执行权限：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 激活环境

安装完成后，使用以下命令激活环境：

```bash
conda activate domain-llm-env
```

退出环境：

```bash
conda deactivate
```

### 手动安装依赖（替代方法）

如果您不想使用自动脚本，也可以手动安装依赖：

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 数据收集

使用数据收集脚本从各种来源收集数据：

```bash
python scripts/collect_data.py --config_path configs/collect_config.json --output_dir ./data/raw
```

配置文件示例：

```json
{
  "sources": [
    {
      "type": "github",
      "name": "python_examples",
      "repo": "username/repo",
      "path": "examples",
      "token": "your_github_token"
    },
    {
      "type": "api",
      "name": "code_qa",
      "url": "https://api.example.com/code_qa",
      "headers": {
        "Authorization": "Bearer your_api_token"
      }
    }
  ]
}
```

### 2. 数据处理

使用数据处理脚本处理收集到的数据：

```bash
python scripts/process_data.py --input_path ./data/raw --output_dir ./data/processed --check_quality --clean_data --convert_format --target_format instruction --split_data
```

### 3. 模型训练

使用训练脚本训练模型：

```bash
python scripts/train.py --model_size base --batch_size 8 --epochs 3 --train_data ./data/processed/train.json --val_data ./data/processed/val.json --output_dir ./outputs --use_amp
```

### 4. 模型评估

使用评估脚本评估模型：

```bash
python scripts/evaluate.py --model_path ./outputs/final_model.pt --test_data ./data/processed/test.json --output_dir ./evaluation --generate
```

### 5. 模型推理

使用推理脚本进行交互式推理：

```bash
python scripts/infer.py --model_path ./outputs/final_model.pt --interactive
```

或者批量处理文件：

```bash
python scripts/infer.py --model_path ./outputs/final_model.pt --input_file ./prompts.txt --output_file ./responses.txt
```

### 6. 服务部署

使用服务脚本部署REST API服务：

```bash
python scripts/serve.py --model_path ./outputs/final_model.pt --port 8000
```

## 配置说明

### 模型配置

可以在`config/model_config.py`中修改模型配置，包括：

- 模型大小（small、base、large）
- 隐藏层大小
- 注意力头数量
- 最大位置编码长度
- 词汇表大小
- 特殊token

### 训练配置

可以在`config/train_config.py`中修改训练配置，包括：

- 批次大小
- 学习率
- 优化器类型
- 学习率调度器
- 梯度累积步数
- 混合精度训练
- 检查点保存频率

## 目录结构 