#!/bin/bash
# setup_env.sh - 为垂直领域LLM训练框架设置虚拟环境和项目结构

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
    echo -e "${BLUE}==================================================${NC}"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未找到。请先安装 $1。"
        exit 1
    fi
}

# 检查Python版本
check_python_version() {
    python_version=$(python --version 2>&1 | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
        print_error "需要Python 3.8或更高版本。当前版本: $python_version"
        exit 1
    fi
    
    print_message "Python版本检查通过: $python_version"
}

# 创建requirements.txt文件
create_requirements_file() {
    print_message "创建requirements.txt文件..."
    cat > requirements.txt << 'EOL'
# 核心依赖
torch>=1.10.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.12.0
sentencepiece>=0.1.96
tokenizers>=0.12.1

# 数据处理
nltk>=3.6.5
beautifulsoup4>=4.10.0
requests>=2.26.0
tqdm>=4.62.3
pandas>=1.3.4
numpy>=1.21.4
regex>=2021.11.10
rouge>=1.0.1

# 评估
scikit-learn>=1.0.1
sacrebleu>=2.0.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.2

# 日志和实验跟踪
wandb>=0.12.9
tensorboard>=2.7.0

# API服务
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.2

# 工具
PyYAML>=6.0
jsonlines>=2.0.0
EOL
    print_message "requirements.txt文件已创建"
}

# 主函数
main() {
    print_step "开始设置垂直领域LLM训练框架环境"
    
    # 检查必要的命令
    check_command python
    check_python_version
    
    # 设置变量
    ENV_NAME="domain-llm-env"
    PROJECT_DIR="domain_llm"
    
    # 询问是否使用CUDA
    read -p "是否使用CUDA进行训练? (y/n): " use_cuda
    if [[ $use_cuda == "y" || $use_cuda == "Y" ]]; then
        read -p "请输入CUDA版本 (例如: 11.3): " cuda_version
    fi
    
    print_step "创建虚拟环境: $ENV_NAME"
    
    # 创建虚拟环境
    python -m venv $ENV_NAME
    
    if [ $? -ne 0 ]; then
        print_error "创建虚拟环境失败"
        exit 1
    fi
    
    print_message "虚拟环境已创建: $ENV_NAME"
    
    # 激活虚拟环境
    source $ENV_NAME/bin/activate
    
    if [ $? -ne 0 ]; then
        print_error "激活虚拟环境失败"
        exit 1
    fi
    
    print_message "虚拟环境已激活: $ENV_NAME"
    
    # 创建requirements.txt
    create_requirements_file
    
    print_step "安装依赖"
    
    # 如果使用CUDA，先安装对应版本的PyTorch
    if [[ $use_cuda == "y" || $use_cuda == "Y" ]]; then
        print_message "安装CUDA $cuda_version 版本的PyTorch..."
        
        # 根据CUDA版本选择正确的PyTorch安装命令
        if [[ $cuda_version == "11.3" ]]; then
            pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
        elif [[ $cuda_version == "11.6" ]]; then
            pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
        elif [[ $cuda_version == "11.7" ]]; then
            pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
        else
            print_warning "未预设CUDA $cuda_version 的PyTorch安装命令，将安装最新版本"
            pip install torch torchvision torchaudio
        fi
    else
        print_message "安装CPU版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安装其他依赖
    print_message "安装其他依赖..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "安装依赖失败"
        exit 1
    fi
    
    print_message "依赖安装完成"
    
    print_step "创建项目结构"
    
    # 创建项目目录结构
    mkdir -p $PROJECT_DIR/{config,data/{collection,preprocessing,augmentation},model,training,evaluation,inference,utils,scripts}
    mkdir -p configs logs outputs data/{raw,processed}
    
    print_message "项目结构已创建"
    
    # 导出环境配置
    pip freeze > environment.yml
    print_message "环境配置已导出到 environment.yml"
    
    print_step "环境设置完成"
    print_message "虚拟环境: $ENV_NAME"
    print_message "项目目录: $PROJECT_DIR"
    print_message "使用以下命令激活环境:"
    echo "  source $ENV_NAME/bin/activate"
    
    print_message "使用以下命令退出环境:"
    echo "  deactivate"
    
    print_message "祝您训练顺利！"
}

# 执行主函数
main 