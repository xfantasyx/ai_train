#!/bin/bash
# setup_all.sh - 一键安装Miniconda并设置LLM训练环境

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

# 检查conda是否已安装
check_conda() {
    if command -v conda &> /dev/null; then
        print_message "检测到conda已安装"
        return 0
    else
        print_message "未检测到conda，将进行安装"
        return 1
    fi
}

# 安装conda部分
install_conda() {
    print_step "开始安装Miniconda"
    
    # 检测操作系统类型
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="MacOSX"
    else
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    print_message "检测到操作系统: $OS"
    
    # 检测系统架构
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        ARCH="x86_64"
    elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        ARCH="aarch64"
    else
        print_error "不支持的系统架构: $ARCH"
        exit 1
    fi
    print_message "检测到系统架构: $ARCH"
    
    # 下载Miniconda安装脚本
    print_message "下载Miniconda安装脚本"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-$OS-$ARCH.sh"
    MINICONDA_INSTALLER="miniconda.sh"
    
    print_message "从 $MINICONDA_URL 下载Miniconda安装脚本"
    if command -v curl &> /dev/null; then
        curl -o "$MINICONDA_INSTALLER" "$MINICONDA_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"
    else
        print_error "未找到curl或wget，无法下载Miniconda"
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        print_error "下载Miniconda安装脚本失败"
        exit 1
    fi
    
    # 安装Miniconda
    print_message "安装Miniconda"
    read -p "请输入Miniconda安装目录 [默认: $HOME/miniconda3]: " INSTALL_DIR
    INSTALL_DIR=${INSTALL_DIR:-"$HOME/miniconda3"}
    
    bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"
    if [ $? -ne 0 ]; then
        print_error "安装Miniconda失败"
        exit 1
    fi
    
    print_message "Miniconda已安装到: $INSTALL_DIR"
    rm "$MINICONDA_INSTALLER"
    
    # 配置conda
    print_message "配置conda"
    CONDA_PATH="$INSTALL_DIR/bin/conda"
    
    # 初始化conda
    "$INSTALL_DIR/bin/conda" init bash
    if [[ -f "$HOME/.zshrc" ]]; then
        "$INSTALL_DIR/bin/conda" init zsh
    fi
    
    # 配置conda
    "$CONDA_PATH" config --set auto_activate_base false
    "$CONDA_PATH" config --set channel_priority flexible
    "$CONDA_PATH" config --add channels conda-forge
    "$CONDA_PATH" config --add channels pytorch
    
    # 更新conda
    "$CONDA_PATH" update -n base -c defaults conda -y
    
    print_message "conda安装和配置完成"
    
    # 使conda命令在当前shell中可用
    if [[ "$OS" == "Linux" ]]; then
        source "$HOME/.bashrc"
    elif [[ "$OS" == "MacOSX" ]]; then
        source "$HOME/.bash_profile"
    fi
    
    # 确保conda命令可用
    export PATH="$INSTALL_DIR/bin:$PATH"
}

# 创建conda环境文件
create_environment_file() {
    print_message "创建conda环境文件..."
    cat > environment.yml << 'EOL'
name: domain-llm-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  # 基础依赖
  - pip
  - numpy>=1.21.4
  - pandas>=1.3.4
  # 可视化
  - matplotlib>=3.5.0
  - seaborn>=0.11.2
  # 数据处理
  - nltk>=3.6.5
  - beautifulsoup4>=4.10.0
  - requests>=2.26.0
  - tqdm>=4.62.3
  - regex>=2021.11.10
  # 评估
  - scikit-learn>=1.0.1
  # 工具
  - pyyaml>=6.0
  # 通过pip安装的依赖
  - pip:
    - transformers>=4.20.0
    - datasets>=2.0.0
    - accelerate>=0.12.0
    - sentencepiece>=0.1.96
    - tokenizers>=0.12.1
    - rouge>=1.0.1
    - sacrebleu>=2.0.0
    - wandb>=0.12.9
    - tensorboard>=2.7.0
    - fastapi>=0.70.0
    - uvicorn>=0.15.0
    - pydantic>=1.8.2
    - jsonlines>=2.0.0
EOL
    print_message "conda环境文件已创建"
}

# 设置LLM训练环境
setup_llm_env() {
    print_step "设置LLM训练环境"
    
    # 设置变量
    ENV_NAME="domain-llm-env"
    PROJECT_DIR="domain_llm"
    
    # 询问是否使用CUDA
    read -p "是否使用CUDA进行训练? (y/n): " use_cuda
    if [[ $use_cuda == "y" || $use_cuda == "Y" ]]; then
        read -p "请输入CUDA版本 (例如: 11.3): " cuda_version
    fi
    
    # 创建环境文件
    create_environment_file
    
    # 创建conda环境
    print_message "创建conda环境: $ENV_NAME"
    conda env create -f environment.yml
    
    if [ $? -ne 0 ]; then
        print_error "创建conda环境失败"
        exit 1
    fi
    
    print_message "conda环境已创建: $ENV_NAME"
    
    # 激活conda环境
    print_message "激活conda环境: $ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    if [ $? -ne 0 ]; then
        print_error "激活conda环境失败"
        exit 1
    fi
    
    # 安装PyTorch
    print_message "安装PyTorch"
    if [[ $use_cuda == "y" || $use_cuda == "Y" ]]; then
        print_message "安装CUDA $cuda_version 版本的PyTorch..."
        
        # 根据CUDA版本选择正确的PyTorch安装命令
        if [[ $cuda_version == "11.3" ]]; then
            conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
        elif [[ $cuda_version == "11.6" ]]; then
            conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -y
        elif [[ $cuda_version == "11.7" ]]; then
            conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
        else
            print_warning "未预设CUDA $cuda_version 的PyTorch安装命令，将尝试安装"
            conda install pytorch torchvision torchaudio pytorch-cuda=$cuda_version -c pytorch -c nvidia -y
        fi
    else
        print_message "安装CPU版本的PyTorch..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    if [ $? -ne 0 ]; then
        print_error "安装PyTorch失败"
        exit 1
    fi
    
    # 创建项目结构
    print_message "创建项目结构"
    mkdir -p $PROJECT_DIR/{config,data/{collection,preprocessing,augmentation},model,training,evaluation,inference,utils,scripts}
    mkdir -p configs logs outputs data/{raw,processed}
    
    # 导出环境配置
    conda env export > environment_full.yml
    print_message "完整环境配置已导出到 environment_full.yml"
    
    # 验证安装
    print_message "验证PyTorch安装"
    python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
}

# 主函数
main() {
    print_step "开始一键安装Miniconda并设置LLM训练环境"
    
    # 检查conda是否已安装
    if ! check_conda; then
        install_conda
    fi
    
    # 设置LLM训练环境
    setup_llm_env
    
    print_step "安装完成"
    print_message "LLM训练环境已设置完成"
    print_message "使用以下命令激活环境:"
    echo "  conda activate domain-llm-env"
    
    print_message "祝您训练顺利！"
}

# 执行主函数
main 