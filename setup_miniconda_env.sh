#!/bin/bash
# setup_miniconda_env.sh - 为垂直领域LLM训练框架设置Miniconda环境和项目结构

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

# 检测操作系统类型
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="MacOSX"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="Windows"
    else
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    print_message "检测到操作系统: $OS"
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
    detect_os
    
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
    print_message "检测到系统架构: $OS-$ARCH"
    
    # 设置固定版本的Miniconda下载链接，避免使用latest可能导致的问题
    if [[ "$OS" == "Windows" ]]; then
        # Windows使用exe安装程序
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Windows-x86_64.exe"
            MINICONDA_INSTALLER="miniconda.exe"
        else
            print_error "Windows平台暂不支持ARM架构"
            exit 1
        fi
    elif [[ "$OS" == "MacOSX" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-MacOSX-x86_64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-MacOSX-arm64.sh"
        fi
        MINICONDA_INSTALLER="miniconda.sh"
    else # Linux
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-aarch64.sh"
        fi
        MINICONDA_INSTALLER="miniconda.sh"
    fi
    
    print_message "将从 $MINICONDA_URL 下载Miniconda安装程序"
    
    # 提供手动下载选项
    print_message "您可以选择自动下载或手动下载Miniconda安装程序"
    read -p "是否要自动下载Miniconda? (y/n): " auto_download
    
    if [[ $auto_download == "n" || $auto_download == "N" ]]; then
        print_message "请手动下载Miniconda安装程序:"
        print_message "1. 访问 https://docs.conda.io/en/latest/miniconda.html"
        print_message "2. 下载适合您系统的安装程序: $OS-$ARCH"
        print_message "3. 将下载的文件重命名为 $MINICONDA_INSTALLER 并放在当前目录"
        read -p "完成下载后按Enter键继续..."
        
        # 检查文件是否存在
        if [[ ! -f "$MINICONDA_INSTALLER" ]]; then
            print_error "未找到 $MINICONDA_INSTALLER 文件"
            exit 1
        fi
    else
        # 尝试下载，最多重试3次
        MAX_RETRIES=3
        RETRY_COUNT=0
        DOWNLOAD_SUCCESS=false
        
        while [[ $RETRY_COUNT -lt $MAX_RETRIES && $DOWNLOAD_SUCCESS == false ]]; do
            if command -v curl &> /dev/null; then
                print_message "使用curl下载中..."
                curl -L -o "$MINICONDA_INSTALLER" "$MINICONDA_URL" --retry 3 --retry-delay 2 -#
                CURL_EXIT_CODE=$?
                if [[ $CURL_EXIT_CODE -eq 0 ]]; then
                    DOWNLOAD_SUCCESS=true
                else
                    print_warning "下载失败，退出码: $CURL_EXIT_CODE，将重试..."
                    RETRY_COUNT=$((RETRY_COUNT + 1))
                    sleep 2
                fi
            elif command -v wget &> /dev/null; then
                print_message "使用wget下载中..."
                wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL" --tries=3 --wait=2 -q --show-progress
                WGET_EXIT_CODE=$?
                if [[ $WGET_EXIT_CODE -eq 0 ]]; then
                    DOWNLOAD_SUCCESS=true
                else
                    print_warning "下载失败，退出码: $WGET_EXIT_CODE，将重试..."
                    RETRY_COUNT=$((RETRY_COUNT + 1))
                    sleep 2
                fi
            else
                print_error "未找到curl或wget，无法自动下载Miniconda"
                print_message "请手动下载Miniconda安装程序:"
                print_message "1. 访问 https://docs.conda.io/en/latest/miniconda.html"
                print_message "2. 下载适合您系统的安装程序: $OS-$ARCH"
                print_message "3. 将下载的文件重命名为 $MINICONDA_INSTALLER 并放在当前目录"
                read -p "完成下载后按Enter键继续..."
                
                # 检查文件是否存在
                if [[ ! -f "$MINICONDA_INSTALLER" ]]; then
                    print_error "未找到 $MINICONDA_INSTALLER 文件"
                    exit 1
                fi
                DOWNLOAD_SUCCESS=true
            fi
        done
        
        if [[ $DOWNLOAD_SUCCESS == false ]]; then
            print_error "多次尝试下载Miniconda安装程序失败"
            print_message "请手动下载Miniconda安装程序:"
            print_message "1. 访问 https://docs.conda.io/en/latest/miniconda.html"
            print_message "2. 下载适合您系统的安装程序: $OS-$ARCH"
            print_message "3. 将下载的文件重命名为 $MINICONDA_INSTALLER 并放在当前目录"
            read -p "完成下载后按Enter键继续..."
            
            # 检查文件是否存在
            if [[ ! -f "$MINICONDA_INSTALLER" ]]; then
                print_error "未找到 $MINICONDA_INSTALLER 文件"
                exit 1
            fi
        fi
    fi
    
    # 验证下载的文件
    if [[ "$OS" != "Windows" ]]; then
        # 检查文件大小，有效的安装脚本通常大于10MB
        FILE_SIZE=$(stat -f%z "$MINICONDA_INSTALLER" 2>/dev/null || stat -c%s "$MINICONDA_INSTALLER" 2>/dev/null)
        if [[ $FILE_SIZE -lt 10000000 ]]; then  # 小于10MB
            print_warning "下载的文件大小异常（${FILE_SIZE}字节），可能不是有效的安装程序"
            print_message "是否继续安装? (y/n): "
            read continue_install
            if [[ $continue_install != "y" && $continue_install != "Y" ]]; then
                print_message "安装已取消"
                exit 1
            fi
        fi
    fi
    
    # 安装Miniconda
    print_message "安装Miniconda"
    read -p "请输入Miniconda安装目录 [默认: $HOME/miniconda3]: " INSTALL_DIR
    INSTALL_DIR=${INSTALL_DIR:-"$HOME/miniconda3"}
    
    if [[ "$OS" == "Windows" ]]; then
        # Windows平台使用exe安装程序
        print_message "Windows平台安装说明:"
        print_message "1. 将自动启动Miniconda安装程序"
        print_message "2. 请按照安装向导进行安装"
        print_message "3. 建议安装路径: $INSTALL_DIR"
        print_message "4. 安装完成后，请关闭安装程序并按Enter键继续"
        
        # 启动安装程序
        start "$MINICONDA_INSTALLER"
        read -p "安装完成后按Enter键继续..."
        
        # 检查安装是否成功
        if [[ -d "$INSTALL_DIR" ]] || command -v conda &> /dev/null; then
            print_message "Miniconda安装成功"
        else
            print_warning "无法确认Miniconda是否安装成功，将尝试继续执行"
        fi
    else
        # Linux和MacOS使用sh安装脚本
        print_message "执行安装脚本: bash $MINICONDA_INSTALLER -b -p $INSTALL_DIR"
        bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"
        if [ $? -ne 0 ]; then
            print_error "安装Miniconda失败"
            print_message "请尝试手动安装Miniconda:"
            print_message "1. 访问 https://docs.conda.io/en/latest/miniconda.html"
            print_message "2. 下载适合您系统的安装程序"
            print_message "3. 按照官方指南安装"
            exit 1
        fi
    fi
    
    print_message "Miniconda已安装到: $INSTALL_DIR"
    
    # 删除安装程序
    rm "$MINICONDA_INSTALLER"
    
    # 配置conda
    print_message "配置conda"
    
    if [[ "$OS" == "Windows" ]]; then
        # Windows平台可能需要手动添加到PATH
        print_message "请确保Miniconda已添加到系统PATH中"
        print_message "如果在新终端中无法使用conda命令，请手动添加以下路径到系统PATH:"
        echo "  $INSTALL_DIR"
        echo "  $INSTALL_DIR/Scripts"
        echo "  $INSTALL_DIR/Library/bin"
        
        # 尝试使用conda命令
        if command -v conda &> /dev/null; then
            CONDA_PATH="conda"
        else
            print_warning "无法在PATH中找到conda命令，将尝试使用安装目录中的conda"
            CONDA_PATH="$INSTALL_DIR/Scripts/conda.exe"
            # 临时添加到PATH
            export PATH="$INSTALL_DIR:$INSTALL_DIR/Scripts:$INSTALL_DIR/Library/bin:$PATH"
        fi
    else
        # Linux和MacOS
        CONDA_PATH="$INSTALL_DIR/bin/conda"
        
        # 初始化conda
        print_message "初始化conda..."
        "$INSTALL_DIR/bin/conda" init bash
        if [[ -f "$HOME/.zshrc" ]]; then
            "$INSTALL_DIR/bin/conda" init zsh
        fi
        
        # 使conda命令在当前shell中可用
        print_message "使conda命令在当前shell中可用..."
        if [[ "$OS" == "Linux" ]]; then
            source "$HOME/.bashrc" || print_warning "无法加载 .bashrc"
        elif [[ "$OS" == "MacOSX" ]]; then
            source "$HOME/.bash_profile" || source "$HOME/.bashrc" || print_warning "无法加载配置文件"
        fi
        
        # 确保conda命令可用
        export PATH="$INSTALL_DIR/bin:$PATH"
        print_message "已将 $INSTALL_DIR/bin 添加到PATH"
    fi
    
    # 配置conda
    print_message "配置conda频道和设置..."
    "$CONDA_PATH" config --set auto_activate_base false
    "$CONDA_PATH" config --set channel_priority flexible
    "$CONDA_PATH" config --add channels conda-forge
    "$CONDA_PATH" config --add channels pytorch
    
    # 更新conda
    print_message "更新conda到最新版本..."
    "$CONDA_PATH" update -n base -c defaults conda -y
    
    print_message "conda安装和配置完成"
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
    - fastapi>=0.70.0
    - uvicorn>=0.15.0
    - pydantic>=1.8.2
    - jsonlines>=2.0.0
EOL
    print_message "conda环境文件已创建"
    
    # 创建pip配置文件，增加超时设置和镜像源
    mkdir -p $HOME/.pip
    cat > $HOME/.pip/pip.conf << 'EOL'
[global]
timeout = 300
retries = 10
default-timeout = 300
socket-timeout = 300
EOL
    print_message "已创建pip配置文件，设置超时为300秒，重试次数为10次"
}

# 安装额外依赖包
install_extra_packages() {
    print_message "安装额外依赖包..."
    
    # 询问是否安装可能导致超时的大型依赖包
    print_message "是否安装可能导致网络超时的额外依赖包(wandb, tensorboard)?"
    read -p "这些包可能导致下载超时，但可以稍后手动安装 (y/n): " install_extra
    
    if [[ $install_extra == "y" || $install_extra == "Y" ]]; then
        # 创建临时目录用于下载
        TEMP_DIR=$(mktemp -d)
        print_message "创建临时目录用于下载: $TEMP_DIR"
        
        # 尝试使用不同的方法下载wandb
        print_message "尝试下载wandb..."
        
        # 方法1: 直接pip安装，增加超时时间
        pip install wandb --timeout 600 --retries 10
        
        if [ $? -ne 0 ]; then
            print_warning "直接安装wandb失败，尝试替代方法..."
            
            # 方法2: 使用curl或wget下载wheel文件
            if command -v curl &> /dev/null; then
                print_message "使用curl下载wandb wheel文件..."
                curl -L -o "$TEMP_DIR/wandb.whl" "https://files.pythonhosted.org/packages/py3/w/wandb/wandb-0.15.12-py3-none-any.whl" --retry 5 --retry-delay 5 --connect-timeout 60 --max-time 600
                if [ $? -eq 0 ]; then
                    pip install "$TEMP_DIR/wandb.whl"
                fi
            elif command -v wget &> /dev/null; then
                print_message "使用wget下载wandb wheel文件..."
                wget -O "$TEMP_DIR/wandb.whl" "https://files.pythonhosted.org/packages/py3/w/wandb/wandb-0.15.12-py3-none-any.whl" --tries=5 --timeout=60
                if [ $? -eq 0 ]; then
                    pip install "$TEMP_DIR/wandb.whl"
                fi
            fi
            
            if [ $? -ne 0 ]; then
                print_warning "安装wandb失败，请稍后手动安装"
            else
                print_message "wandb安装成功"
            fi
        else
            print_message "wandb安装成功"
        fi
        
        # 安装tensorboard
        print_message "安装tensorboard..."
        pip install tensorboard --timeout 600 --retries 10
        
        if [ $? -ne 0 ]; then
            print_warning "安装tensorboard失败，请稍后手动安装"
        else
            print_message "tensorboard安装成功"
        fi
        
        # 清理临时目录
        rm -rf "$TEMP_DIR"
    else
        print_message "跳过安装额外依赖包，您可以稍后使用以下命令手动安装:"
        echo "  pip install wandb tensorboard"
    fi
}

# 主函数
main() {
    print_step "开始设置垂直领域LLM训练框架环境"
    
    # 检查conda是否已安装，如果没有则安装
    if ! check_conda; then
        install_conda
    fi
    
    # 检测操作系统类型
    detect_os
    
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
    
    print_step "创建conda环境: $ENV_NAME"
    
    # 创建基础conda环境
    print_message "正在创建基础环境，这可能需要几分钟..."
    
    # 检查环境是否已存在
    if conda env list | grep -q "$ENV_NAME"; then
        print_warning "环境 $ENV_NAME 已存在"
        read -p "是否要移除现有环境并重新创建? (y/n): " remove_env
        if [[ $remove_env == "y" || $remove_env == "Y" ]]; then
            print_message "移除现有环境..."
            conda env remove -n $ENV_NAME
            print_message "创建新环境..."
            conda env create -f environment.yml
        else
            print_message "将使用现有环境"
        fi
    else
        # 创建新环境
        conda env create -f environment.yml
    fi
    
    if [ $? -ne 0 ]; then
        print_error "创建conda环境失败"
        print_message "尝试分步创建环境..."
        
        # 创建基础环境
        conda create -n $ENV_NAME python=3.9 -y
        
        if [ $? -ne 0 ]; then
            print_error "创建基础环境失败，请检查网络连接"
            exit 1
        fi
        
        # 激活环境
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        
        # 分批安装依赖
        print_message "安装基础Python包..."
        conda install -y numpy pandas matplotlib seaborn
        
        print_message "安装数据处理包..."
        conda install -y nltk beautifulsoup4 requests tqdm regex
        
        print_message "安装评估和工具包..."
        conda install -y scikit-learn pyyaml
        
        print_message "安装基础pip包..."
        pip install transformers datasets accelerate --timeout 300
        
        print_message "安装NLP相关包..."
        pip install sentencepiece tokenizers rouge sacrebleu --timeout 300
        
        print_message "安装API相关包..."
        pip install fastapi uvicorn pydantic jsonlines --timeout 300
    else
        # 激活conda环境
        print_message "激活conda环境: $ENV_NAME"
        
        if [[ "$OS" == "Windows" ]]; then
            # Windows平台激活环境
            conda activate $ENV_NAME || (print_error "激活conda环境失败" && exit 1)
        else
            # Linux和MacOS激活环境
            eval "$(conda shell.bash hook)"
            conda activate $ENV_NAME
            
            if [ $? -ne 0 ]; then
                print_error "激活conda环境失败"
                exit 1
            fi
        fi
    fi
    
    print_message "conda环境已激活: $ENV_NAME"
    
    print_step "安装PyTorch"
    
    # 根据是否使用CUDA安装PyTorch
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
    
    print_message "PyTorch安装完成"
    
    # 安装额外依赖包
    install_extra_packages
    
    print_step "创建项目结构"
    
    # 创建项目目录结构（根据操作系统使用不同的命令）
    if [[ "$OS" == "Windows" ]]; then
        # Windows平台使用mkdir命令
        mkdir -p $PROJECT_DIR/config $PROJECT_DIR/data/collection $PROJECT_DIR/data/preprocessing $PROJECT_DIR/data/augmentation $PROJECT_DIR/model $PROJECT_DIR/training $PROJECT_DIR/evaluation $PROJECT_DIR/inference $PROJECT_DIR/utils $PROJECT_DIR/scripts
        mkdir -p configs logs outputs data/raw data/processed
    else
        # Linux和MacOS使用更简洁的命令
        mkdir -p $PROJECT_DIR/{config,data/{collection,preprocessing,augmentation},model,training,evaluation,inference,utils,scripts}
        mkdir -p configs logs outputs data/{raw,processed}
    fi
    
    print_message "项目结构已创建"
    
    # 导出环境配置
    conda env export > environment_full.yml
    print_message "完整环境配置已导出到 environment_full.yml"
    
    # 验证安装
    print_message "验证PyTorch安装"
    python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
    
    print_step "环境设置完成"
    print_message "conda环境: $ENV_NAME"
    print_message "项目目录: $PROJECT_DIR"
    print_message "使用以下命令激活环境:"
    echo "  conda activate $ENV_NAME"
    
    print_message "使用以下命令退出环境:"
    echo "  conda deactivate"
    
    print_message "如果您跳过了安装额外依赖包，可以稍后使用以下命令手动安装:"
    echo "  pip install wandb tensorboard --timeout 600"
    
    print_message "祝您训练顺利！"
}

# 执行主函数
main 