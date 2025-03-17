# 项目结构
domain_llm/
├── config/                 # 配置文件
│   ├── model_config.py     # 模型配置
│   └── train_config.py     # 训练配置
├── data/                   # 数据处理
│   ├── preprocessing/      # 数据预处理
│   ├── augmentation/       # 数据增强
│   └── dataloader.py       # 数据加载器
├── model/                  # 模型定义
│   ├── architecture.py     # 模型架构
│   ├── tokenizer.py        # 分词器
│   └── layers/             # 自定义层
├── training/               # 训练相关
│   ├── trainer.py          # 训练器
│   ├── optimizer.py        # 优化器
│   └── scheduler.py        # 学习率调度器
├── evaluation/             # 评估相关
│   ├── metrics.py          # 评估指标
│   └── evaluator.py        # 评估器
├── inference/              # 推理相关
│   ├── engine.py           # 推理引擎
│   └── serving.py          # 服务部署
├── utils/                  # 工具函数
│   ├── logging.py          # 日志
│   └── visualization.py    # 可视化
└── main.py                 # 主入口 