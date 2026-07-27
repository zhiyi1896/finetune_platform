# finetune_plat

一个面向本地开发的大模型 LoRA 微调、量化与部署平台原型。项目使用
FastAPI 提供后端接口，使用 Gradio 提供可视化界面，并将数据处理、LoRA
微调、模型量化、实验记录和 Docker 镜像构建组织为一条工作流。

> 当前项目适合学习、实验和二次开发，不建议直接用于生产环境。模型权重
> 不包含在 Git 仓库中。

## 功能

- 上传并处理 JSON、JSONL、CSV 格式的数据集
- 将数据划分为训练集和验证集
- 基于 PEFT 对因果语言模型进行 LoRA 微调
- 使用 bitsandbytes 以 INT4、INT8 或 FP16 方式加载模型
- 通过本地 JSON 文件记录实验参数
- 生成 FastAPI 推理服务模板
- 尝试将完整模型和推理服务打包为 Docker 镜像
- 使用命令行脚本与微调模型进行多轮对话

## 技术栈

- Python 3.12+
- PyTorch
- Transformers
- PEFT
- Accelerate
- bitsandbytes
- FastAPI + Uvicorn
- Gradio
- Docker SDK for Python
- uv

## 项目结构

```text
finetune_plat/
├── api/
│   └── main.py                    # 平台 REST API
├── data/                          # 数据集与处理结果
├── models/                        # 本地模型、LoRA 和量化产物
├── modules/
│   ├── data/data_processor.py     # 数据加载、清洗、划分和保存
│   ├── finetune/finetuner.py      # LoRA 微调
│   ├── quantization/quantizer.py  # INT4、INT8、FP16 加载与保存
│   ├── deployment/deployer.py     # 推理 API 和 Docker 镜像构建
│   └── management/manager.py      # 实验记录和模型版本管理
├── scripts/start.ps1              # Windows 启动脚本
├── storage/                       # 本地实验与模型版本记录
├── ui/app.py                      # Gradio 界面
├── chat_finetuned.py              # 基础模型 + LoRA 对话
├── chat_finetuned_int4.py         # INT4 模型对话示例
├── config.py                      # 环境变量与默认配置
├── pyproject.toml                 # 项目元数据和依赖
└── uv.lock                        # uv 锁文件
```

## 工作流程

```text
上传数据
   ↓
清洗并划分数据集
   ↓
加载基础模型并挂载 LoRA
   ↓
训练并保存 LoRA adapter
   ↓
量化加载或合并后量化
   ↓
生成推理 API 并构建 Docker 镜像
```

LoRA 默认只训练少量增量权重。`PeftModel.from_pretrained()` 是在基础模型上
动态挂载 adapter；只有调用 `merge_and_unload()` 才会生成不再依赖 adapter
的完整合并模型。

## 环境要求

- Python 3.12 或更高版本
- 推荐使用支持 CUDA 的 NVIDIA GPU
- INT4、INT8 功能需要 bitsandbytes 和兼容的运行环境
- Docker 构建功能需要本机已安装并启动 Docker Desktop 或 Docker Engine

当前依赖版本见 [pyproject.toml](pyproject.toml)。

## 安装

推荐使用 uv：

```powershell
git clone https://github.com/<username>/finetune_plat.git
cd finetune_plat
uv sync
```

确认 PyTorch 可以识别 GPU：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果项目安装的 PyTorch 与本机 CUDA 环境不匹配，请按照 PyTorch 官方说明
安装适合当前驱动的构建版本。

## 准备基础模型

模型权重通常超过 GitHub 单文件大小限制，因此 `models/` 下的权重文件不会
提交到仓库。请将 Hugging Face 格式的基础模型放在本地，例如：

```text
models/
└── Qwen2___5-0___5B-Instruct/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

也可以通过环境变量指定其他本地模型目录或 Hugging Face 模型名称：

```powershell
$env:DEFAULT_MODEL = "D:\models\Qwen2.5-0.5B-Instruct"
```

## 数据格式

平台支持 `.json`、`.jsonl` 和 `.csv`。训练记录采用以下结构：

```json
{
  "system": "你是一个乐于助人的助手",
  "conversation": [
    {
      "human": "你好",
      "assistant": "你好，有什么可以帮助你的？"
    }
  ]
}
```

CSV 文件至少应包含 `system` 和 `conversation` 两列，其中 `conversation`
是上述对话列表的字符串表示。

## 启动

API 和 UI 需要在两个终端中分别启动。

终端一：

```powershell
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

终端二：

```powershell
uv run python ui/app.py
```

Gradio UI：

```text
http://127.0.0.1:7864
```

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/upload-data` | 上传数据文件 |
| `POST` | `/process-data` | 清洗并划分数据 |
| `POST` | `/finetune` | 执行 LoRA 微调 |
| `POST` | `/quantize` | 以指定精度处理模型 |
| `POST` | `/deploy` | 构建模型推理镜像 |
| `GET` | `/experiments` | 获取实验记录 |
| `GET` | `/` | 获取服务信息 |

微调、量化和镜像构建是耗时操作。当前版本会在 API 请求进程中同步执行，
尚未实现后台任务队列、进度查询和任务取消。

## LoRA 推理

微调完成后，默认 adapter 目录类似：

```text
models/Qwen2___5-0___5B-Instruct_finetuned/
```

使用基础模型和 LoRA adapter 进行对话：

```powershell
uv run python chat_finetuned.py
```

动态 INT4 推理的推荐结构是：

```text
INT4 基础模型 + FP16/FP32 LoRA adapter
```

即在 `AutoModelForCausalLM.from_pretrained()` 中通过
`BitsAndBytesConfig(load_in_4bit=True)` 加载基础模型，再通过
`PeftModel.from_pretrained()` 挂载 LoRA。

## 量化产物

量化有两种不同目标：

1. 动态挂载：量化基础模型后加载 LoRA adapter，效果灵活，但部署时需要同时
   提供基础模型和 adapter。
2. 独立模型：先通过 `merge_and_unload()` 合并基础模型与 LoRA，再量化并保存
   完整模型，部署简单，但 LoRA 增量也会受到量化误差影响。

请根据实际保存内容区分“完整量化模型”和“量化基础模型 + adapter”，不要仅
根据目录名中的 `_int4` 判断模型类型。

## Docker 部署

部署模块的目标是构建包含以下内容的 Docker 镜像：

```text
Python运行环境
+ 推理依赖
+ 完整模型目录
+ FastAPI推理服务
```

当前部署模块仍属于实验实现。使用前需要确认：

- 输入目录是可独立加载的完整模型，或推理模板已支持基础模型加 adapter
- 构建上下文内存在部署专用 `requirements.txt`
- 推理 API 在执行 `docker build` 前生成并复制到上下文
- Dockerfile 的 Python、PyTorch 和 CUDA 环境相互兼容
- 镜像构建成功不等于容器已经启动

## 配置

主要环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `data` | 数据目录 |
| `DEFAULT_MODEL` | 本地 Qwen2.5-0.5B 目录 | 基础模型 |
| `DEFAULT_LORA_RANK` | `8` | LoRA rank |
| `DEFAULT_EPOCHS` | `3` | 训练轮数 |
| `DEFAULT_LEARNING_RATE` | `1e-4` | 学习率 |
| `DEFAULT_PRECISION` | `int4` | 默认量化精度 |
| `DEFAULT_IMAGE_NAME` | `lora-finetuned-model` | Docker 镜像名 |
| `API_HOST` | `0.0.0.0` | API 地址 |
| `API_PORT` | `8000` | API 端口 |
| `UI_HOST` | `127.0.0.1` | UI 地址 |
| `UI_PORT` | `7860` | 配置中的 UI 端口 |

注意：当前 `ui/app.py` 实际使用端口 `7864`，尚未读取 `UI_PORT`。

## 当前限制

- README 所描述的量化与部署流程仍需要进一步完善和自动化测试
- 微调接口尚未使用验证集执行评估
- 训练目标未屏蔽 system、user 和 padding 区域
- 实验记录使用本地 JSON，不适合并发写入
- 上传路径和模型路径需要增加安全校验
- 模型版本管理尚未接入 API 和 UI
- 项目暂未提供自动化测试和 CI

## 安全提示

不要向公开仓库提交以下内容：

- 模型权重和 LoRA 训练产物
- `.env`、访问令牌和云服务凭据
- 包含隐私或未授权内容的数据集
- 本地实验记录和绝对路径

在将服务暴露到公网前，请增加身份验证、请求大小限制、路径校验、任务隔离和
资源配额。
