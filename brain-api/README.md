# OpenBot Brain API

Robot Brain API 是 OpenBot × Qwen-Omni 项目的局域网服务端中枢，负责整合：

- **Qwen2.5-Omni-3B** 多模态决策（端口 8000）
- **TTS** 定制音色语音合成（端口 8001）
- **Depth Anything 3** 3D 感知与导航（端口 8002）

Brain API 本身监听 **8080** 端口，对外暴露 `/decide` 接口。

## 目录结构

```
brain-api/
├── main.py                  # FastAPI 入口
├── clients.py               # Qwen / TTS / DA3 客户端
├── prompts.py               # System prompt 与 JSON 解析
├── schemas.py               # Pydantic 请求/响应模型
├── requirements.txt         # Python 依赖
├── test_client.py           # 测试客户端
├── download_da3.sh          # DA3 模型下载脚本
└── perception/              # Depth Anything 3 3D 感知模块
    ├── da3_service.py       # 3D 感知 FastAPI 服务
    ├── depth_estimator.py   # 深度估计
    ├── pose_tracker.py      # 相机位姿估计
    ├── mapping_3d.py        # 局部点云建图
    └── navigator.py         # 障碍物检测与导航
```

## 快速开始

### 1. 安装依赖

```bash
cd brain-api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 下载 Depth Anything 3 模型

```bash
bash download_da3.sh
```

> 脚本默认使用 `https://hf-mirror.com`。如需官方源，可设置 `HF_ENDPOINT=https://huggingface.co`。

### 3. 启动四个服务

```bash
# Terminal 1: Qwen2.5-Omni
vllm serve Qwen/Qwen2.5-Omni-3B-AWQ \
  --omni --host 0.0.0.0 --port 8000 \
  --quantization awq --dtype auto

# Terminal 2: TTS (以 GPT-SoVITS 为例)
cd /path/to/GPT-SoVITS
python api.py --default_refer_path ./voices/ref.wav \
  --default_refer_text "参考文本" --default_refer_language zh

# Terminal 3: Depth Anything 3
cd brain-api
python -m perception.da3_service

# Terminal 4: Brain API
cd brain-api
python main.py
```

### 4. 测试

```bash
python test_client.py --image test.jpg --command "前进" --sonar 100
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_BASE_URL` | `http://localhost:8000` | Qwen vLLM 服务地址 |
| `TTS_BASE_URL` | `http://localhost:8001` | TTS 服务地址 |
| `DA3_BASE_URL` | `http://localhost:8002` | DA3 3D 感知服务地址 |

## 接口

### `GET /health`

返回 Qwen / TTS / DA3 连接状态。

### `POST /decide`

请求示例：

```json
{
  "input": {
    "image_base64": "/9j/4AAQ...",
    "sonar": 45.0,
    "command_text": "跟着前面的人"
  },
  "enable_3d": true,
  "target_angle_deg": 0.0
}
```

响应示例：

```json
{
  "command": "TURN_LEFT",
  "param": 0.5,
  "duration_ms": 800,
  "tts_text": "左转绕行",
  "tts_audio_base64": "UklGRiQ...",
  "reason": "obstacle on the right",
  "perception_3d": {
    "navigation_hint": "FREE_LEFT",
    "obstacles": [
      {"distance_m": 0.8, "angle_deg": -15, "width_m": 0.3, "label": "obstacle"}
    ],
    "pose": {"x": 0, "y": 0, "z": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0}
  }
}
```
