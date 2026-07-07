# OpenBot × Qwen-Omni 二次开发实施计划

> 本计划依据 [《OpenBot × Qwen-Omni 二次开发落地方案》](../设计方案/Qwen-Omni-OpenBot-二次开发落地方案.md) 制定，目标是把方案拆成可执行、可验证的具体任务。

---

## 1. 项目信息

| 项 | 内容 |
|----|------|
| 项目名称 | OpenBot × Qwen-Omni 智能机器人小车 |
| 代码路径 | `/data/code/OpenBot-MrRobt` |
| 核心模型 | Qwen2.5-Omni-3B（int4 量化） |
| 服务端显存 | 4GB |
| 网络环境 | 同一局域网 |
| 控制模式 | 高层语义指令 + 手机本地执行 |

---

## 2. 任务总览

| 阶段 | 任务 | 预计工时 | 依赖 |
|------|------|----------|------|
| **P0 环境准备** | 搭建 Python 环境与模型下载 | 0.5d | 无 |
| **P1 服务端** | 部署 Qwen + TTS + Robot Brain API | 2d | P0 |
| **P2 协议联调** | 手机-服务端数据协议打通 | 1d | P1 |
| **P3 Android 改造** | OmniBrainClient + 本地执行 + 安全层 | 3d | P2 |
| **P4 闭环验证** | 端到端实车测试与优化 | 2d | P3 |
| **P5 文档交付** | 整理部署/使用文档 | 0.5d | P4 |

---

## 3. 详细任务清单

### P0 环境准备

#### P0-1 准备局域网电脑环境

- **目标**：确保电脑能运行 Qwen2.5-Omni-3B int4 与 TTS。
- **环境要求**：
  - NVIDIA GPU（4GB 显存）
  - CUDA 12.x
  - Python 3.10+
  - Conda 或 venv
- **验证命令**：
  ```bash
  nvidia-smi
  python --version
  ```

#### P0-2 创建 Python 虚拟环境

```bash
conda create -n openbot-brain python=3.10 -y
conda activate openbot-brain
pip install --upgrade pip
```

#### P0-3 下载 Qwen2.5-Omni-3B 量化模型

```bash
# 方式 1：从 HuggingFace 下载（需要网络）
huggingface-cli download Qwen/Qwen2.5-Omni-3B-AWQ --local-dir ./models/Qwen2.5-Omni-3B-AWQ

# 方式 2：从 ModelScope 下载
modelscope download --model qwen/Qwen2.5-Omni-3B-AWQ --local_dir ./models/Qwen2.5-Omni-3B-AWQ
```

**验收标准**：模型文件存在于 `./models/Qwen2.5-Omni-3B-AWQ`。

---

### P1 服务端部署

#### P1-1 部署 Qwen2.5-Omni-3B 推理服务

- **目标**：在 `http://pc-ip:8000` 提供 OpenAI 兼容 API。
- **操作步骤**：
  1. 安装 vLLM：
     ```bash
     pip install vllm vllm[audio] --pre --extra-index-url https://wheels.vllm.ai/nightly -U
     pip install git+https://github.com/huggingface/transformers
     pip install accelerate
     ```
  2. 启动服务：
     ```bash
     vllm serve ./models/Qwen2.5-Omni-3B-AWQ \
       --omni \
       --host 0.0.0.0 \
       --port 8000 \
       --quantization awq \
       --dtype auto \
       --gpu-memory-utilization 0.85 \
       --max-model-len 4096
     ```
- **验证命令**：
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen2.5-Omni-3B-AWQ","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]}'
  ```

#### P1-2 部署 TTS 服务（GPT-SoVITS）

- **目标**：在 `http://pc-ip:8001` 提供定制音色 TTS。
- **操作步骤**：
  1. 克隆仓库：
     ```bash
     git clone https://github.com/RVC-Boss/GPT-SoVITS.git
     cd GPT-SoVITS
     pip install -r requirements.txt
     ```
  2. 准备参考音频：
     - 录制 5-10s 目标音色音频，保存为 `./voices/ref.wav`。
  3. 启动 API：
     ```bash
     python api.py \
       --default_refer_path ./voices/ref.wav \
       --default_refer_text "你好，我是你的智能小车助手" \
       --default_refer_language zh
     ```
- **验证命令**：
  ```bash
  curl -X POST http://localhost:8001/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"前方有障碍物，准备绕行","text_language":"zh"}' \
    --output test.wav
  ```

#### P1-3 开发 Robot Brain API

- **目标**：整合 Qwen + TTS，对外暴露 `/decide` 接口。
- **新建目录**：`/data/code/OpenBot-MrRobt/brain-api/`
- **核心文件**：
  - `main.py`：FastAPI 入口
  - `prompts.py`：system prompt 与解析逻辑
  - `clients.py`：Qwen client + TTS client 封装
  - `schemas.py`：Pydantic 请求/响应模型
- **接口定义**：
  - `POST /decide`
  - 请求：`RobotInput`（image_base64, sonar, command_text）
  - 响应：`RobotDecision`（command, param, duration_ms, tts_text, tts_audio_base64, reason）
- **验收标准**：用 `curl` 发送测试请求，能返回正确 JSON 和 base64 音频。

**示例启动命令**：

```bash
cd /data/code/OpenBot-MrRobt/brain-api
pip install -r requirements.txt
python main.py --host 0.0.0.0 --port 8080
```

---

### P2 协议联调

#### P2-1 定义并文档化数据协议

- **上行协议**（手机 → Brain API）：
  ```json
  {
    "image_base64": "...",
    "sonar": 45.0,
    "left_rpm": 120.0,
    "right_rpm": 118.0,
    "battery": 11.2,
    "command_text": "跟着前面的人"
  }
  ```
- **下行协议**（Brain API → 手机）：
  ```json
  {
    "command": "TURN_LEFT",
    "param": 0.5,
    "duration_ms": 800,
    "tts_text": "左转绕行",
    "tts_audio_base64": "...",
    "reason": "obstacle on the right"
  }
  ```

#### P2-2 用手机模拟器/真机发送测试请求

- 写一个简单的 Python 脚本模拟手机：
  ```bash
  python brain-api/test_client.py --image test.jpg --sonar 45 --server http://pc-ip:8080
  ```
- **验收标准**：能正确返回高层指令 + TTS 音频。

#### P2-3 测试 WebSocket 长连接稳定性

- Brain API 增加 `/ws` WebSocket 端点。
- 测试连续发送 100 帧，观察连接是否稳定。
- **验收标准**：无异常断开，平均延迟 < 1s。

---

### P3 Android 端改造

#### P3-1 新建 `OmniBrainClient`

- **文件**：`android/robot/src/main/java/org/openbot/brain/OmniBrainClient.java`
- **职责**：
  - 使用 OkHttp WebSocket 连接 Brain API。
  - 发送图像帧、传感器数据、语音/文本指令。
  - 接收并回调高层指令结果。
- **关键方法**：
  - `connect(String serverUrl)`
  - `sendDecisionRequest(byte[] jpeg, float sonar, String commandText)`
  - `setListener(OmniBrainListener listener)`

#### P3-2 新建 `LocalCommandExecutor`

- **文件**：`android/robot/src/main/java/org/openbot/brain/LocalCommandExecutor.java`
- **职责**：把高层指令映射为 `Vehicle.setControl(left, right)`。
- **命令映射表**：
  | 命令 | left | right |
  |------|------|-------|
  | STOP | 0 | 0 |
  | FORWARD | param | param |
  | BACKWARD | -param | -param |
  | TURN_LEFT | -param | param |
  | TURN_RIGHT | param | -param |

#### P3-3 新建 `SafetyArbiter`

- **文件**：`android/robot/src/main/java/org/openbot/brain/SafetyArbiter.java`
- **职责**：
  - 监测超声波距离，小于阈值时强制 STOP。
  - 监测 Qwen 指令超时，超时自动停车。
  - 监听蓝牙手柄事件，随时接管。
  - 对电机输出做范围裁剪。
- **阈值建议**：
  - 停车距离：30cm
  - 指令超时：1000ms
  - 最大速度系数：0.7

#### P3-4 新增 Omni 自动驾驶 Fragment / 模式

- **参考文件**：`android/robot/src/main/java/org/openbot/original/DefaultActivity.java`
- **改造点**：
  - 在主界面增加“Omni Brain”模式开关。
  - 开启时启动 `OmniBrainClient`，关闭时断开并停车。
  - 相机帧通过 `CameraActivity` 的回调获取，JPEG 压缩后发送。

#### P3-5 语音采集与播放

- **语音采集**：使用 Android `MediaRecorder` 或 `AudioRecord`，录音后上传 `command_text`（若做语音输入）。
- **语音播放**：收到 `tts_audio_base64` 后写入临时文件，用 `MediaPlayer` 播放。
- **文件**：`android/robot/src/main/java/org/openbot/brain/VoiceManager.java`

#### P3-6 设置页面增加 Brain API 地址配置

- **文件**：`android/robot/src/main/res/xml/preferences.xml` 及对应 Settings Fragment
- **配置项**：
  - `brain_server_url`：默认 `ws://192.168.1.100:8080/ws`
  - `brain_mode_enabled`：开关

---

### P4 闭环验证

#### P4-1 桌面联调

- 小车不装轮子/架空，只验证电机指令是否正确响应。
- 验证命令：**STOP / FORWARD / TURN_LEFT / TURN_RIGHT**。
- **验收标准**：电机按预期转动。

#### P4-2 安全功能验证

- 用手遮挡超声波，观察是否强制停车。
- 断开 WiFi，观察是否自动停车。
- 按手柄按钮，观察是否切换手动模式。

#### P4-3 端到端场景测试

| 场景 | 预期行为 |
|------|----------|
| 前方空旷 | Qwen 输出 FORWARD，小车前进 |
| 前方有障碍物 | Qwen 输出 AVOID_LEFT / STOP，并播放语音 |
| 语音指令“跟着我” | Qwen 输出 FOLLOW，小车跟随 |
| 网络延迟高 | 本地安全层超时停车 |

#### P4-4 性能优化

- 测量 Qwen 推理延迟，目标 < 500ms。
- 优化 JPEG 压缩质量与分辨率（建议 224x224，quality 70）。
- 若延迟过高，考虑降低模型精度或增加本地策略缓存。

---

### P5 文档交付

#### P5-1 编写部署文档

- 文件：`docs/设计方案/部署手册.md`
- 内容：环境安装、模型下载、服务启动、手机配置。

#### P5-2 编写使用说明

- 文件：`docs/设计方案/使用说明.md`
- 内容：开机顺序、语音指令示例、安全注意事项。

---

## 4. 关键文件清单

### 服务端

```
brain-api/
├── main.py
├── clients.py
├── prompts.py
├── schemas.py
├── requirements.txt
└── test_client.py
```

### Android 端

```
android/robot/src/main/java/org/openbot/brain/
├── OmniBrainClient.java
├── LocalCommandExecutor.java
├── SafetyArbiter.java
└── VoiceManager.java
```

### 配置文件

```
android/robot/src/main/res/xml/preferences.xml
```

---

## 5. 验收标准

| 验收项 | 通过标准 |
|--------|----------|
| Qwen 服务可用 | `curl /v1/chat/completions` 返回正常 |
| TTS 服务可用 | 输入文本能生成可播放的 `.wav` |
| Brain API 可用 | `/decide` 返回正确 JSON + base64 音频 |
| 手机能连 Brain | WebSocket 稳定，100 帧无断开 |
| 指令能驱动电机 | FORWARD / TURN_LEFT / STOP 等执行正确 |
| 安全兜底生效 | 超声波触发停车、断网停车、手柄接管 |
| 语音反馈正常 | 每次决策后播放对应 TTS |

---

## 6. 风险与应对

| 风险 | 应对 |
|------|------|
| 4GB 显存无法运行 3B AWQ | 换用 GPTQ / bitsandbytes 4-bit / CPU offload |
| Qwen 输出非标准 JSON | 增加 JSON 解析容错 + prompt 约束 |
| WebSocket 不稳定 | 增加自动重连 + 降级为 HTTP polling |
| 语音克隆效果差 | 换用更高质量参考音频或换 F5-TTS / Fish Speech |
| 小车碰撞损坏 | 始终架空测试，再低速实车验证 |

---

## 7. 下一步行动

1. 确认局域网电脑配置（GPU 型号、CUDA 版本、显存）。
2. 确认目标 TTS 方案（GPT-SoVITS / F5-TTS / 其他）。
3. 准备 5-10s 参考音频用于音色克隆。
4. 开始 P0 环境搭建。

确认后可直接进入 P1 服务端部署。
