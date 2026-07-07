# OpenBot × Qwen-Omni 二次开发落地方案

## 1. 项目背景与目标

基于 `MrRobt/OpenBot`（Intel OpenBot Fork）进行二次开发，目标是把 **Qwen-Omni 多模态大模型** 作为小车的“大脑”，部署在局域网电脑中；OpenBot 手机端负责感知采集、本地执行和安全兜底，实现可语音交互的自主导航驾驶能力。

### 1.1 核心目标

1. **局域网离线大脑**：Qwen2.5-Omni-3B 部署在局域网电脑，不依赖公网云端。
2. **高层指令输出**：Qwen 只输出高层语义指令（如 `TURN_LEFT`、`FOLLOW`），由手机端本地映射为底层电机控制。
3. **语音交互与定制音色**：集成 TTS 服务，支持通过参考音频定制音色，实现自然语音反馈。
4. **安全兜底**：超声波避障、超时停车、手柄接管等多重安全机制。

### 1.2 关键约束

| 约束项 | 内容 |
|--------|------|
| 模型 | Qwen2.5-Omni-3B |
| 服务端显存 | 4GB |
| 控制模式 | 高层指令 + 本地执行 |
| 语音 | 需要 TTS，支持定制音色 |
| 网络 | 同一局域网 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      局域网电脑（4GB GPU）                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │ Qwen2.5-Omni-3B         │    │ TTS 服务（定制音色）         │ │
│  │ int4 量化 / vLLM 服务    │    │ GPT-SoVITS / F5-TTS         │ │
│  │ http://pc-ip:8000        │    │ http://pc-ip:8001            │ │
│  └───────────┬─────────────┘    └──────────────┬──────────────┘ │
│              │                                  │                │
│              └──────────────┬───────────────────┘                │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │ Robot Brain API     │                        │
│                  │ FastAPI  orchestrator│                       │
│                  │ http://pc-ip:8080   │                        │
│                  └──────────┬──────────┘                        │
└─────────────────────────────┼───────────────────────────────────┘
                              │ WiFi
                              ▼
                   ┌─────────────────────┐
                   │   OpenBot 手机       │
                   │  感知 + 本地执行层   │
                   │                     │
                   │  · 相机帧采集        │
                   │  · 传感器读取        │
                   │  · 语音采集/播放     │
                   │  · 命令解释器        │
                   │  · 安全仲裁层        │
                   └──────────┬──────────┘
                              │ USB / BLE
                              ▼
                        Arduino / 小车电机
```

### 2.1 分层职责

| 层级 | 位置 | 职责 | 频率 |
|------|------|------|------|
| 大脑决策 | 局域网电脑 | 接收图像/传感器/语音，输出高层指令 + TTS 文本 | 0.5~1 Hz |
| 感知采集 | OpenBot 手机 | 相机、麦克风、超声波、里程计 | 持续 |
| 本地执行 | OpenBot 手机 | 指令映射、电机控制、避障、急停 | 20~50 Hz |
| 运动执行 | Arduino/小车 | PWM 驱动电机 | 实时 |

---

## 3. 局域网服务端设计

### 3.1 Qwen2.5-Omni-3B 部署

由于显存只有 4GB，必须做 4-bit 量化。

**推荐方式：vLLM + AWQ/GPTQ int4**

```bash
vllm serve Qwen/Qwen2.5-Omni-3B-AWQ \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**备选方式**

| 方案 | 说明 |
|------|------|
| transformers + bitsandbytes 4-bit | 简单易部署，但吞吐较低 |
| llama.cpp GGUF Q4_K_M | 显存占用最低，但多模态支持可能不完整 |
| vLLM GPTQ | 若 AWQ 不可得，可用 GPTQ |

### 3.2 TTS 服务（定制音色）

4GB 显存已被 Qwen 占用，TTS 建议跑 CPU。

**推荐：GPT-SoVITS（中文/英文，5-10s 参考音频克隆）**

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
python api.py \
  --default_refer_path ./voices/ref.wav \
  --default_refer_text "这是一段参考文本，用于音色克隆" \
  --default_refer_language zh
```

**备选**

| TTS | 特点 |
|-----|------|
| F5-TTS | 轻量、支持参考音频克隆 |
| XTTS v2 | 英文效果好，显存需求高 |
| Fish Speech | 中文自然，可 CPU |

### 3.3 Robot Brain API（整合层）

FastAPI 服务，统一对外暴露一个接口 `/decide`，内部调用 Qwen + TTS。

**功能**

- 接收手机上传的图像、传感器、语音/文本指令。
- 构造 system prompt + user prompt，调用 Qwen。
- 解析 Qwen 返回的 JSON，提取高层指令与 TTS 文本。
- 调用 TTS 服务生成音频，base64 编码后一并返回。

---

## 4. OpenBot Android 端改造

### 4.1 新增 `OmniBrainClient`

独立模块，通过 WebSocket 或 HTTP 连接 Robot Brain API，不破坏原有 `PhoneController`。

```java
public class OmniBrainClient {
    private final String serverUrl = "ws://<pc-ip>:8080/ws";
    private WebSocket webSocket;

    public void connect();
    public void sendFrame(byte[] jpeg, float sonar, String voiceCmd);
    public void disconnect();
}
```

### 4.2 本地命令执行器（`LocalCommandExecutor`）

把 Qwen 返回的高层指令映射为左右电机速度，调用 `Vehicle.setControl(left, right)`。

```java
public class LocalCommandExecutor {
    private Vehicle vehicle;

    public void execute(String command, float param) {
        switch (command) {
            case "STOP":       vehicle.stopBot(); break;
            case "FORWARD":    vehicle.setControl(param, param); break;
            case "BACKWARD":   vehicle.setControl(-param, -param); break;
            case "TURN_LEFT":  vehicle.setControl(-param, param); break;
            case "TURN_RIGHT": vehicle.setControl(param, -param); break;
            case "ROTATE_LEFT":vehicle.setControl(-param, param); break;
            case "ROTATE_RIGHT":vehicle.setControl(param, -param); break;
        }
    }
}
```

### 4.3 安全仲裁层

优先级：**安全规则 > 手柄输入 > Qwen 指令**

| 场景 | 动作 |
|------|------|
| 超声波距离 < 30cm | 强制 STOP，忽略大脑指令 |
| 超过 1s 未收到新指令 | 自动停车 |
| 蓝牙手柄按键触发 | 立即切换手动模式 |
| 电机输出越界 | 裁剪到安全范围 |
| 网络断开 | 停车并提示 |

### 4.4 语音播放

收到 `tts_audio_base64` 后，用 Android `MediaPlayer` 播放。

---

## 5. 数据协议

### 5.1 上行：手机 → Robot Brain API

```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQ...",
  "sonar": 45.0,
  "left_rpm": 120.0,
  "right_rpm": 118.0,
  "battery": 11.2,
  "command_text": "跟着前面的人"
}
```

### 5.2 下行：Robot Brain API → 手机

```json
{
  "command": "TURN_LEFT",
  "param": 0.5,
  "duration_ms": 800,
  "tts_text": "左转绕行",
  "tts_audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "reason": "obstacle ahead on the right"
}
```

### 5.3 命令集

| 命令 | 含义 | 参数说明 |
|------|------|----------|
| STOP | 停止 | 无 |
| FORWARD | 前进 | param = 速度 0.0~1.0 |
| BACKWARD | 后退 | param = 速度 0.0~1.0 |
| TURN_LEFT | 左转 | param = 转向强度 |
| TURN_RIGHT | 右转 | param = 转向强度 |
| AVOID_LEFT | 向左避让 | param = 强度 |
| AVOID_RIGHT | 向右避让 | param = 强度 |
| FOLLOW | 跟随目标 | param = 速度 |
| ROTATE_LEFT | 原地左转 | param = 速度 |
| ROTATE_RIGHT | 原地右转 | param = 速度 |

---

## 6. Qwen Prompt 设计

```text
You are the brain of a small robot car. You receive a camera image and sensor data.
Your job is to output ONE high-level driving command as valid JSON.

Allowed commands: STOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, AVOID_LEFT, AVOID_RIGHT, FOLLOW, ROTATE_LEFT, ROTATE_RIGHT.

Output format:
{
  "command": "TURN_LEFT",
  "param": 0.5,
  "duration_ms": 800,
  "tts_text": "左转绕行",
  "reason": "obstacle on the right"
}

Rules:
- param must be between 0.0 and 1.0.
- duration_ms should be between 200 and 2000.
- tts_text should be a short Chinese phrase.
- If sonar distance is very small, prefer STOP or AVOID.
```

---

## 7. 部署与启动流程

### 7.1 电脑端

```bash
# 1. 启动 Qwen 服务
vllm serve Qwen/Qwen2.5-Omni-3B-AWQ \
  --omni --host 0.0.0.0 --port 8000 \
  --quantization awq --dtype auto \
  --gpu-memory-utilization 0.85

# 2. 启动 TTS 服务（CPU）
cd GPT-SoVITS
python api.py --default_refer_path ./voices/ref.wav \
  --default_refer_text "参考文本" --default_refer_language zh

# 3. 启动 Robot Brain API
cd /path/to/brain-api
python main.py --host 0.0.0.0 --port 8080
```

### 7.2 手机端

1. 安装修改后的 OpenBot App。
2. 连接小车（USB/BLE）。
3. 在设置中填入 Robot Brain API 地址（如 `ws://192.168.1.100:8080/ws`）。
4. 启动 Omni 自动驾驶模式。

---

## 8. 里程碑

| 阶段 | 目标 | 产出 |
|------|------|------|
| M1 | 环境搭建 | Qwen + TTS + Brain API 在局域网跑通 |
| M2 | 协议联调 | 手机能上传图像/传感器，能接收指令/音频 |
| M3 | 本地执行 | 命令解释器 + 安全仲裁 + 电机控制 |
| M4 | 端到端闭环 | 小车能按高层指令移动并语音反馈 |
| M5 | 优化迭代 | 降低延迟、提升稳定性、增加命令集 |

---

## 9. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 4GB 显存无法承载 3B 模型 | 部署失败 | 使用 int4 量化，必要时用 CPU offload |
| Qwen 推理延迟高 | 控制不连贯 | 高层指令 + 本地持续执行，超时停车 |
| 网络抖动/断开 | 小车失控 | 心跳检测，断网自动 STOP |
| 语音克隆质量差 | 体验差 | 准备高质量参考音频，必要时换 TTS 方案 |
| 泛化能力差 | 遇到未训练场景失效 | 增加 system prompt 约束 + 安全规则兜底 |

---

## 10. 参考链接

- OpenBot 仓库：`/data/code/OpenBot-MrRobt`
- 关键接入点：
  - `android/robot/src/main/java/org/openbot/env/PhoneController.java:128`
  - `android/robot/src/main/java/org/openbot/vehicle/Vehicle.java:286`
  - `android/robot/src/main/java/org/openbot/tflite/Autopilot.java:69`
