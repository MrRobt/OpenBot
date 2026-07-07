import { useEffect, useRef, useState } from 'react'
import Simulation from './components/Simulation'
import './App.css'

const COMMANDS = [
  'STOP', 'FORWARD', 'BACKWARD', 'TURN_LEFT', 'TURN_RIGHT',
  'AVOID_LEFT', 'AVOID_RIGHT', 'FOLLOW', 'ROTATE_LEFT', 'ROTATE_RIGHT'
]

function App() {
  const [tab, setTab] = useState('control')
  const [serverUrl, setServerUrl] = useState('/api')
  const [imageBase64, setImageBase64] = useState('')
  const [commandText, setCommandText] = useState('前进')
  const [sonar, setSonar] = useState(100)
  const [decision, setDecision] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [targetAngle, setTargetAngle] = useState(0)
  const fileRef = useRef(null)

  const api = (path) => serverUrl + path

  const checkHealth = async () => {
    try {
      const res = await fetch(api('/health'))
      setHealth(await res.json())
      setError('')
    } catch (e) {
      setHealth(null)
      setError('无法连接 Brain API: ' + e.message)
    }
  }

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 3000)
    return () => clearInterval(id)
  }, [serverUrl])

  const handleFile = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setImageBase64(ev.target.result.split(',')[1])
    reader.readAsDataURL(file)
  }

  const sendCommand = async (cmdOverride) => {
    if (!imageBase64) {
      setError('请先上传图片')
      return
    }
    setLoading(true)
    setError('')
    try {
      const payload = {
        input: {
          image_base64: imageBase64,
          sonar,
          command_text: cmdOverride || commandText
        },
        enable_3d: true,
        target_angle_deg: Number(targetAngle)
      }
      const res = await fetch(api('/decide'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setDecision(await res.json())
    } catch (e) {
      setError('决策请求失败: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  const generateImage = () => {
    const canvas = document.createElement('canvas')
    canvas.width = 224
    canvas.height = 224
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(0, 0, 224, 224)
    ctx.fillStyle = '#10b981'
    ctx.fillRect(60, 80, 100, 60)
    setImageBase64(canvas.toDataURL('image/jpeg').split(',')[1])
  }

  return (
    <div className="app">
      <header className="header">
        <h1>OpenBot Brain 控制台</h1>
        <p>Qwen-Omni + Depth Anything 3 机器人大脑 Web 控制面板</p>
      </header>

      <nav className="tabs">
        <button className={tab === 'control' ? 'active' : ''} onClick={() => setTab('control')}>控制面板</button>
        <button className={tab === 'simulation' ? 'active' : ''} onClick={() => setTab('simulation')}>Web 仿真环境</button>
      </nav>

      {tab === 'control' && (
        <>
          <section className="card">
            <h2>服务状态</h2>
            <div className="status-grid">
              <div className={`status ${health ? 'ok' : 'err'}`}>
                <span>Brain API</span>
                <strong>{health ? '在线' : '离线'}</strong>
              </div>
              {health && (
                <>
                  <div className={`status ${health.qwen ? 'ok' : 'warn'}`}>
                    <span>Qwen</span><strong>{health.qwen ? '已连接' : '未连接'}</strong>
                  </div>
                  <div className={`status ${health.tts ? 'ok' : 'warn'}`}>
                    <span>TTS</span><strong>{health.tts ? '已连接' : '未连接'}</strong>
                  </div>
                  <div className={`status ${health.da3 ? 'ok' : 'warn'}`}>
                    <span>DA3 3D感知</span><strong>{health.da3 ? '已连接' : '未连接'}</strong>
                  </div>
                </>
              )}
            </div>
            <div className="row">
              <label>后端地址</label>
              <input value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} placeholder="/api 或 http://localhost:8081" />
              <button onClick={checkHealth}>刷新状态</button>
            </div>
            {error && <div className="error">{error}</div>}
          </section>

          <section className="card">
            <h2>输入</h2>
            <div className="row">
              <label>上传图片</label>
              <input type="file" accept="image/*" ref={fileRef} onChange={handleFile} />
              <button onClick={generateImage}>生成测试图</button>
            </div>
            {imageBase64 && (
              <img src={`data:image/jpeg;base64,${imageBase64}`} alt="input" className="preview" />
            )}
            <div className="row">
              <label>语音/文本指令</label>
              <input value={commandText} onChange={(e) => setCommandText(e.target.value)} />
            </div>
            <div className="row">
              <label>超声波距离 (cm)</label>
              <input type="number" value={sonar} onChange={(e) => setSonar(Number(e.target.value))} />
            </div>
            <div className="row">
              <label>目标角度 (°)</label>
              <input type="range" min="-90" max="90" value={targetAngle} onChange={(e) => setTargetAngle(Number(e.target.value))} />
              <span>{targetAngle}°</span>
            </div>
            <button className="primary" onClick={() => sendCommand()} disabled={loading}>
              {loading ? '推理中...' : '发送决策请求'}
            </button>
          </section>

          <section className="card">
            <h2>手动控制</h2>
            <div className="cmd-grid">
              {COMMANDS.map((cmd) => (
                <button key={cmd} onClick={() => sendCommand(cmd)} disabled={loading}>
                  {cmd}
                </button>
              ))}
            </div>
          </section>

          {decision && (
            <section className="card">
              <h2>决策结果</h2>
              <div className="decision">
                <div className="big-cmd">{decision.command}</div>
                <div>速度: <strong>{decision.param}</strong></div>
                <div>持续: <strong>{decision.duration_ms}ms</strong></div>
                <div>语音: <strong>{decision.tts_text}</strong></div>
                <div>原因: {decision.reason}</div>
              </div>

              {decision.perception_3d && (
                <div className="perception">
                  <h3>3D 感知</h3>
                  <div>导航提示: <strong>{decision.perception_3d.navigation_hint}</strong></div>
                  <div>位姿: {decision.perception_3d.pose ? JSON.stringify(decision.perception_3d.pose) : '无'}</div>
                  <h4>障碍物</h4>
                  {decision.perception_3d.obstacles?.length === 0 && <p>无障碍物</p>}
                  <ul>
                    {decision.perception_3d.obstacles?.map((o, i) => (
                      <li key={i}>{o.distance_m?.toFixed(2)}m / {o.angle_deg}° / 宽 {o.width_m?.toFixed(2)}m</li>
                    ))}
                  </ul>
                  {decision.perception_3d.depth_map_base64 && (
                    <img
                      src={`data:image/png;base64,${decision.perception_3d.depth_map_base64}`}
                      alt="depth"
                      className="depth-map"
                    />
                  )}
                </div>
              )}

              {decision.tts_audio_base64 && (
                <audio controls src={`data:audio/wav;base64,${decision.tts_audio_base64}`} />
              )}
            </section>
          )}
        </>
      )}

      {tab === 'simulation' && <Simulation />}
    </div>
  )
}

export default App
