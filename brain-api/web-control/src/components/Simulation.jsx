import { useEffect, useRef, useState } from 'react'
import { World } from '../simulation/World'
import { Robot } from '../simulation/Robot'
import { Renderer } from '../simulation/Renderer'

const SERVER_URL = '/api'

export default function Simulation() {
  const worldCanvasRef = useRef(null)
  const cameraCanvasRef = useRef(null)
  const worldRef = useRef(null)
  const robotRef = useRef(null)
  const rendererRef = useRef(null)
  const animationRef = useRef(null)

  const [running, setRunning] = useState(false)
  const [autoDecide, setAutoDecide] = useState(false)
  const [decision, setDecision] = useState(null)
  const [status, setStatus] = useState('就绪')
  const [decisionInterval, setDecisionInterval] = useState(1500)
  const [commandText, setCommandText] = useState('探索')

  useEffect(() => {
    const world = new World(800, 600)
    const robot = new Robot(100, 100, 0)
    worldRef.current = world
    robotRef.current = robot

    const renderer = new Renderer(worldCanvasRef.current, cameraCanvasRef.current)
    rendererRef.current = renderer

    renderer.renderWorld(world, robot)
    renderer.renderCameraView(world, robot)

    return () => cancelAnimationFrame(animationRef.current)
  }, [])

  useEffect(() => {
    let lastDecide = 0

    const loop = () => {
      if (running) {
        robotRef.current.update(worldRef.current)
      }
      rendererRef.current.renderWorld(worldRef.current, robotRef.current)
      rendererRef.current.renderCameraView(worldRef.current, robotRef.current)

      if (autoDecide && running && Date.now() - lastDecide > decisionInterval) {
        lastDecide = Date.now()
        sendDecideRequest()
      }

      animationRef.current = requestAnimationFrame(loop)
    }

    animationRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animationRef.current)
  }, [running, autoDecide, decisionInterval])

  const sendDecideRequest = async () => {
    const img = rendererRef.current.getCameraImageBase64()
    const robot = robotRef.current
    const payload = {
      input: {
        image_base64: img,
        sonar: Math.min(robot.sonarDistance, 400),
        command_text: commandText,
      },
      enable_3d: true,
      target_angle_deg: 0,
    }

    try {
      setStatus('请求大脑...')
      const res = await fetch(`${SERVER_URL}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      setDecision(data)
      robot.setCommand(data.command, data.param, data.duration_ms)
      setStatus(`执行: ${data.command}`)
    } catch (e) {
      setStatus('请求失败: ' + e.message)
    }
  }

  const manualCommand = (cmd) => {
    robotRef.current.setCommand(cmd, 0.5, 800)
    setStatus(`手动: ${cmd}`)
  }

  const resetRobot = () => {
    robotRef.current.x = 100
    robotRef.current.y = 100
    robotRef.current.angle = 0
    robotRef.current.trail = []
    robotRef.current.setCommand('STOP')
  }

  const addRandomObstacle = () => {
    const world = worldRef.current
    const x = 100 + Math.random() * (world.width - 200)
    const y = 100 + Math.random() * (world.height - 200)
    const w = 40 + Math.random() * 80
    const h = 40 + Math.random() * 80
    const colors = ['#ef4444', '#f59e0b', '#22c55e', '#8b5cf6', '#ec4899']
    world.addObstacle({ x, y, w, h, color: colors[Math.floor(Math.random() * colors.length)] })
  }

  return (
    <div className="simulation">
      <h2>Web 仿真环境</h2>
      <p className="hint">机器人在 2D 虚拟世界中运行，相机视图实时送入 Brain API，API 返回的指令驱动机器人移动。当前为 Mock 模式。  </p>

      <div className="sim-controls">
        <button onClick={() => setRunning(!running)}>{running ? '暂停仿真' : '开始仿真'}</button>
        <button onClick={() => setAutoDecide(!autoDecide)}>{autoDecide ? '关闭自动决策' : '开启自动决策'}</button>
        <button onClick={sendDecideRequest}>手动请求大脑</button>
        <button onClick={resetRobot}>重置位置</button>
        <button onClick={addRandomObstacle}>添加随机障碍</button>
      </div>

      <div className="sim-row">
        <label>自动决策间隔 ms</label>
        <input
          type="range"
          min="500"
          max="3000"
          step="100"
          value={decisionInterval}
          onChange={(e) => setDecisionInterval(Number(e.target.value))}
        />
        <span>{decisionInterval}ms</span>
      </div>

      <div className="sim-row">
        <label>默认指令</label>
        <input value={commandText} onChange={(e) => setCommandText(e.target.value)} />
      </div>

      <div className="sim-row">
        <strong>状态:</strong> {status}
      </div>

      <div className="sim-canvases">
        <div className="canvas-panel">
          <h3>世界视图</h3>
          <canvas ref={worldCanvasRef} width={800} height={600} />
        </div>
        <div className="canvas-panel">
          <h3>机器人相机视图</h3>
          <canvas ref={cameraCanvasRef} width={224} height={224} />
        </div>
      </div>

      <div className="sim-manual">
        {['STOP', 'FORWARD', 'BACKWARD', 'TURN_LEFT', 'TURN_RIGHT', 'ROTATE_LEFT', 'ROTATE_RIGHT'].map((cmd) => (
          <button key={cmd} onClick={() => manualCommand(cmd)}>{cmd}</button>
        ))}
      </div>

      {decision && (
        <div className="sim-decision">
          <h3>最新决策</h3>
          <div>命令: <strong>{decision.command}</strong> | 参数: {decision.param} | 持续: {decision.duration_ms}ms</div>
          <div>原因: {decision.reason}</div>
          <div>语音: {decision.tts_text}</div>
          {decision.perception_3d && (
            <div>3D 提示: {decision.perception_3d.navigation_hint} | 障碍物数: {decision.perception_3d.obstacles?.length || 0}</div>
          )}
        </div>
      )}
    </div>
  )
}
