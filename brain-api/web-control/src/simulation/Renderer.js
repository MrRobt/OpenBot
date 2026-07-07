export class Renderer {
  constructor(worldCanvas, cameraCanvas) {
    this.worldCanvas = worldCanvas
    this.cameraCanvas = cameraCanvas
    this.worldCtx = worldCanvas.getContext('2d')
    this.cameraCtx = cameraCanvas.getContext('2d')
  }

  renderWorld(world, robot) {
    const ctx = this.worldCtx
    const w = this.worldCanvas.width
    const h = this.worldCanvas.height

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }

    // Sonar ray
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(robot.x, robot.y)
    ctx.lineTo(
      robot.x + Math.cos(robot.angle) * robot.sonarDistance,
      robot.y + Math.sin(robot.angle) * robot.sonarDistance
    )
    ctx.stroke()

    // Obstacles
    for (const obs of world.obstacles) {
      ctx.fillStyle = obs.color
      ctx.fillRect(obs.x, obs.y, obs.w, obs.h)
      ctx.strokeStyle = '#475569'
      ctx.strokeRect(obs.x, obs.y, obs.w, obs.h)
    }

    // Trail
    if (robot.trail.length > 1) {
      ctx.strokeStyle = 'rgba(14, 165, 233, 0.5)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(robot.trail[0].x, robot.trail[0].y)
      for (const p of robot.trail.slice(1)) ctx.lineTo(p.x, p.y)
      ctx.stroke()
    }

    // Robot
    ctx.save()
    ctx.translate(robot.x, robot.y)
    ctx.rotate(robot.angle)

    ctx.fillStyle = '#0ea5e9'
    ctx.beginPath()
    ctx.moveTo(20, 0)
    ctx.lineTo(-12, 10)
    ctx.lineTo(-12, -10)
    ctx.closePath()
    ctx.fill()

    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.restore()

    // FOV arc
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)'
    ctx.beginPath()
    ctx.arc(robot.x, robot.y, 120, robot.angle - 0.5, robot.angle + 0.5)
    ctx.stroke()
  }

  renderCameraView(world, robot) {
    const ctx = this.cameraCtx
    const w = this.cameraCanvas.width
    const h = this.cameraCanvas.height

    // Sky / ground
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, '#0f172a')
    grad.addColorStop(0.5, '#1e293b')
    grad.addColorStop(1, '#334155')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, w, h)

    // Cast rays across FOV and draw vertical stripes
    const fov = Math.PI / 3
    const numRays = 64
    const stripWidth = w / numRays

    for (let i = 0; i < numRays; i++) {
      const t = (i / (numRays - 1)) * 2 - 1
      const rayAngle = robot.angle + t * (fov / 2)
      const dist = world.raycast(robot.x, robot.y, rayAngle, 500)

      // Simple perspective projection
      const correctedDist = dist * Math.cos(t * (fov / 2))
      const wallHeight = Math.min(h, (3000 / (correctedDist + 1)))
      const top = (h - wallHeight) / 2

      // Color based on distance
      const brightness = Math.max(0.2, 1 - dist / 500)
      ctx.fillStyle = `rgba(100, 116, 139, ${brightness})`
      ctx.fillRect(i * stripWidth, top, stripWidth + 1, wallHeight)
    }
  }

  getCameraImageBase64() {
    return this.cameraCanvas.toDataURL('image/jpeg', 0.7).split(',')[1]
  }
}
