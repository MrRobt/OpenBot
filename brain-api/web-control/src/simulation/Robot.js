export class Robot {
  constructor(x = 100, y = 100, angle = 0) {
    this.x = x
    this.y = y
    this.angle = angle
    this.radius = 15
    this.speed = 0
    this.angularSpeed = 0
    this.maxSpeed = 4
    this.maxAngularSpeed = 0.08
    this.command = 'STOP'
    this.commandParam = 0
    this.commandDuration = 0
    this.trail = []
    this.sonarDistance = 400
  }

  setCommand(command, param = 0.5, durationMs = 800) {
    this.command = command
    this.commandParam = param
    this.commandDuration = durationMs
    this.commandStartTime = Date.now()

    const speed = this.maxSpeed * param
    const turn = this.maxAngularSpeed * param

    switch (command) {
      case 'FORWARD':
        this.speed = speed
        this.angularSpeed = 0
        break
      case 'BACKWARD':
        this.speed = -speed
        this.angularSpeed = 0
        break
      case 'TURN_LEFT':
        this.speed = speed * 0.3
        this.angularSpeed = -turn
        break
      case 'TURN_RIGHT':
        this.speed = speed * 0.3
        this.angularSpeed = turn
        break
      case 'ROTATE_LEFT':
        this.speed = 0
        this.angularSpeed = -turn
        break
      case 'ROTATE_RIGHT':
        this.speed = 0
        this.angularSpeed = turn
        break
      case 'STOP':
      default:
        this.speed = 0
        this.angularSpeed = 0
    }
  }

  update(world) {
    if (this.commandDuration > 0 && Date.now() - this.commandStartTime > this.commandDuration) {
      this.setCommand('STOP')
      this.commandDuration = 0
    }

    const prevX = this.x
    const prevY = this.y

    this.angle += this.angularSpeed
    this.x += Math.cos(this.angle) * this.speed
    this.y += Math.sin(this.angle) * this.speed

    if (world.isColliding(this.x, this.y, this.radius)) {
      this.x = prevX
      this.y = prevY
      this.speed = 0
    }

    // Keep in bounds
    this.x = Math.max(this.radius, Math.min(world.width - this.radius, this.x))
    this.y = Math.max(this.radius, Math.min(world.height - this.radius, this.y))

    this.trail.push({ x: this.x, y: this.y })
    if (this.trail.length > 200) this.trail.shift()

    this.sonarDistance = world.raycast(this.x, this.y, this.angle)
  }

  getState() {
    return {
      x: this.x,
      y: this.y,
      angle: this.angle,
      command: this.command,
      sonarDistance: this.sonarDistance,
    }
  }
}
