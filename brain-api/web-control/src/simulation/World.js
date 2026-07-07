export class Obstacle {
  constructor(x, y, w, h, color = '#64748b') {
    this.x = x
    this.y = y
    this.w = w
    this.h = h
    this.color = color
  }

  contains(px, py) {
    return px >= this.x && px <= this.x + this.w && py >= this.y && py <= this.y + this.h
  }

  getBounds() {
    return { x: this.x, y: this.y, w: this.w, h: this.h }
  }
}

export class World {
  constructor(width = 800, height = 600) {
    this.width = width
    this.height = height
    this.obstacles = []
    this._buildDefaultWorld()
  }

  _buildDefaultWorld() {
    // Outer walls
    this.obstacles.push(new Obstacle(0, 0, this.width, 20, '#334155'))
    this.obstacles.push(new Obstacle(0, this.height - 20, this.width, 20, '#334155'))
    this.obstacles.push(new Obstacle(0, 0, 20, this.height, '#334155'))
    this.obstacles.push(new Obstacle(this.width - 20, 0, 20, this.height, '#334155'))

    // Internal obstacles
    this.obstacles.push(new Obstacle(200, 150, 120, 120, '#ef4444'))
    this.obstacles.push(new Obstacle(500, 300, 150, 80, '#f59e0b'))
    this.obstacles.push(new Obstacle(350, 450, 80, 80, '#22c55e'))
    this.obstacles.push(new Obstacle(600, 100, 60, 200, '#8b5cf6'))
  }

  addObstacle(obs) {
    this.obstacles.push(obs)
  }

  clearObstacles() {
    this.obstacles = []
    this._buildDefaultWorld()
  }

  isColliding(x, y, radius = 15) {
    for (const obs of this.obstacles) {
      const closestX = Math.max(obs.x, Math.min(x, obs.x + obs.w))
      const closestY = Math.max(obs.y, Math.min(y, obs.y + obs.h))
      const dx = x - closestX
      const dy = y - closestY
      if (dx * dx + dy * dy < radius * radius) {
        return true
      }
    }
    return false
  }

  raycast(x, y, angle, maxDistance = 400) {
    const cos = Math.cos(angle)
    const sin = Math.sin(angle)
    let distance = maxDistance

    for (const obs of this.obstacles) {
      const t1 = (obs.x - x) / cos
      const t2 = (obs.x + obs.w - x) / cos
      const t3 = (obs.y - y) / sin
      const t4 = (obs.y + obs.h - y) / sin

      const tmin = Math.max(Math.min(t1, t2), Math.min(t3, t4))
      const tmax = Math.min(Math.max(t1, t2), Math.max(t3, t4))

      if (tmax >= 0 && tmin <= tmax && tmin < distance) {
        distance = tmin
      }
    }

    return distance
  }
}
