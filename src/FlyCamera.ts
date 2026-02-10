export interface FlyCameraOptions {
  speed?: number;
  fastMultiplier?: number;
  sensitivity?: number;
  position?: Float64Array;
  yaw?: number;
  pitch?: number;
}

const DEG89 = (89 * Math.PI) / 180;

export class FlyCamera {
  readonly position: Float64Array;
  private yaw: number;
  private pitch: number;
  private speed: number;
  private fastMultiplier: number;
  private sensitivity: number;

  // Input state
  private keys = new Set<string>();
  private canvas: HTMLCanvasElement | null = null;
  private boundKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private boundKeyUp: ((e: KeyboardEvent) => void) | null = null;
  private boundMouseMove: ((e: MouseEvent) => void) | null = null;
  private boundClick: (() => void) | null = null;

  // Reusable output
  private viewMat = new Float32Array(16);

  constructor(opts?: FlyCameraOptions) {
    this.position = opts?.position
      ? new Float64Array(opts.position)
      : new Float64Array([0, 0, 0]);
    this.yaw = opts?.yaw ?? 0;
    this.pitch = opts?.pitch ?? 0;
    this.speed = opts?.speed ?? 20;
    this.fastMultiplier = opts?.fastMultiplier ?? 5;
    this.sensitivity = opts?.sensitivity ?? 0.002;
  }

  attach(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;

    this.boundKeyDown = (e: KeyboardEvent) => {
      this.keys.add(e.code);
    };
    this.boundKeyUp = (e: KeyboardEvent) => {
      this.keys.delete(e.code);
    };
    this.boundMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== canvas) return;
      this.yaw += e.movementX * this.sensitivity;
      this.pitch += e.movementY * this.sensitivity;
      this.pitch = Math.max(-DEG89, Math.min(DEG89, this.pitch));
    };
    this.boundClick = () => {
      if (document.pointerLockElement !== canvas) {
        canvas.requestPointerLock();
      }
    };

    window.addEventListener('keydown', this.boundKeyDown);
    window.addEventListener('keyup', this.boundKeyUp);
    document.addEventListener('mousemove', this.boundMouseMove);
    canvas.addEventListener('click', this.boundClick);
  }

  detach(): void {
    if (this.boundKeyDown) window.removeEventListener('keydown', this.boundKeyDown);
    if (this.boundKeyUp) window.removeEventListener('keyup', this.boundKeyUp);
    if (this.boundMouseMove) document.removeEventListener('mousemove', this.boundMouseMove);
    if (this.boundClick && this.canvas) this.canvas.removeEventListener('click', this.boundClick);
    this.boundKeyDown = null;
    this.boundKeyUp = null;
    this.boundMouseMove = null;
    this.boundClick = null;
    this.canvas = null;
  }

  update(dt: number): void {
    const fast = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const spd = this.speed * (fast ? this.fastMultiplier : 1) * dt;

    // Forward direction (pitch-aware) — must match view matrix look direction
    const fx = Math.sin(this.yaw) * Math.cos(this.pitch);
    const fy = -Math.sin(this.pitch);
    const fz = -Math.cos(this.yaw) * Math.cos(this.pitch);

    // Strafe direction (always horizontal)
    const sx = Math.cos(this.yaw);
    const sz = Math.sin(this.yaw);

    // W/S — forward/back
    if (this.keys.has('KeyW')) {
      this.position[0] += fx * spd;
      this.position[1] += fy * spd;
      this.position[2] += fz * spd;
    }
    if (this.keys.has('KeyS')) {
      this.position[0] -= fx * spd;
      this.position[1] -= fy * spd;
      this.position[2] -= fz * spd;
    }

    // A/D — strafe
    if (this.keys.has('KeyA')) {
      this.position[0] -= sx * spd;
      this.position[2] -= sz * spd;
    }
    if (this.keys.has('KeyD')) {
      this.position[0] += sx * spd;
      this.position[2] += sz * spd;
    }

    // Space/Ctrl — vertical
    if (this.keys.has('Space')) {
      this.position[1] += spd;
    }
    if (this.keys.has('ControlLeft') || this.keys.has('ControlRight')) {
      this.position[1] -= spd;
    }
  }

  getViewMatrix(): Float32Array {
    const out = this.viewMat;

    // Build rotation-only view matrix from yaw/pitch
    // Camera at origin for RTE — no translation
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);

    // Right vector
    out[0] = cy;
    out[1] = sp * sy;
    out[2] = -cp * sy;
    out[3] = 0;

    // Up vector
    out[4] = 0;
    out[5] = cp;
    out[6] = sp;
    out[7] = 0;

    // Forward vector (looking along -Z in yaw=0)
    out[8] = sy;
    out[9] = -sp * cy;
    out[10] = cp * cy;
    out[11] = 0;

    // No translation (RTE: camera at origin)
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;

    return out;
  }

  getPosition(): Float64Array {
    return this.position;
  }

  getSpeed(): number {
    return this.speed;
  }

  setSpeed(s: number): void {
    this.speed = Math.max(1, s);
  }
}
