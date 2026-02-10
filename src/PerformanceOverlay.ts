import { RingBuffer } from './perf/RingBuffer.js';

export enum OverlayMode {
  Off = 0,
  Compact = 1,
  Expanded = 2,
}

export interface OverlayStats {
  frameDeltaMs: number;
  voxelCount: number;
}

const COMPACT_W = 200;
const COMPACT_H = 28;
const GRAPH_W = 240;
const GRAPH_H = 112;
const EXPANDED_W = GRAPH_W;
const EXPANDED_H = COMPACT_H + GRAPH_H;
const MARGIN = 8;

const REDRAW_INTERVAL = 1000 / 20; // 20 fps throttle

export class PerformanceOverlay {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private target: HTMLCanvasElement;
  private mode: OverlayMode = OverlayMode.Off;
  private ring: RingBuffer;
  private smoothedFps = 60;
  private lastVoxelCount = 0;
  private lastDrawTime = 0;
  private firstFrame = true;

  constructor(target: HTMLCanvasElement, historySeconds = 10) {
    this.target = target;
    this.ring = new RingBuffer(Math.ceil(historySeconds * 120));

    const c = document.createElement('canvas');
    c.style.position = 'fixed';
    c.style.top = '0';
    c.style.right = '0';
    c.style.pointerEvents = 'none';
    c.style.zIndex = '9999';
    this.canvas = c;
    this.ctx = c.getContext('2d')!;

    target.parentElement!.appendChild(c);
    this.syncSize();
  }

  getMode(): OverlayMode {
    return this.mode;
  }

  setMode(mode: OverlayMode): void {
    this.mode = mode;
    if (mode === OverlayMode.Off) {
      this.canvas.style.display = 'none';
    } else {
      this.canvas.style.display = 'block';
    }
  }

  cycle(): OverlayMode {
    const next = ((this.mode + 1) % 3) as OverlayMode;
    this.setMode(next);
    return next;
  }

  update(stats: OverlayStats): void {
    if (this.firstFrame) {
      this.firstFrame = false;
      this.lastVoxelCount = stats.voxelCount;
      return;
    }

    const dt = Math.min(stats.frameDeltaMs, 200);
    this.ring.push(dt);
    this.lastVoxelCount = stats.voxelCount;

    const fps = dt > 0 ? 1000 / dt : 0;
    const alpha = 0.05;
    this.smoothedFps = this.smoothedFps * (1 - alpha) + fps * alpha;

    if (this.mode === OverlayMode.Off) return;

    const now = performance.now();
    if (now - this.lastDrawTime < REDRAW_INTERVAL) return;
    this.lastDrawTime = now;

    this.draw();
  }

  resize(): void {
    this.syncSize();
  }

  dispose(): void {
    this.canvas.remove();
  }

  // ── private ────────────────────────────────────────────

  private syncSize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private draw(): void {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, this.canvas.width / dpr, this.canvas.height / dpr);

    if (this.mode === OverlayMode.Compact) {
      this.drawCompact(window.innerWidth - COMPACT_W - MARGIN, MARGIN);
    } else if (this.mode === OverlayMode.Expanded) {
      const x = window.innerWidth - EXPANDED_W - MARGIN;
      this.drawCompact(x, MARGIN);
      this.drawGraph(x, MARGIN + COMPACT_H);
    }
  }

  private drawCompact(x: number, y: number): void {
    const ctx = this.ctx;
    const w = this.mode === OverlayMode.Expanded ? EXPANDED_W : COMPACT_W;

    // Background
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    roundRect(ctx, x, y, w, COMPACT_H, 4);
    ctx.fill();

    // FPS color
    const fps = this.smoothedFps;
    let fpsColor = '#4ade80'; // green
    if (fps < 30) fpsColor = '#f87171'; // red
    else if (fps < 55) fpsColor = '#facc15'; // yellow

    ctx.font = '12px monospace';
    ctx.textBaseline = 'middle';
    const cy = y + COMPACT_H / 2;

    // FPS
    ctx.fillStyle = fpsColor;
    ctx.fillText(`FPS: ${Math.round(fps)}`, x + 8, cy);

    // Frame time
    const latest = this.ring.length > 0 ? this.ring.get(this.ring.length - 1) : 0;
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(`| ${latest.toFixed(1)}ms`, x + 72, cy);

    // Voxel count
    ctx.fillText(`| ${this.lastVoxelCount} V`, x + 138, cy);
  }

  private drawGraph(x: number, y: number): void {
    const ctx = this.ctx;
    const w = GRAPH_W;
    const h = GRAPH_H;
    const pad = 6;
    const graphX = x + pad;
    const graphY = y + pad;
    const graphW = w - pad * 2;
    const graphH = h - pad * 2;

    // Background
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    roundRect(ctx, x, y, w, h, 4);
    ctx.fill();

    const len = this.ring.length;
    if (len < 2) return;

    // Y-axis: auto-scale with minimum cap at 33.33ms
    const { max } = this.ring.minMax();
    const yMax = Math.max(33.33, max * 1.1);

    const toX = (i: number) => graphX + (i / (len - 1)) * graphW;
    const toY = (v: number) => graphY + graphH - (v / yMax) * graphH;

    // Reference lines
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;

    // 60fps line (16.67ms)
    ctx.strokeStyle = 'rgba(74,222,128,0.4)';
    const y60 = toY(16.67);
    ctx.beginPath();
    ctx.moveTo(graphX, y60);
    ctx.lineTo(graphX + graphW, y60);
    ctx.stroke();

    // 30fps line (33.33ms)
    ctx.strokeStyle = 'rgba(248,113,113,0.4)';
    const y30 = toY(33.33);
    ctx.beginPath();
    ctx.moveTo(graphX, y30);
    ctx.lineTo(graphX + graphW, y30);
    ctx.stroke();

    ctx.setLineDash([]);

    // Build polyline path
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(this.ring.get(0)));
    for (let i = 1; i < len; i++) {
      ctx.lineTo(toX(i), toY(this.ring.get(i)));
    }

    // Stroke polyline
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Filled area under curve
    ctx.lineTo(toX(len - 1), graphY + graphH);
    ctx.lineTo(toX(0), graphY + graphH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(96,165,250,0.15)';
    ctx.fill();

    // Labels
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(74,222,128,0.7)';
    ctx.fillText('16.7ms', graphX + 2, y60 - 3);
    ctx.fillStyle = 'rgba(248,113,113,0.7)';
    ctx.fillText('33.3ms', graphX + 2, y30 - 3);
  }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.closePath();
}
