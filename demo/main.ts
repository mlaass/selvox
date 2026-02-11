import { VoxelRenderer, PerformanceOverlay, OverlayMode, PerlinNoise, FlyCamera } from '../src/index.js';
import { mat4Create, mat4Perspective } from '../src/gpu/math.js';

const VOXEL_STRIDE = 64; // bytes per voxel
const GRID_SIZE = 800;
const SURFACE_DEPTH = 25;
const VOXEL_SIZE = 1.0;

function generateTerrain(): { data: ArrayBuffer; count: number } {
  const noise = new PerlinNoise(42);
  const fbmOpts = { frequency: 0.01, octaves: 6 };
  const heightScale = 40;
  const halfGrid = GRID_SIZE / 2;

  // First pass: count voxels
  let count = 0;
  for (let x = 0; x < GRID_SIZE; x++) {
    for (let z = 0; z < GRID_SIZE; z++) {
      const wx = x - halfGrid;
      const wz = z - halfGrid;
      const h = Math.floor(noise.fbm2D(wx, wz, fbmOpts) * heightScale);
      for (let dy = 0; dy < SURFACE_DEPTH; dy++) {
        const wy = h - dy;
        if (wy >= -heightScale) {
          count++;
        }
      }
    }
  }

  // Second pass: fill buffer
  const data = new ArrayBuffer(count * VOXEL_STRIDE);
  const view = new DataView(data);
  let offset = 0;

  for (let x = 0; x < GRID_SIZE; x++) {
    for (let z = 0; z < GRID_SIZE; z++) {
      const wx = x - halfGrid;
      const wz = z - halfGrid;
      const h = Math.floor(noise.fbm2D(wx, wz, fbmOpts) * heightScale);
      for (let dy = 0; dy < SURFACE_DEPTH; dy++) {
        const wy = h - dy;
        if (wy < -heightScale) continue;

        // Color: green top, brown below
        const isTop = dy === 0;
        const r = isTop ? 60 : 120;
        const g = isTop ? 160 : 80;
        const b = isTop ? 40 : 40;
        const packedColor = r | (g << 8) | (b << 16) | (255 << 24);

        const px = wx * VOXEL_SIZE;
        const py = wy * VOXEL_SIZE;
        const pz = wz * VOXEL_SIZE;

        // pos_a (vec4 = 16 bytes)
        view.setFloat32(offset, px, true);
        view.setFloat32(offset + 4, py, true);
        view.setFloat32(offset + 8, pz, true);
        view.setFloat32(offset + 12, 0, true);
        // color_a (u32)
        view.setUint32(offset + 16, packedColor, true);
        // size_a (f32)
        view.setFloat32(offset + 20, VOXEL_SIZE, true);
        // _pad0
        view.setFloat32(offset + 24, 0, true);
        view.setFloat32(offset + 28, 0, true);
        // pos_b = pos_a
        view.setFloat32(offset + 32, px, true);
        view.setFloat32(offset + 36, py, true);
        view.setFloat32(offset + 40, pz, true);
        view.setFloat32(offset + 44, 0, true);
        // color_b = color_a
        view.setUint32(offset + 48, packedColor, true);
        // size_b = size_a
        view.setFloat32(offset + 52, VOXEL_SIZE, true);
        // _pad1
        view.setFloat32(offset + 56, 0, true);
        view.setFloat32(offset + 60, 0, true);

        offset += VOXEL_STRIDE;
      }
    }
  }

  console.log(`selvox: generated ${count} voxels (${GRID_SIZE}x${GRID_SIZE}, depth ${SURFACE_DEPTH})`);
  return { data, count };
}

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) {
    console.error('Canvas element #gpu-canvas not found.');
    return;
  }

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const renderer = new VoxelRenderer(canvas, { maxVoxels: 16_000_000 });

  try {
    await renderer.initialize();
    console.log('selvox: WebGPU initialized successfully.');
  } catch (err) {
    console.error('selvox: Failed to initialize WebGPU.', err);
    document.body.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui;color:#ccc;background:#111;">
        <p>WebGPU is not available. Use a supported browser (Chrome 113+, Edge 113+, or Firefox Nightly with flags).</p>
      </div>`;
    return;
  }

  // Generate and upload terrain in one shot
  const { data, count } = generateTerrain();
  renderer.loadChunk('terrain', data, count, new Float64Array([0, 0, 0]), 0);

  // Flying camera — start above terrain looking slightly down
  const camera = new FlyCamera({
    position: new Float64Array([0, 60, 0]),
    yaw: 0,
    pitch: -0.3,
    speed: 20,
  });
  camera.attach(canvas);

  // Performance overlay
  const overlay = new PerformanceOverlay(canvas);
  let lastTimeMs = -1;

  // Info overlay
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = 'position:fixed;bottom:8px;left:8px;color:#ccc;font:12px monospace;background:rgba(0,0,0,0.65);padding:4px 8px;border-radius:4px;pointer-events:none;z-index:9999;white-space:pre';
  document.body.appendChild(infoDiv);

  // Projection matrix
  const proj = mat4Create();

  // Diagnostic logging state
  let lastDiagnosticTime = 0;

  window.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') overlay.cycle();
    if (e.key === 'l' || e.key === 'L') {
      renderer.setDebugFlags(renderer.currentDebugFlags ^ 1);
      console.log(`selvox: LOD debug ${(renderer.currentDebugFlags & 1) ? 'ON' : 'OFF'}`);
    }
    if (e.key === 'b' || e.key === 'B') {
      renderer.setDebugFlags(renderer.currentDebugFlags ^ 2);
      console.log(`selvox: Billboard debug ${(renderer.currentDebugFlags & 2) ? 'ON' : 'OFF'}`);
    }
    if (e.key === 'g' || e.key === 'G') {
      renderer.setDebugFlags(renderer.currentDebugFlags ^ 4);
      console.log(`selvox: AABB wireframe debug ${(renderer.currentDebugFlags & 4) ? 'ON' : 'OFF'}`);
    }
    if (e.key === '=' || e.key === '+') {
      camera.setSpeed(camera.getSpeed() * 1.5);
      console.log(`selvox: speed = ${camera.getSpeed().toFixed(1)}`);
    }
    if (e.key === '-' || e.key === '_') {
      camera.setSpeed(camera.getSpeed() / 1.5);
      console.log(`selvox: speed = ${camera.getSpeed().toFixed(1)}`);
    }
  });

  window.addEventListener('resize', () => {
    renderer.resize(window.innerWidth, window.innerHeight);
    overlay.resize();
  });

  let infoUpdateCounter = 0;

  function frame(timeMs: number) {
    const dt = lastTimeMs < 0 ? 0 : (timeMs - lastTimeMs) * 0.001;
    lastTimeMs = timeMs;

    // Update camera
    camera.update(dt);

    const camPos = camera.getPosition();
    const viewMat = camera.getViewMatrix();

    // Projection
    const aspect = canvas.width / canvas.height;
    mat4Perspective(proj, Math.PI / 4, aspect, 0.1, 10000);

    // Render
    renderer.setTime(0);
    renderer.setColorTime(0);
    renderer.updateCamera(viewMat, proj, camPos);
    renderer.render();

    // Perf overlay
    const frameDeltaMs = dt * 1000;
    overlay.update({ frameDeltaMs, voxelCount: renderer.voxelCount });

    // Info overlay (every 30 frames)
    infoUpdateCounter++;
    if (infoUpdateCounter % 30 === 0) {
      const pos = camPos;
      const mem = renderer.getGpuMemoryStats();
      const mb = (b: number) => (b / (1024 * 1024)).toFixed(1);
      const memLine = mem
        ? `GPU: ${mb(mem.total)} MB (voxels ${mb(mem.voxelBuffer)}, indices ${mb(mem.visibleIndices)}, instances ${mb(mem.instanceData)}, other ${mb(mem.indirectArgs + mem.uniforms)})`
        : '';
      infoDiv.textContent =
        `Voxels: ${renderer.voxelCount.toLocaleString()} | Draws: ${renderer.drawCallCount}\n` +
        `${memLine}\n` +
        `Pos: [${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)}]\n` +
        `Speed: ${camera.getSpeed().toFixed(1)}\n` +
        `[WASD]fly [Mouse]look [Space/Ctrl]up/down [Shift]fast\n` +
        `[P]perf [L]LOD [B]billboard [G]wireframe [+/-]speed`;
    }

    // Diagnostic logging — every 2s when overlay is Expanded
    if (overlay.getMode() === OverlayMode.Expanded && timeMs - lastDiagnosticTime > 2000) {
      lastDiagnosticTime = timeMs;
      const fps = dt > 0 ? (1000 / (dt * 1000)).toFixed(0) : '?';
      const frameTimeStr = (dt * 1000).toFixed(1);
      const stats = renderer.getPoolStats();
      if (stats) {
        let msg = `[selvox diagnostic]\n`;
        msg += `  FPS: ${fps} | Frame: ${frameTimeStr}ms\n`;
        msg += `  Draw calls: ${renderer.drawCallCount}\n`;
        msg += `  Total voxels: ${stats.totalVoxels}\n`;
        msg += `  Buffer: ${stats.bufferUsedSlots}/${stats.bufferCapacity} (${(stats.bufferUtilization * 100).toFixed(1)}%)\n`;
        msg += `  Chunks: ${stats.chunkCount}\n`;
        if (stats.perLod.size > 0) {
          msg += `  Per-LOD:\n`;
          for (const [lod, info] of [...stats.perLod.entries()].sort((a, b) => a[0] - b[0])) {
            msg += `    LOD ${lod}: ${info.chunkCount} chunks, ${info.voxelCount} voxels\n`;
          }
        }
        console.log(msg);
      }
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
