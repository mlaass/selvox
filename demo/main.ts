import { VoxelRenderer, PerformanceOverlay } from '../src/index.js';
import { mat4Create, mat4Perspective, mat4LookAt } from '../src/gpu/math.js';

const VOXEL_STRIDE = 64; // bytes per voxel

// Pack RGBA color as u32 (little-endian: ABGR byte order)
function packColor(r: number, g: number, b: number): number {
  return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | (0xff << 24);
}

function writeVoxel(
  buf: ArrayBuffer,
  index: number,
  posA: [number, number, number], colorA: number, sizeA: number,
  posB: [number, number, number], colorB: number, sizeB: number,
): void {
  const offset = index * VOXEL_STRIDE;
  const f32 = new Float32Array(buf, offset, 16);
  const u32 = new Uint32Array(buf, offset, 16);

  // State A
  f32[0] = posA[0]; f32[1] = posA[1]; f32[2] = posA[2]; f32[3] = 0;
  u32[4] = colorA;
  f32[5] = sizeA;
  f32[6] = 0; f32[7] = 0;

  // State B
  f32[8] = posB[0]; f32[9] = posB[1]; f32[10] = posB[2]; f32[11] = 0;
  u32[12] = colorB;
  f32[13] = sizeB;
  f32[14] = 0; f32[15] = 0;
}

// Dense chunk: 10x10x10 grid colored by height.
// offsetA/offsetB shift the gradient for State A / State B; the GPU interpolates between them.
function buildCityChunk(offsetA: number = 0, offsetB: number = 0): { buffer: ArrayBuffer; count: number } {
  const N = 10;
  const count = N * N * N;
  const buffer = new ArrayBuffer(VOXEL_STRIDE * count);

  function heightColor(y: number, offset: number): number {
    const t = ((y / (N - 1)) + offset) % 1.0;
    const r = Math.round(30 + (1 - t) * 100);
    const g = Math.round(180 - t * 80);
    const b = Math.round(60 + t * 195);
    return packColor(r, g, b);
  }

  let idx = 0;
  for (let y = 0; y < N; y++) {
    const colorA = heightColor(y, offsetA);
    const colorB = heightColor(y, offsetB);

    for (let x = 0; x < N; x++) {
      for (let z = 0; z < N; z++) {
        const px = x - N / 2 + 0.5;
        const py = y - N / 2 + 0.5;
        const pz = z - N / 2 + 0.5;
        writeVoxel(
          buffer, idx,
          [px, py, pz], colorA, 0.9,
          [px, py, pz], colorB, 0.9,
        );
        idx++;
      }
    }
  }

  return { buffer, count };
}

// Sparse chunk: single large beacon voxel
function buildBeaconChunk(): { buffer: ArrayBuffer; count: number } {
  const buffer = new ArrayBuffer(VOXEL_STRIDE);
  writeVoxel(
    buffer, 0,
    [0, 7, 0], packColor(255, 60, 30), 1.5,
    [0, 7.5, 0], packColor(255, 220, 40), 1.8,
  );
  return { buffer, count: 1 };
}

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) {
    console.error('Canvas element #gpu-canvas not found.');
    return;
  }

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const renderer = new VoxelRenderer(canvas);

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

  // Color cycling config
  const CYCLE_STEPS = 10; // one step per height level
  const CYCLE_SPEED = 0.1; // full cycle in ~10s

  function cityOffsets(step: number): [number, number] {
    const offsetA = (1 - (step % CYCLE_STEPS) / CYCLE_STEPS) % 1.0;
    const offsetB = (1 - ((step + 1) % CYCLE_STEPS) / CYCLE_STEPS) % 1.0;
    return [offsetA, offsetB];
  }

  // Load multi-chunk scene
  const [initA, initB] = cityOffsets(0);
  const initialCity = buildCityChunk(initA, initB);
  const beacon = buildBeaconChunk();
  renderer.loadChunk('city', initialCity.buffer, initialCity.count);
  renderer.loadChunk('beacon', beacon.buffer, beacon.count);

  // Performance overlay
  const overlay = new PerformanceOverlay(canvas);
  let lastTimeMs = -1;
  let cityLoaded = true;
  let prevStep = 0;

  window.addEventListener('keydown', (e) => {
    if (e.key === 'd' || e.key === 'D') overlay.cycle();
    if (e.key === 'r' || e.key === 'R') {
      if (cityLoaded) {
        renderer.unloadChunk('city');
        cityLoaded = false;
        console.log('selvox: unloaded "city" chunk');
      } else {
        prevStep = 0;
        const [oA, oB] = cityOffsets(0);
        const city = buildCityChunk(oA, oB);
        renderer.loadChunk('city', city.buffer, city.count);
        cityLoaded = true;
        console.log('selvox: reloaded "city" chunk');
      }
    }
  });

  // Camera setup
  const view = mat4Create();
  const proj = mat4Create();
  const eye = new Float32Array(3);
  const center = new Float32Array([0, 0, 0]);
  const up = new Float32Array([0, 1, 0]);

  window.addEventListener('resize', () => {
    renderer.resize(window.innerWidth, window.innerHeight);
    overlay.resize();
  });

  function frame(timeMs: number) {
    const t = timeMs * 0.001;

    // Ping-pong interpolation factor 0↔1 (drives beacon position/size)
    const interp = Math.sin(t * 0.8) * 0.5 + 0.5;
    renderer.setTime(interp);

    // Step-based color cycling — rebuild only when the step changes (~1/s)
    if (cityLoaded) {
      const phase = t * CYCLE_SPEED * CYCLE_STEPS;
      const stepIndex = Math.floor(phase);
      const colorFrac = phase - stepIndex;

      if (stepIndex !== prevStep) {
        prevStep = stepIndex;
        const [oA, oB] = cityOffsets(stepIndex);
        const city = buildCityChunk(oA, oB);
        renderer.loadChunk('city', city.buffer, city.count);
      }

      renderer.setColorTime(colorFrac);
    }

    // Orbiting camera — pulled back to see full scene
    const radius = 20;
    const angle = t * 0.4;
    eye[0] = Math.cos(angle) * radius;
    eye[1] = 8;
    eye[2] = Math.sin(angle) * radius;

    const aspect = canvas.width / canvas.height;
    mat4Perspective(proj, Math.PI / 4, aspect, 0.1, 1000);
    mat4LookAt(view, eye, center, up);

    renderer.updateCamera(view, proj);
    renderer.render();

    const frameDeltaMs = lastTimeMs < 0 ? 0 : timeMs - lastTimeMs;
    lastTimeMs = timeMs;
    overlay.update({ frameDeltaMs, voxelCount: renderer.voxelCount });

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
