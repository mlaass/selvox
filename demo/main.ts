import { VoxelRenderer } from '../src/index.js';
import { mat4Create, mat4Perspective, mat4LookAt } from '../src/gpu/math.js';

// Pack RGBA color as u32 (little-endian: ABGR byte order)
function packColor(r: number, g: number, b: number): number {
  return (r & 0xff) | ((g & 0xff) << 8) | ((b & 0xff) << 16) | (0xff << 24);
}

// 64 bytes per voxel — matches GPU struct layout
function buildVoxelData(): { buffer: ArrayBuffer; count: number } {
  const voxels = [
    // Voxel 0: red→blue, moves left→right, shrinks
    {
      posA: [0, 0, 0], colorA: packColor(255, 60, 60), sizeA: 1.0,
      posB: [2, 0, 0], colorB: packColor(60, 60, 255), sizeB: 0.5,
    },
    // Voxel 1: green→yellow, moves up, grows
    {
      posA: [-2, -1, 1], colorA: packColor(60, 220, 60), sizeA: 0.6,
      posB: [-2, 1, 1],  colorB: packColor(255, 220, 40), sizeB: 1.2,
    },
    // Voxel 2: cyan→magenta, moves forward, pulses
    {
      posA: [1, 1, -1],  colorA: packColor(40, 220, 220), sizeA: 0.8,
      posB: [1, -1, -1], colorB: packColor(220, 40, 220), sizeB: 1.0,
    },
  ];

  const VOXEL_STRIDE = 64; // bytes
  const count = voxels.length;
  const buffer = new ArrayBuffer(VOXEL_STRIDE * count);

  for (let i = 0; i < count; i++) {
    const v = voxels[i];
    const offset = i * VOXEL_STRIDE;
    const f32 = new Float32Array(buffer, offset, 16);
    const u32 = new Uint32Array(buffer, offset, 16);

    // State A: pos_a (vec4), color_a (u32), size_a (f32), pad (vec2)
    f32[0] = v.posA[0]; f32[1] = v.posA[1]; f32[2] = v.posA[2]; f32[3] = 0;
    u32[4] = v.colorA;
    f32[5] = v.sizeA;
    f32[6] = 0; f32[7] = 0;

    // State B: pos_b (vec4), color_b (u32), size_b (f32), pad (vec2)
    f32[8] = v.posB[0]; f32[9] = v.posB[1]; f32[10] = v.posB[2]; f32[11] = 0;
    u32[12] = v.colorB;
    f32[13] = v.sizeB;
    f32[14] = 0; f32[15] = 0;
  }

  return { buffer, count };
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

  // Upload demo voxel data
  const { buffer, count } = buildVoxelData();
  renderer.uploadVoxels(buffer, count);

  // Camera setup
  const view = mat4Create();
  const proj = mat4Create();
  const eye = new Float32Array(3);
  const center = new Float32Array([0, 0, 0]);
  const up = new Float32Array([0, 1, 0]);

  window.addEventListener('resize', () => {
    renderer.resize(window.innerWidth, window.innerHeight);
  });

  function frame(timeMs: number) {
    const t = timeMs * 0.001; // seconds

    // Ping-pong interpolation factor 0↔1
    const interp = Math.sin(t * 0.8) * 0.5 + 0.5;
    renderer.setTime(interp);

    // Orbiting camera
    const radius = 8;
    const angle = t * 0.4;
    eye[0] = Math.cos(angle) * radius;
    eye[1] = 3;
    eye[2] = Math.sin(angle) * radius;

    const aspect = canvas.width / canvas.height;
    mat4Perspective(proj, Math.PI / 4, aspect, 0.1, 1000);
    mat4LookAt(view, eye, center, up);

    renderer.updateCamera(view, proj);
    renderer.render();

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
