import { VoxelRenderer, PerformanceOverlay, MockOctreeSource, ChunkManager } from '../src/index.js';
import { mat4Create, mat4Perspective, mat4LookAt } from '../src/gpu/math.js';

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

  // Set up mock octree data source and chunk manager
  const source = new MockOctreeSource(3); // 3x3x3 grid of chunk positions
  const metadata = await source.getMetadata();
  renderer.setDataSource(source);

  const chunkManager = new ChunkManager(renderer, source);
  await chunkManager.initialize();

  // World center from metadata (large geospatial offset)
  const worldCenter = new Float64Array([
    (metadata.worldBounds[0] + metadata.worldBounds[3]) * 0.5,
    (metadata.worldBounds[1] + metadata.worldBounds[4]) * 0.5,
    (metadata.worldBounds[2] + metadata.worldBounds[5]) * 0.5,
  ]);

  console.log(`selvox: World center at [${worldCenter[0]}, ${worldCenter[1]}, ${worldCenter[2]}] (RTE mode)`);

  // Performance overlay
  const overlay = new PerformanceOverlay(canvas);
  let lastTimeMs = -1;

  // Info overlay for chunk/LOD stats
  const infoDiv = document.createElement('div');
  infoDiv.style.cssText = 'position:fixed;bottom:8px;left:8px;color:#ccc;font:12px monospace;background:rgba(0,0,0,0.65);padding:4px 8px;border-radius:4px;pointer-events:none;z-index:9999';
  document.body.appendChild(infoDiv);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'd' || e.key === 'D') overlay.cycle();
    if (e.key === 'l' || e.key === 'L') {
      renderer.setDebugFlags(renderer.currentDebugFlags ^ 1);
      console.log(`selvox: LOD debug ${(renderer.currentDebugFlags & 1) ? 'ON' : 'OFF'}`);
    }
    if (e.key === 'r' || e.key === 'R') {
      chunkManager.unloadAll();
      console.log('selvox: force-reloaded all chunks');
    }
  });

  // Camera setup — orbit around world center
  const view = mat4Create();
  const proj = mat4Create();
  // eye/center/up in RTE-local coordinates (relative to camera)
  const eyeLocal = new Float32Array(3);
  const centerLocal = new Float32Array([0, 0, 0]);
  const up = new Float32Array([0, 1, 0]);
  // High-precision camera position
  const cameraPositionHigh = new Float64Array(3);

  window.addEventListener('resize', () => {
    renderer.resize(window.innerWidth, window.innerHeight);
    overlay.resize();
  });

  let infoUpdateCounter = 0;

  function frame(timeMs: number) {
    const t = timeMs * 0.001;

    // Orbiting camera at large world offset
    const radius = 20;
    const angle = t * 0.4;
    const camHeight = 8;

    // Camera position in world (double precision)
    cameraPositionHigh[0] = worldCenter[0] + Math.cos(angle) * radius;
    cameraPositionHigh[1] = worldCenter[1] + camHeight;
    cameraPositionHigh[2] = worldCenter[2] + Math.sin(angle) * radius;

    // In RTE space, eye is at the orbit offset from origin
    // The RTE offset per-chunk will shift chunks relative to camera
    eyeLocal[0] = Math.cos(angle) * radius;
    eyeLocal[1] = camHeight;
    eyeLocal[2] = Math.sin(angle) * radius;

    const aspect = canvas.width / canvas.height;
    mat4Perspective(proj, Math.PI / 4, aspect, 0.1, 1000);
    mat4LookAt(view, eyeLocal, centerLocal, up);

    // Update chunks based on camera position
    chunkManager.update(cameraPositionHigh);

    // No interpolation animation for octree demo
    renderer.setTime(0);
    renderer.setColorTime(0);

    renderer.updateCamera(view, proj, cameraPositionHigh);
    renderer.render();

    const frameDeltaMs = lastTimeMs < 0 ? 0 : timeMs - lastTimeMs;
    lastTimeMs = timeMs;
    overlay.update({ frameDeltaMs, voxelCount: renderer.voxelCount });

    // Update info overlay at lower frequency
    infoUpdateCounter++;
    if (infoUpdateCounter % 30 === 0) {
      const lodLevels = Array.from(chunkManager.getActiveLodLevels()).sort();
      infoDiv.textContent = `Chunks: ${renderer.chunkCount} | LODs: [${lodLevels.join(',')}] | [D]perf [L]LOD [R]reload`;
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
