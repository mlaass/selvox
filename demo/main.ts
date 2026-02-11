import { VoxelRenderer, PerformanceOverlay, OverlayMode, PerlinNoise, FlyCamera } from '../src/index.js';
import { mat4Create, mat4Perspective } from '../src/gpu/math.js';

const VOXEL_STRIDE = 64; // bytes per voxel
const VOXEL_SIZE = 0.2;
const GRID_SIZE = 3162; // 3162² ≈ 10M surface voxels
const MAX_VOXELS = 16_000_000;

function packColor(r: number, g: number, b: number): number {
  return r | (g << 8) | (b << 16) | (255 << 24);
}

function writeVoxel(
  view: DataView,
  offset: number,
  x: number,
  y: number,
  z: number,
  color: number,
  size: number,
): void {
  // pos_a (vec4 = 16 bytes)
  view.setFloat32(offset, x, true);
  view.setFloat32(offset + 4, y, true);
  view.setFloat32(offset + 8, z, true);
  view.setFloat32(offset + 12, 0, true);
  // color_a (u32)
  view.setUint32(offset + 16, color, true);
  // size_a (f32)
  view.setFloat32(offset + 20, size, true);
  // _pad0
  view.setFloat32(offset + 24, 0, true);
  view.setFloat32(offset + 28, 0, true);
  // pos_b = pos_a
  view.setFloat32(offset + 32, x, true);
  view.setFloat32(offset + 36, y, true);
  view.setFloat32(offset + 40, z, true);
  view.setFloat32(offset + 44, 0, true);
  // color_b = color_a
  view.setUint32(offset + 48, color, true);
  // size_b = size_a
  view.setFloat32(offset + 52, size, true);
  // _pad1
  view.setFloat32(offset + 56, 0, true);
  view.setFloat32(offset + 60, 0, true);
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Sample the heightMap at a world-space (wx, wz) position
function sampleHeight(heightMap: Float32Array, wx: number, wz: number): number {
  const halfGrid = GRID_SIZE / 2;
  const gx = Math.round(wx / VOXEL_SIZE) + halfGrid;
  const gz = Math.round(wz / VOXEL_SIZE) + halfGrid;
  const cx = Math.max(0, Math.min(GRID_SIZE - 1, gx));
  const cz = Math.max(0, Math.min(GRID_SIZE - 1, gz));
  return heightMap[cx * GRID_SIZE + cz];
}

const HOUSE_COLORS = [
  packColor(210, 190, 160), // beige
  packColor(160, 70, 50),   // brick red
  packColor(230, 220, 200), // cream
  packColor(120, 120, 130), // slate grey
  packColor(195, 175, 140), // tan
  packColor(60, 60, 65),    // charcoal
];

interface ChunkData {
  data: ArrayBuffer;
  count: number;
}

function generateWorld(): { landscape: ChunkData; houses: ChunkData } {
  // --- Phase 1: Generate landscape heightmap ---
  const noise = new PerlinNoise(42);
  const fbmOpts = { frequency: 0.006, octaves: 6, persistence: 0.4 };
  const heightScale = 12; // meters
  const halfGrid = GRID_SIZE / 2;

  const landscapeCount = GRID_SIZE * GRID_SIZE;
  const heightMap = new Float32Array(landscapeCount);

  // Pre-compute heights
  for (let x = 0; x < GRID_SIZE; x++) {
    const wx = x - halfGrid;
    for (let z = 0; z < GRID_SIZE; z++) {
      const wz = z - halfGrid;
      const h = noise.fbm2D(wx * VOXEL_SIZE, wz * VOXEL_SIZE, fbmOpts) * heightScale;
      heightMap[x * GRID_SIZE + z] = Math.floor(h / VOXEL_SIZE) * VOXEL_SIZE;
    }
  }

  // --- Phase 2: Pre-compute house definitions ---
  const rng = mulberry32(123);
  const densityNoise = new PerlinNoise(99);
  const worldExtent = GRID_SIZE * VOXEL_SIZE;
  const halfExtent = worldExtent / 2;
  const cellSize = 20;
  const margin = 40;
  const cellsPerAxis = Math.floor((worldExtent - 2 * margin) / cellSize);

  interface Building {
    wx: number; wz: number;
    widthM: number; depthM: number; heightM: number;
    color: number;
  }
  const buildings: Building[] = [];
  const maxHouseVoxels = MAX_VOXELS - landscapeCount;
  let houseVoxelCount = 0;

  for (let cx = 0; cx < cellsPerAxis; cx++) {
    for (let cz = 0; cz < cellsPerAxis; cz++) {
      const cellX = -halfExtent + margin + (cx + 0.5) * cellSize;
      const cellZ = -halfExtent + margin + (cz + 0.5) * cellSize;

      const density = densityNoise.fbm2D(cellX, cellZ, { frequency: 0.008, octaves: 2 });
      const prob = 0.6 * (0.5 + 0.5 * density);
      if (rng() > prob) continue;

      const jitterX = (rng() - 0.5) * cellSize * 0.6;
      const jitterZ = (rng() - 0.5) * cellSize * 0.6;
      const wx = cellX + jitterX;
      const wz = cellZ + jitterZ;

      const roll = rng();
      let widthM: number, depthM: number, heightM: number;
      if (roll < 0.40) {
        widthM = 3 + rng() * 3;
        depthM = 3 + rng() * 3;
        heightM = 2.4 + rng() * 1.6;
      } else if (roll < 0.75) {
        widthM = 5 + rng() * 5;
        depthM = 5 + rng() * 5;
        heightM = 3 + rng() * 4;
      } else {
        widthM = 7 + rng() * 5;
        depthM = 7 + rng() * 5;
        heightM = 5 + rng() * 7;
      }

      const voxelsW = Math.ceil(widthM / VOXEL_SIZE);
      const voxelsD = Math.ceil(depthM / VOXEL_SIZE);
      const voxelsH = Math.ceil(heightM / VOXEL_SIZE);
      const buildingVoxels = voxelsW * voxelsD * voxelsH;

      if (houseVoxelCount + buildingVoxels > maxHouseVoxels) continue;
      houseVoxelCount += buildingVoxels;

      const color = HOUSE_COLORS[Math.floor(rng() * HOUSE_COLORS.length)];
      buildings.push({ wx, wz, widthM, depthM, heightM, color });
    }
  }

  console.log(`selvox: landscape ${landscapeCount.toLocaleString()} voxels, ${buildings.length} buildings ~${houseVoxelCount.toLocaleString()} voxels`);

  // --- Phase 3: Write landscape buffer ---
  const landscapeData = new ArrayBuffer(landscapeCount * VOXEL_STRIDE);
  const landscapeView = new DataView(landscapeData);
  let offset = 0;

  const grassColor = packColor(76, 153, 0);
  for (let x = 0; x < GRID_SIZE; x++) {
    const wx = x - halfGrid;
    for (let z = 0; z < GRID_SIZE; z++) {
      const wz = z - halfGrid;
      const wy = heightMap[x * GRID_SIZE + z];
      writeVoxel(landscapeView, offset, wx * VOXEL_SIZE, wy, wz * VOXEL_SIZE, grassColor, VOXEL_SIZE);
      offset += VOXEL_STRIDE;
    }
  }
  console.log(`selvox: wrote landscape chunk, ${landscapeCount.toLocaleString()} voxels`);

  // --- Phase 4: Write houses buffer ---
  const housesData = new ArrayBuffer(houseVoxelCount * VOXEL_STRIDE);
  const housesView = new DataView(housesData);
  offset = 0;
  let houseActual = 0;

  for (const b of buildings) {
    const voxelsW = Math.ceil(b.widthM / VOXEL_SIZE);
    const voxelsD = Math.ceil(b.depthM / VOXEL_SIZE);
    const voxelsH = Math.ceil(b.heightM / VOXEL_SIZE);

    const hw = b.widthM / 2;
    const hd = b.depthM / 2;
    const h0 = sampleHeight(heightMap, b.wx - hw, b.wz - hd);
    const h1 = sampleHeight(heightMap, b.wx + hw, b.wz - hd);
    const h2 = sampleHeight(heightMap, b.wx - hw, b.wz + hd);
    const h3 = sampleHeight(heightMap, b.wx + hw, b.wz + hd);
    const baseY = Math.min(h0, h1, h2, h3) + VOXEL_SIZE;

    const startX = b.wx - hw;
    const startZ = b.wz - hd;

    for (let ix = 0; ix < voxelsW; ix++) {
      const px = startX + ix * VOXEL_SIZE;
      for (let iz = 0; iz < voxelsD; iz++) {
        const pz = startZ + iz * VOXEL_SIZE;
        for (let iy = 0; iy < voxelsH; iy++) {
          const py = baseY + iy * VOXEL_SIZE;
          writeVoxel(housesView, offset, px, py, pz, b.color, VOXEL_SIZE);
          offset += VOXEL_STRIDE;
          houseActual++;
        }
      }
    }
  }
  console.log(`selvox: wrote houses chunk, ${houseActual.toLocaleString()} voxels`);

  return {
    landscape: { data: landscapeData, count: landscapeCount },
    houses: { data: housesData, count: houseActual },
  };
}

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) {
    console.error('Canvas element #gpu-canvas not found.');
    return;
  }

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const renderer = new VoxelRenderer(canvas, { maxVoxels: MAX_VOXELS });

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

  // Generate and upload world as two separate chunks
  const { landscape, houses } = generateWorld();
  renderer.loadChunk('landscape', landscape.data, landscape.count, new Float64Array([0, 0, 0]), 0);
  renderer.loadChunk('houses', houses.data, houses.count, new Float64Array([0, 0, 0]), 0);

  // Flying camera — above buildings, pulled back, looking down
  const camera = new FlyCamera({
    position: new Float64Array([0, 40, 80]),
    yaw: 0,
    pitch: -0.35,
    speed: 15,
  });
  camera.attach(canvas);

  // Performance overlay
  const overlay = new PerformanceOverlay(canvas);
  let lastTimeMs = -1;

  // Controls panel (toggled with O)
  const debugToggles: { bit: number; label: string; checkbox: HTMLInputElement }[] = [];
  const panel = document.createElement('div');
  panel.style.cssText = 'position:fixed;top:160px;right:8px;color:#ccc;font:12px monospace;background:rgba(0,0,0,0.75);padding:8px;border-radius:4px;pointer-events:auto;z-index:9999;width:180px;display:none';

  const title = document.createElement('div');
  title.textContent = 'Controls';
  title.style.cssText = 'font-weight:bold;margin-bottom:6px';
  panel.appendChild(title);

  const toggleDefs: { bit: number; label: string }[] = [
    { bit: 0, label: 'LOD colors' },
    { bit: 1, label: 'Billboard' },
    { bit: 2, label: 'Wireframe' },
    { bit: 3, label: 'Normals' },
    { bit: 4, label: 'Depth' },
  ];

  function toggleDebugBit(bit: number) {
    renderer.setDebugFlags(renderer.currentDebugFlags ^ (1 << bit));
    const entry = debugToggles.find(t => t.bit === bit);
    if (entry) entry.checkbox.checked = (renderer.currentDebugFlags & (1 << bit)) !== 0;
  }

  for (const def of toggleDefs) {
    const label = document.createElement('label');
    label.style.cssText = 'display:flex;align-items:center;gap:4px;margin-bottom:3px;cursor:pointer';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.style.cssText = 'margin:0;cursor:pointer';
    const entry = { ...def, checkbox: cb };
    debugToggles.push(entry);
    cb.addEventListener('change', () => toggleDebugBit(def.bit));
    label.appendChild(cb);
    label.appendChild(document.createTextNode(def.label));
    panel.appendChild(label);
  }

  // Separator
  const sep1 = document.createElement('hr');
  sep1.style.cssText = 'border:none;border-top:1px solid #555;margin:6px 0';
  panel.appendChild(sep1);

  // Speed slider
  const speedRow = document.createElement('div');
  speedRow.style.cssText = 'display:flex;align-items:center;gap:4px;margin-bottom:2px';
  const speedLabel = document.createElement('span');
  speedLabel.textContent = 'Speed';
  const speedVal = document.createElement('span');
  speedVal.style.cssText = 'margin-left:auto;min-width:28px;text-align:right';
  speedVal.textContent = String(Math.round(camera.getSpeed()));
  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '1';
  speedSlider.max = '200';
  speedSlider.value = String(Math.round(camera.getSpeed()));
  speedSlider.style.cssText = 'flex:1;cursor:pointer';
  speedSlider.addEventListener('input', () => {
    camera.setSpeed(Number(speedSlider.value));
    speedVal.textContent = speedSlider.value;
  });
  speedRow.appendChild(speedLabel);
  speedRow.appendChild(speedSlider);
  speedRow.appendChild(speedVal);
  panel.appendChild(speedRow);

  // Separator
  const sep2 = document.createElement('hr');
  sep2.style.cssText = 'border:none;border-top:1px solid #555;margin:6px 0';
  panel.appendChild(sep2);

  // Stats section
  const statsDiv = document.createElement('div');
  statsDiv.style.cssText = 'white-space:pre';
  panel.appendChild(statsDiv);

  document.body.appendChild(panel);

  // Projection matrix
  const proj = mat4Create();

  // Diagnostic logging state
  let lastDiagnosticTime = 0;

  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'p') overlay.cycle();
    if (k === 'o') panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  });

  window.addEventListener('resize', () => {
    renderer.resize(window.innerWidth, window.innerHeight);
    overlay.resize();
  });

  let statsUpdateCounter = 0;

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

    // Stats update (every 30 frames)
    statsUpdateCounter++;
    if (statsUpdateCounter % 30 === 0) {
      const pos = camPos;
      const stats = renderer.getPoolStats();
      let text =
        `Voxels: ${renderer.voxelCount.toLocaleString()}\n` +
        `Pos: [${pos[0].toFixed(0)}, ${pos[1].toFixed(0)}, ${pos[2].toFixed(0)}]\n` +
        `Draw calls: ${renderer.drawCallCount}`;
      if (stats) {
        text += `\nChunks: ${stats.chunkCount}`;
        text += `\nBuffer: ${(stats.bufferUtilization * 100).toFixed(0)}%`;
      }
      statsDiv.textContent = text;
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
