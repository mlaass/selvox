import { VoxelRenderer } from '../src/index.js';

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) {
    console.error('Canvas element #gpu-canvas not found.');
    return;
  }

  // Size canvas to viewport
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

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    renderer.resize(canvas.width, canvas.height);
  });
}

main();
