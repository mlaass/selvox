import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'Selvox',
      formats: ['es', 'cjs'],
      fileName: (format) => `selvox.${format === 'es' ? 'js' : 'cjs'}`,
    },
  },
});
