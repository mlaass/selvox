import { readFileSync } from 'node:fs';
import { defineConfig, type Plugin } from 'vite';
import JavaScriptObfuscator from 'javascript-obfuscator';

function wgslMinify(): Plugin {
  return {
    name: 'wgsl-minify',
    enforce: 'pre',
    load(id) {
      if (!id.endsWith('.wgsl?raw')) return;
      const filePath = id.replace(/\?raw$/, '');
      let src = readFileSync(filePath, 'utf-8');
      // Strip // line comments
      src = src.replace(/\/\/[^\n]*/g, '');
      // Strip /* */ block comments
      src = src.replace(/\/\*[\s\S]*?\*\//g, '');
      // Collapse whitespace to single spaces
      src = src.replace(/\s+/g, ' ');
      // Remove spaces around WGSL punctuation
      src = src.replace(/\s*([{}();,:<>=@\[\]+\-*\/&|!^%])\s*/g, '$1');
      src = src.trim();
      return `export default ${JSON.stringify(src)}`;
    },
  };
}

function jsObfuscate(): Plugin {
  return {
    name: 'js-obfuscate',
    apply: 'build',
    enforce: 'post',
    renderChunk(code) {
      const result = JavaScriptObfuscator.obfuscate(code, {
        identifierNamesGenerator: 'hexadecimal',
        stringArray: true,
        stringArrayEncoding: ['base64'],
        stringArrayShuffle: true,
        splitStrings: true,
        splitStringsChunkLength: 10,
        // Disabled for 60fps render loop performance
        controlFlowFlattening: false,
        deadCodeInjection: false,
        selfDefending: false,
        debugProtection: false,
      });
      return { code: result.getObfuscatedCode(), map: null };
    },
  };
}

export default defineConfig({
  build: {
    outDir: 'dist',
  },
  plugins: [
    wgslMinify(),
    jsObfuscate(),
  ],
});
