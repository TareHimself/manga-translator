import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    viteStaticCopy({
      targets: [
        {
          src: 'public/manifest.json',
          dest: '.',
        },
         {
          src: 'public/extension_icon128.png',
          dest: '.',
        }
      ],
    }),
  ],
  build: {
    minify: false,
    sourcemap: true,
    emptyOutDir: false,
    rollupOptions: {
      input: {
        popup: "./popup.html",
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: `assets/[name].js`,
        assetFileNames: "[name][extname]",
        inlineDynamicImports: true
      }
    }
  }
})
