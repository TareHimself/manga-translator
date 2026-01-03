import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  build: {
    minify: false,
    sourcemap: true,
    emptyOutDir: false,
    rollupOptions: {
      input: {
        worker: "src/worker/worker.ts"
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
