import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
  ],
  build: {
    minify: false,
    sourcemap: true,
    emptyOutDir: false,
    rollupOptions: {
      input: {
        page: "src/page.ts"
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
