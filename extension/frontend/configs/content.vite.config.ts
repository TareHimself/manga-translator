import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'public/page_styles.css',
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
        content: "src/content/content.ts"
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
