/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  // viteSingleFile inlines all JS and CSS, so `npm run build` produces one
  // dist/index.html that opens straight from disk and can be shared as a file.
  plugins: [react(), tailwindcss(), viteSingleFile()],
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
  },
});
