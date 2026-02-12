import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://snakamoto404.github.io',
  output: 'static',
  vite: {
    plugins: [tailwindcss()]
  }
});
