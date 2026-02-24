import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeSlug from 'rehype-slug';
import remarkDirective from 'remark-directive';
import { remarkTheorem } from './src/plugins/remark-theorem.mjs';
import rehypeMathjaxTwoPass from './src/plugins/rehype-mathjax-two-pass.mjs';

export default defineConfig({
  site: 'https://snakamoto404.github.io',
  output: 'static',
  integrations: [
    sitemap({
      filter: (page) => {
        const pathname = new URL(page).pathname;
        return !['/subscribe/', '/subscribed/', '/unsubscribe/', '/unsubscribed/'].includes(pathname);
      }
    })
  ],
  markdown: {
    remarkPlugins: [remarkMath, remarkDirective, remarkTheorem],
    rehypePlugins: [
      rehypeSlug,
      [rehypeMathjaxTwoPass, {
        tex: {
          tags: "ams",
          macros: {
            df: ["\\dfrac{#1}{#2}", 2],
            pd: ["\\partial_{#1}", 1],
            la: "\\langle",
            ra: "\\rangle",
            mbf: ["\\mathbf{#1}", 1],
            mbb: ["\\mathbb{#1}", 1],
            mrm: ["\\mathrm{#1}", 1],
            R: "\\mathbb{R}",
            E: "\\mathbb{E}",
          },
          processRefs: true,
          tagSide: "right",
        },
      }],
    ],
  },
  vite: {
    plugins: [tailwindcss()]
  }
});
