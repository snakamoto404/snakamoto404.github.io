import { defineCollection, z } from 'astro:content';

const archive = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
    summary: z.string().optional()
  })
});

export const collections = {
  archive
};
