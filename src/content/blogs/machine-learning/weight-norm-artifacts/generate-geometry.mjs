// Regenerate the Markdown post's standalone, theme-specific SVGs with:
// node src/content/blogs/machine-learning/weight-norm-artifacts/generate-geometry.mjs
import { mkdir, writeFile } from 'node:fs/promises';
import { JSDOM } from 'jsdom';
import { jsdomAdaptor } from 'mathjax-full/js/adaptors/jsdomAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { mathjax } from 'mathjax-full/js/mathjax.js';

const adaptor = jsdomAdaptor(JSDOM);
RegisterHTMLHandler(adaptor);
const math = mathjax.document('', {
  InputJax: new TeX(),
  OutputJax: new SVG({ fontCache: 'none' }),
});

const origin = { x: 48, y: 239 };
const radius = 299;
// Shorten the overall layout to 75%, then flatten the arc by another 5%.
// Position math labels separately so their glyphs retain normal proportions.
const verticalScale = 0.75 * 0.95;
const angle = 35 * Math.PI / 180;
const point = (r, theta) => ({ x: origin.x + r * Math.cos(theta), y: origin.y - r * Math.sin(theta) * verticalScale });
const weight = point(radius, angle);
const gradient = { x: weight.x - 92 * Math.sin(angle), y: weight.y - 92 * Math.cos(angle) * verticalScale };
const decay = point(radius * 0.79, angle);
const line = (from, to) => `M${from.x} ${from.y}L${to.x} ${to.y}`;
const angles = [15, 35, 55, 75];
const contours = angles.map((degrees) => line(point(22, degrees * Math.PI / 180), point(radius * 1.1, degrees * Math.PI / 180)));
const decayArrows = angles.flatMap((degrees) =>
  [0.32, 0.54, 0.76, 1.055]
    // Leave room for the decay vector anchored at W.
    .filter((scale) => degrees !== 35 || scale < 1)
    .map((scale) => line(point(radius * scale, degrees * Math.PI / 180), point(radius * scale * 0.85, degrees * Math.PI / 180)))
);
const corner = [
  { x: weight.x - 10 * Math.cos(angle), y: weight.y + 10 * Math.sin(angle) * verticalScale },
  { x: weight.x - 10 * (Math.cos(angle) + Math.sin(angle)), y: weight.y + 10 * (Math.sin(angle) - Math.cos(angle)) * verticalScale },
  { x: weight.x - 10 * Math.sin(angle), y: weight.y - 10 * Math.cos(angle) * verticalScale },
];

const labels = [
  { tex: String.raw`W_{\mathrm{axis}}`, x: 374, y: origin.y - 27, width: 100 },
  { tex: String.raw`\|W_\perp\|`, x: 38, y: -4, width: 86 },
  { tex: String.raw`\|W\|=R`, x: origin.x + radius - 100, y: origin.y - 28, width: 100, size: 17, opacity: 0.65 },
  { tex: 'W', x: weight.x + 9, y: weight.y - 16, width: 30 },
  { tex: String.raw`G=\nabla_W L`, x: gradient.x - 25, y: gradient.y - 29, width: 154, color: 'gradient' },
  { tex: String.raw`-\lambda W`, x: weight.x - 59, y: weight.y + 34, width: 80, size: 17, color: 'decay', opacity: 0.65 },
];

function renderLabel({ tex, x, y, width, size = 18, color = 'foreground', opacity = 1 }, palette) {
  const node = adaptor.firstChild(math.convert(tex, { display: false }));
  const ex = size * 0.442;
  const mathWidth = parseFloat(adaptor.getAttribute(node, 'width')) * ex;
  const mathHeight = parseFloat(adaptor.getAttribute(node, 'height')) * ex;
  for (const [name, value] of Object.entries({
    x: x + (width - mathWidth) / 2,
    y: y + (30 - mathHeight) / 2,
    width: mathWidth,
    height: mathHeight,
    color: palette[color],
    opacity,
    'aria-hidden': 'true',
  })) adaptor.setAttribute(node, name, String(value));
  return adaptor.outerHTML(node);
}

function renderFigure(palette) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="500" height="244" viewBox="0 0 500 244" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="title desc">
  <title id="title">Scale-invariant weight geometry</title>
  <desc id="desc">A schematic with unequal axis scales: the constant-norm arc is compressed vertically. W lies at a model-space angle of 35 degrees. Its gradient is tangent to the arc. Four equally faint radial loss contours carry fifteen inward weight-decay arrows.</desc>
  <style>
    .vector, .arrow-head { fill: none; stroke-linecap: round; stroke-linejoin: round; }
    .axes { stroke: ${palette.foreground}; stroke-opacity: 0.6; stroke-width: 1.2; }
    .contours { stroke: ${palette.foreground}; stroke-opacity: 0.12; stroke-width: 1; }
    .contour-decay { stroke: ${palette.decay}; stroke-width: 1.2; opacity: 0.42; }
    .norm-circle { stroke: ${palette.foreground}; stroke-opacity: 0.5; stroke-width: 1.3; stroke-dasharray: 5 6; }
    .corner { stroke: ${palette.foreground}; stroke-opacity: 0.45; stroke-width: 1; }
    .gradient { stroke: ${palette.gradient}; stroke-width: 1.8; }
    .gradient-head { stroke: ${palette.gradient}; stroke-width: 1.2; }
    .decay { stroke: ${palette.decay}; stroke-width: 1.5; opacity: 0.65; }
    .decay-head { stroke: ${palette.decay}; stroke-width: 1.2; }
    .weight { fill: ${palette.foreground}; stroke: ${palette.background}; stroke-width: 1.5; }
  </style>
  <defs>
    <marker id="gradient-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow-head gradient-head" d="M1 1L7 4L1 7" /></marker>
    <marker id="decay-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow-head decay-head" d="M1 1L7 4L1 7" /></marker>
  </defs>
  <g class="contours">${contours.map((d) => `<path class="vector" d="${d}" />`).join('')}</g>
  <g class="contour-decay">${decayArrows.map((d) => `<path class="vector" d="${d}" marker-end="url(#decay-arrow)" />`).join('')}</g>
  <path class="vector axes" d="M48 6V${origin.y}H454M44 13L48 6L52 13M447 ${origin.y - 4}L454 ${origin.y}L447 ${origin.y + 4}" />
  <path class="vector norm-circle" d="M${origin.x + radius} ${origin.y}A${radius} ${radius * verticalScale} 0 0 0 ${origin.x} ${origin.y - radius * verticalScale}" />
  <path class="vector corner" d="M${corner.map(({ x, y }) => `${x} ${y}`).join('L')}" />
  <path class="vector decay" d="${line(weight, decay)}" marker-end="url(#decay-arrow)" />
  <path class="vector gradient" d="${line(weight, gradient)}" marker-end="url(#gradient-arrow)" />
  <circle class="weight" cx="${weight.x}" cy="${weight.y}" r="4" />
  ${labels.map((label) => renderLabel(label, palette)).join('\n  ')}
</svg>\n`;
}

const output = new URL('../../../../../public/plots/', import.meta.url);
await mkdir(output, { recursive: true });
for (const [theme, palette] of Object.entries({
  light: { foreground: '#242629', background: '#fdfdfd', gradient: '#1d6fd1', decay: '#9a3642' },
  dark: { foreground: '#ecf0f7', background: '#1e293b', gradient: '#60a5fa', decay: '#d9909a' },
})) {
  const file = new URL(`weight-norm-${theme}.svg`, output);
  await writeFile(file, renderFigure(palette));
  console.log(file.pathname);
}
