import { JSDOM } from 'jsdom';
import { jsdomAdaptor } from 'mathjax-full/js/adaptors/jsdomAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { mathjax } from 'mathjax-full/js/mathjax.js';

const adaptor = jsdomAdaptor(JSDOM);
RegisterHTMLHandler(adaptor);

export function renderMathLabels(expressions) {
  const document = mathjax.document('', {
    InputJax: new TeX(),
    // Inline paths avoid duplicate glyph IDs when figures share a page with MDX math.
    OutputJax: new SVG({ fontCache: 'none' }),
  });
  return expressions.map((expression) => adaptor.outerHTML(document.convert(expression, { display: false })));
}
