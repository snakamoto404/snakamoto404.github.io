import { fromDom } from 'hast-util-from-dom';
import { toText } from 'hast-util-to-text';
import { SKIP, visitParents } from 'unist-util-visit-parents';
import { JSDOM } from 'jsdom';
import { jsdomAdaptor } from 'mathjax-full/js/adaptors/jsdomAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { AllPackages } from 'mathjax-full/js/input/tex/AllPackages.js';
import { mathjax } from 'mathjax-full/js/mathjax.js';
import { SVG } from 'mathjax-full/js/output/svg.js';

const adaptor = jsdomAdaptor(JSDOM);
RegisterHTMLHandler(adaptor);

const EMPTY_OPTIONS = {};
const EMPTY_CLASSES = [];

function hasRefCommand(value) {
  return /\\(?:eqref|ref)\s*\{/.test(value);
}

function hasUnresolvedRef(node) {
  const stack = [node];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || current.type !== 'element') continue;

    if (current.tagName === 'a' && current.properties?.href === '#') {
      return true;
    }

    if (Array.isArray(current.children)) {
      for (const child of current.children) {
        stack.push(child);
      }
    }
  }

  return false;
}

function createRenderer(options) {
  const output = new SVG(options.svg || undefined);
  const input = new TeX({ packages: AllPackages, ...options.tex });
  const doc = mathjax.document('', { InputJax: input, OutputJax: output });

  return {
    render(value, renderOptions) {
      const domNode = doc.convert(value, renderOptions);
      return [fromDom(domNode)];
    },
    styleSheet() {
      const value = adaptor.textContent(output.styleSheet(doc));
      return {
        type: 'element',
        tagName: 'style',
        properties: {},
        children: [{ type: 'text', value }]
      };
    }
  };
}

export default function rehypeMathjaxTwoPass(options = EMPTY_OPTIONS) {
  return function transformer(tree) {
    // MathJax TeX/SVG docs keep internal state (labels/counters/macros). In dev
    // watch mode, reusing one renderer across transforms can leak state between
    // reloads and produce flaky equation output. Create a fresh renderer per tree.
    const renderer = createRenderer(options);
    let found = false;
    let context = tree;
    const rerenderQueue = [];

    visitParents(tree, 'element', function (element, parents) {
      const classes = Array.isArray(element.properties?.className)
        ? element.properties.className
        : EMPTY_CLASSES;
      const languageMath = classes.includes('language-math');
      const mathDisplay = classes.includes('math-display');
      const mathInline = classes.includes('math-inline');
      let display = mathDisplay;

      if (element.tagName === 'head') {
        context = element;
      }

      if (!languageMath && !mathDisplay && !mathInline) {
        return;
      }

      let parent = parents[parents.length - 1];
      let scope = element;

      if (
        element.tagName === 'code' &&
        languageMath &&
        parent &&
        parent.type === 'element' &&
        parent.tagName === 'pre'
      ) {
        scope = parent;
        parent = parents[parents.length - 2];
        display = true;
      }

      if (!parent || !('children' in parent)) return;

      const value = toText(element, { whitespace: 'pre' }).replace(/\r?\n|\r/g, '');
      const index = parent.children.indexOf(scope);
      if (index < 0) return;

      const rendered = renderer.render(value, { display });
      parent.children.splice(index, 1, ...rendered);
      found = true;

      if (hasRefCommand(value) && rendered.some(hasUnresolvedRef)) {
        rerenderQueue.push({ parent, index, value, display });
      }

      return [SKIP, index];
    });

    // Two-pass resolution: once all labels have been seen, rerender unresolved refs.
    for (const entry of rerenderQueue) {
      const { parent, index, value, display } = entry;
      if (!parent.children[index]) continue;
      const rendered = renderer.render(value, { display });
      parent.children.splice(index, 1, ...rendered);
    }

    if (found && renderer.styleSheet) {
      context.children.unshift(renderer.styleSheet());
    }
  };
}
