import { visit } from "unist-util-visit";

const supportedTypes = new Set([
  "theorem",
  "proposition",
  "lemma",
  "definition",
  "proof",
  "remark",
  "corollary",
  "example",
]);

const numberedTypes = new Set([
  "theorem",
  "proposition",
  "lemma",
  "definition",
  "corollary",
]);

const typeTitles = {
  theorem: "Theorem",
  proposition: "Proposition",
  lemma: "Lemma",
  definition: "Definition",
  proof: "Proof",
  remark: "Remark",
  corollary: "Corollary",
  example: "Example",
};

function attributeMap(attributes) {
  const map = Object.create(null);
  if (!attributes) {
    return map;
  }
  if (Array.isArray(attributes)) {
    for (const attribute of attributes) {
      if (!attribute || typeof attribute.name !== "string") {
        continue;
      }
      const key = attribute.name;
      const value =
        attribute.value === null || attribute.value === undefined
          ? true
          : String(attribute.value);
      map[key] = value;
    }
    return map;
  }
  return Object.fromEntries(Object.entries(attributes));
}

function normalizeLabel(value) {
  const normalized = typeof value === "string" ? value.trim() : "";
  return normalized ? normalized.replace(/^#/, "") : "";
}

function directiveLabel(node) {
  if (typeof node.label === "string" && node.label.trim()) {
    return node.label.trim();
  }
  const attrs = attributeMap(node.attributes);
  const raw = attrs.label || attrs["#"] || attrs.id;
  return normalizeLabel(raw);
}

function directiveTitle(node) {
  if (typeof node.label === "string" && node.label.trim()) {
    return node.label.trim();
  }
  const attrs = attributeMap(node.attributes);
  return typeof attrs.title === "string" ? attrs.title.trim() : "";
}

function refTarget(node) {
  if (typeof node.label === "string" && node.label.trim()) {
    return normalizeLabel(node.label);
  }
  const attrs = attributeMap(node.attributes);
  const raw = attrs.label || attrs["#"] || attrs.id;
  if (raw) {
    return normalizeLabel(raw);
  }
  if (Array.isArray(node.children)) {
    const first = node.children[0];
    if (
      first &&
      first.type === "text" &&
      typeof first.value === "string" &&
      first.value.trim()
    ) {
      return normalizeLabel(first.value);
    }
  }
  return "";
}

function theoremHead(type, headingText, titleText) {
  const children = [];

  if (type === "proof") {
    const text =
      titleText.length > 0 ? `Proof (${titleText}).` : "Proof.";
    children.push({
      type: "emphasis",
      children: [{ type: "text", value: text }],
    });
    return {
      type: "paragraph",
      data: {
        hName: "p",
        hProperties: { className: "theorem-env__head" },
      },
      children,
    };
  }

  if (headingText.length > 0) {
    children.push({
      type: "strong",
      children: [{ type: "text", value: headingText }],
    });
  }

  if (titleText.length > 0) {
    children.push({ type: "text", value: " " });
    children.push({
      type: "emphasis",
      children: [{ type: "text", value: `(${titleText})` }],
    });
  }

  if (type !== "proof") {
    children.push({
      type: "strong",
      children: [{ type: "text", value: "." }],
    });
  }

  return {
    type: "paragraph",
    data: {
      hName: "p",
      hProperties: { className: "theorem-env__head" },
    },
    children,
  };
}

export function remarkTheorem() {
  return function transformer(tree) {
    const counters = {
      theorem: 0,
      proposition: 0,
      lemma: 0,
      definition: 0,
      corollary: 0,
    };

    const labelToNumber = new Map();

    visit(tree, "containerDirective", (node) => {
      const type = node.name;
      if (!supportedTypes.has(type)) {
        return;
      }

      const label = directiveLabel(node);
      const title = directiveTitle(node);
      const isNumbered = numberedTypes.has(type);
      let headingText = typeTitles[type];

      if (isNumbered) {
        const value = (counters[type] += 1);
        headingText = `${typeTitles[type]} ${value}`;
        if (label) {
          labelToNumber.set(label, headingText);
        }
      }

      const bodyChildren = Array.isArray(node.children) ? [...node.children] : [];
      if (type === "proof") {
        bodyChildren.push({
          type: "html",
          value: '<span class="theorem-env__qed">□</span>',
        });
      }

      node.children = [
        theoremHead(type, headingText, title),
        { type: "html", value: '<div class="theorem-env__body">' },
        ...bodyChildren,
        { type: "html", value: "</div>" },
      ];

      const data = node.data || (node.data = {});
      data.hName = "div";
      data.hProperties = {
        className: `theorem-env theorem-env--${type}`,
      };
      if (label) {
        data.hProperties.id = label;
      }
    });

    visit(tree, "textDirective", (node, index, parent) => {
      if (node.name !== "ref" || !parent || typeof index !== "number") {
        return;
      }

      const target = refTarget(node);
      const replacement = target ? labelToNumber.get(target) : null;
      if (!replacement) {
        parent.children[index] = {
          type: "text",
          value: target ? `(${target})` : "Ref",
        };
        return;
      }

      parent.children[index] = { type: "text", value: replacement };
    });
  };
}

