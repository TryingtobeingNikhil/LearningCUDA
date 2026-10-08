// tiny helpers on top of rough.js for hand-drawn diagrams
const INK = "#2b2b2b";
const COLORS = {
  teal: "#13a89e",
  coral: "#ff6b6b",
  yellow: "#f6c344",
  purple: "#8c6be8",
  muted: "#8a8478",
};

const svg = document.getElementById("canvas");
const rc = rough.svg(svg);
let seed = 7;

function add(node) { svg.appendChild(node); return node; }

// light hachure fill, drawn faded so text on top stays readable
function fill(node) { node.style.opacity = 0.45; return add(node); }

function box(x, y, w, h, opts = {}) {
  if (opts.fill) fill(rc.rectangle(x, y, w, h, {
    seed: seed++, stroke: "none", roughness: 1.3, fill: opts.fill,
    fillStyle: opts.fillStyle || "hachure", hachureGap: 9, fillWeight: 1.6, hachureAngle: -41,
  }));
  add(rc.rectangle(x, y, w, h, {
    seed: seed++, stroke: opts.stroke || INK, strokeWidth: opts.strokeWidth || 2, roughness: 1.3,
  }));
  if (opts.label !== undefined) {
    text(x + w / 2, y + h / 2 + (opts.size || 32) * 0.35, opts.label, {
      size: opts.size || 32, anchor: "middle", color: opts.color,
    });
  }
}

function circle(cx, cy, d, opts = {}) {
  if (opts.fill) fill(rc.circle(cx, cy, d, {
    seed: seed++, stroke: "none", roughness: 1.2, fill: opts.fill,
    fillStyle: "hachure", hachureGap: 6, fillWeight: 1.4,
  }));
  add(rc.circle(cx, cy, d, { seed: seed++, stroke: opts.stroke || INK, strokeWidth: 2, roughness: 1.2 }));
  if (opts.label !== undefined) {
    text(cx, cy + (opts.size || 30) * 0.35, opts.label, { size: opts.size || 30, anchor: "middle" });
  }
}

function line(x1, y1, x2, y2, opts = {}) {
  add(rc.line(x1, y1, x2, y2, {
    seed: seed++, stroke: opts.stroke || INK, strokeWidth: opts.strokeWidth || 2, roughness: 1.1,
  }));
}

function arrow(x1, y1, x2, y2, opts = {}) {
  line(x1, y1, x2, y2, opts);
  const a = Math.atan2(y2 - y1, x2 - x1), L = opts.head || 16, s = 0.45;
  line(x2, y2, x2 - L * Math.cos(a - s), y2 - L * Math.sin(a - s), opts);
  line(x2, y2, x2 - L * Math.cos(a + s), y2 - L * Math.sin(a + s), opts);
}

function curve(points, opts = {}) {
  add(rc.curve(points, { seed: seed++, stroke: opts.stroke || INK, strokeWidth: 2, roughness: 1 }));
}

function text(x, y, str, opts = {}) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  t.setAttribute("font-family", opts.font || "'Patrick Hand', cursive");
  t.setAttribute("font-size", opts.size || 30);
  t.setAttribute("fill", opts.color || INK);
  t.setAttribute("text-anchor", opts.anchor || "start");
  if (opts.weight) t.setAttribute("font-weight", opts.weight);
  t.style.whiteSpace = "pre";
  t.textContent = str;
  return add(t);
}

// a wobbly highlighter stroke under a text element
function underline(el, color) {
  const b = el.getBBox();
  const y = b.y + b.height - 2;
  const hl = rc.line(b.x - 4, y, b.x + b.width + 4, y, { seed: seed++, stroke: color, strokeWidth: 12, roughness: 1.2 });
  hl.style.opacity = 0.45;
  svg.insertBefore(hl, el);
}

// text that continues right after another text element on the same line
function after(el, str, opts = {}, gap = 14) {
  const x = +el.getAttribute("x") + el.getComputedTextLength() + gap;
  return text(x, +el.getAttribute("y"), str, opts);
}

function title(main, sub) {
  text(80, 115, main, { font: "'Caveat', cursive", size: 84, weight: 700 });
  if (sub) text(84, 165, sub, { size: 32, color: COLORS.muted });
}

// wait until both fonts are really loaded, so text measurements are right
const fontsReady = Promise.all([
  document.fonts.load("32px 'Patrick Hand'"),
  document.fonts.load("700 32px Caveat"),
]);
