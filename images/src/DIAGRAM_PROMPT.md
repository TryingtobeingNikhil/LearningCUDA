# Diagram prompt (paste into any Claude chat)

Copy everything below the line into a new Claude chat (any account or model). Fill in the **MY TOPIC** part at the bottom and attach or paste your `.cu` code.

---

You are making one diagram for my "learning CUDA" posts on X. Keep it **simple and minimal**: a beginner should understand the idea in about 10 seconds.

## Output
- One complete, self-contained HTML file. Use the template below exactly as written, and only replace the part marked `DRAW HERE`.
- Canvas is 1600×900 (16:9). Keep everything inside it with a margin of at least 80px.
- No "100 days" text anywhere. The subtitle is always `day XX · <short idea>`.

## Style rules (do not change)
- Hand-drawn look, like a sketch in a notebook: rough.js shapes on a cream background (`#fdfaf3`).
- Fonts: `Caveat` bold for the title and big labels, `Patrick Hand` for everything else.
- Colours always mean the same thing:
  - yellow = CPU / host
  - teal = GPU / device
  - coral = the ONE thing to notice (one highlighted element, one example)
  - purple = an extra category, only if really needed
  - muted grey = side notes
- Fills are light diagonal pencil hatching (the helpers already do this). Never use solid fills or gradients.
- Use a highlighter `underline()` once per diagram, on the key takeaway.
- At most ~3 short notes of text. No paragraphs. Show it with a picture instead of explaining it in words.
- Use small concrete examples (e.g. 8 elements, 2 blocks of 4 threads) instead of real sizes.
- Text must never overlap a shape edge or other text. Use `after()` when text continues on the same line.

## Helpers available
- `title(main, sub)`: title at the top-left
- `box(x, y, w, h, { fill, label, size })`: rectangle; `fill` is a COLORS value
- `circle(cx, cy, diameter, { fill, label })`
- `line(x1, y1, x2, y2)`, `arrow(x1, y1, x2, y2, { head })`, `curve([[x,y], ...])`
- `text(x, y, str, { size, color, anchor, font, weight })`: returns the element. `y` is the baseline.
- `underline(textElement, COLORS.coral)`: highlighter stroke under a text element
- `after(textElement, str, opts)`: text that continues right after another text element
- `COLORS.teal / coral / yellow / purple / muted`, `INK`

## Template
```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@700&family=Patrick+Hand&display=block" rel="stylesheet">
<style>body { margin: 0; background: #fdfaf3; } svg { display: block; }</style>
</head>
<body>
<svg id="canvas" width="1600" height="900" viewBox="0 0 1600 900"></svg>
<script src="https://cdn.jsdelivr.net/npm/roughjs@4.6.6/bundled/rough.js"></script>
<script>
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
</script>
<script>
fontsReady.then(() => {
  // ===== DRAW HERE =====
  // example:
  // title("vector addition", "day 01 · one thread per element");
  // box(90, 290, 300, 240, { fill: COLORS.yellow, label: "CPU" });
  // arrow(400, 410, 640, 410);
  // underline(text(90, 770, "the key takeaway", { size: 32 }), COLORS.coral);
});
</script>
</body>
</html>
```

## MY TOPIC
- Day: XX
- What I learned / built today: ...
- The one idea the diagram should explain: ...
- My code: (paste or attach the .cu file)
