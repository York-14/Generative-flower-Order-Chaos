/* app.js — 画面の操作。計算はすべて worker.js 側 */
(() => {
"use strict";
const $ = id => document.getElementById(id);
const CP = window.CoasterPipeline, FC = window.FlowerCore;
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ───────── パラメータ ─────────
const PNG_THRESHOLD = 0.065;                       // PNG は点が少なく粒が粗いので少し低めが合う
const SLIDERS = [
  ["diameter",    "外径",       30, 200, 0.5, 1, "mm"],
  ["base",        "厚み",       0.6, 5, 0.05, 2, "mm"],
  ["relief",      "凹凸",       0, 3, 0.05, 2, "mm"],
  ["threshold",   "しきい値",   0.02, 0.3, 0.005, 3, ""],
  ["grow",        "線の太さ",   0, 1, 0.01, 2, "mm"],
  ["bridgeWidth", "ブリッジ幅", 0.6, 4, 0.1, 1, "mm"],
  ["minHole",     "最小の穴",   0, 6, 0.1, 1, "mm²"],
  ["blur",        "なめらかさ", 0, 0.6, 0.01, 2, "mm"],
  ["cell",        "メッシュ",   0.08, 0.3, 0.01, 2, "mm"],
];
const HELP = {
  diameter: "でき上がりの外径", base: "どこでも確保する厚み", relief: "明るい部分ほど高くする量",
  threshold: "このトーン以上を構造として残す（下げると太く・つながりやすく）", grow: "細い線を太らせる半径（線幅は約 2 倍）",
  bridgeWidth: "離れた部分をつなぐ橋の幅", minHole: "これより小さい穴は埋める", blur: "トーンのぼかし",
  cell: "メッシュの細かさ（小さいほど精密・ファイルが重い）",
};
const KEY = "ocs.params.v1";
const P = { ...CP.DEFAULTS, ...(store.get(KEY) || {}) };
let mode = null;                                  // "seed" | "image"
const shows = [];

for (const [k, label, lo, hi, st, dg, unit] of SLIDERS) {
  const lab = document.createElement("label"); lab.className = "sl"; lab.title = HELP[k] || "";
  lab.innerHTML = `<span>${label}</span>`;
  const inp = Object.assign(document.createElement("input"), { type: "range", min: lo, max: hi, step: st });
  const out = document.createElement("output");
  const show = () => { inp.value = P[k]; out.textContent = `${(+P[k]).toFixed(dg)}${unit ? "" : ""}`; };
  inp.addEventListener("input", () => { P[k] = +inp.value; out.textContent = (+inp.value).toFixed(dg); changed(); });
  lab.append(inp, out); $("sliders").append(lab); shows.push(show); show();
}
for (let n = 2; n <= 12; n++) $("nsel").append(new Option(String(n), String(n)));
$("bridges").checked = P.bridges; $("symmetric").checked = P.symmetric;
$("bridges").addEventListener("change", () => { P.bridges = $("bridges").checked; changed(); });
$("symmetric").addEventListener("change", () => { P.symmetric = $("symmetric").checked; changed(); });
$("nsel").addEventListener("change", changed);
$("reset").addEventListener("click", () => {
  Object.assign(P, CP.DEFAULTS); if (mode === "image") P.threshold = PNG_THRESHOLD;
  $("bridges").checked = P.bridges; $("symmetric").checked = P.symmetric; $("nsel").value = "0";
  shows.forEach(f => f()); changed();
});

// ───────── worker ─────────
const worker = new Worker("js/worker.js");
let job = 0, building = false, pending = false, hasSource = false, timer = null;
let positions = null, lastStats = null, baseName = "order-chaos";

function changed() {
  store.set(KEY, P);
  clearTimeout(timer); timer = setTimeout(build, 350);
}
function build() {
  if (!hasSource) return;
  if (building) { pending = true; return; }
  building = true; pending = false;
  const params = {}; for (const k of Object.keys(CP.DEFAULTS)) params[k] = P[k];
  worker.postMessage({ type: "build", job: ++job, params, n: +$("nsel").value });
}
function busy(text) { $("busy").hidden = !text; if (text) $("busyText").textContent = text; }
function message(text, err = false) { $("message").textContent = text || ""; $("message").classList.toggle("error", err); }

worker.onmessage = e => {
  const m = e.data;
  if (m.type === "status") busy(m.text);
  else if (m.type === "error") { busy(null); building = false; message(m.text, true); }
  else if (m.type === "source") {
    hasSource = true; mode = m.mode; $("empty").hidden = true;
    $("nOut").textContent = `n=${m.n}`;
    if (m.mode === "image") $("imgInfo").textContent = `色: ${m.method}` + (pngSeed ? `　シード ${pngSeed.label}` : "");
    message(m.mode === "seed"
      ? `シードから計算しました（n=${m.n}, λ=${m.params.lam.toFixed(3)}, α=${m.params.alpha.toFixed(3)}, β=${m.params.beta.toFixed(3)}, γ=${m.params.gamma.toFixed(3)}, ω=${m.params.omega.toFixed(3)}）`
      : "PNG の色からトーンを復元しました。シードが分かる場合は、シードから作ると線がより精密になります。");
    building = false; build();
  } else if (m.type === "result") {
    building = false;
    if (pending) { build(); return; }
    busy(null);
    draw2D(m.preview); positions = m.positions; lastStats = m.stats;
    $("legend").hidden = false; $("save").disabled = false;
    const st = m.stats, dd = $("stats").querySelectorAll("dd");
    dd[0].textContent = `${st.diameter.toFixed(1)} mm`;
    dd[1].textContent = `${(P.base).toFixed(2)}〜${st.height.toFixed(2)} mm`;
    dd[2].textContent = `${(st.volume / 1000 * 1.24).toFixed(1)} g`;
    dd[3].textContent = st.triangles.toLocaleString();
    $("nOut").textContent = `n=${st.n}`;
    if (st.bridgeFrac > 0.2)
      message(`ブリッジが形の ${Math.round(st.bridgeFrac * 100)}% を占めています。花がばらばらの島に分かれているので、しきい値を下げるか線の太さを上げると、花本来の線が残ります。`, true);
    else if ($("message").classList.contains("error")) message("");
    update3D();
  }
};

// ───────── 2D / 3D 表示 ─────────
function draw2D(rgba) {
  const c = $("view2d"), s = Math.round(Math.sqrt(rgba.length / 4));
  c.width = s; c.height = s; c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba), s, s), 0, 0);
}
let three = null;
function initThree() {
  if (three || !window.THREE) return three;
  const el = $("view3d"), renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); el.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0b10);
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 2000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(-1, 0.6, 1); scene.add(key);
  const rim = new THREE.DirectionalLight(0xffd6e6, 0.35); rim.position.set(1, 0.3, -1); scene.add(rim);
  const controls = new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  const material = new THREE.MeshStandardMaterial({ color: 0xa3a3aa, roughness: 0.85, metalness: 0 });
  three = { renderer, scene, camera, controls, material, mesh: null, el };
  const resize = () => { const w = el.clientWidth, h = el.clientHeight; if (!w) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
  new ResizeObserver(resize).observe(el); resize();
  (function loop() { requestAnimationFrame(loop); if (!el.hidden) { controls.update(); renderer.render(scene, camera); } })();
  return three;
}
function update3D() {
  if (!positions || $("view3d").hidden) return;
  const t = initThree();
  if (!t) { message("3D 表示用のライブラリを読み込めませんでした（オフライン？）。STL の保存はできます。", true); return; }
  if (t.mesh) { t.scene.remove(t.mesh); t.mesh.geometry.dispose(); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.computeVertexNormals();
  t.mesh = new THREE.Mesh(g, t.material); t.mesh.rotation.x = -Math.PI / 2;    // STL は z 上 → three は y 上
  t.scene.add(t.mesh);
  const r = (lastStats ? lastStats.diameter : 90) / 2;
  if (!t.placed) { t.camera.position.set(0, r * 3.0, r * 2.0); t.controls.target.set(0, 0, 0); t.placed = true; }
}
for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === b));
  const is3d = b.dataset.view === "3d";
  $("view3d").hidden = !is3d; $("view2d").hidden = is3d; $("legend").style.visibility = is3d ? "hidden" : "visible";
  if (is3d) update3D();
});

// ───────── シード入力 ─────────
function runSeed(seed, idx) {
  mode = "seed"; baseName = `oc_${FC.pad12(seed)}_f${idx + 1}`;
  if (P.threshold === PNG_THRESHOLD) { P.threshold = CP.DEFAULTS.threshold; shows.forEach(f => f()); }
  worker.postMessage({ type: "seed", seed, idx, points: +$("points").value });
}
$("runSeed").addEventListener("click", () => {
  let s;
  try { s = FC.parseSeed($("seed").value); } catch (e) { message(e.message, true); return; }
  if (s === null) { message("シードを入力してください", true); return; }
  runSeed(s, +$("flower").value);
});

// ───────── PNG 入力 ─────────
let img = null, pngSeed = null;
function readPngText(buf) {                       // tEXt チャンク（Order&Chaos が Seed を書き込む）
  const u = new Uint8Array(buf), out = {};
  if (u[0] !== 0x89 || u[1] !== 0x50) return out;
  const dv = new DataView(buf);
  for (let o = 8; o + 8 <= u.length;) {
    const len = dv.getUint32(o), type = String.fromCharCode(...u.subarray(o + 4, o + 8));
    if (type === "tEXt") { const d = u.subarray(o + 8, o + 8 + len), z = d.indexOf(0);
      out[new TextDecoder("latin1").decode(d.subarray(0, z))] = new TextDecoder("latin1").decode(d.subarray(z + 1)); }
    if (type === "IEND") break;
    o += 12 + len;
  }
  return out;
}
async function loadFile(file) {
  if (!file) return;
  if (!/^image\//.test(file.type) && !/\.png$/i.test(file.name)) { message("画像ファイル（PNG）を選んでください", true); return; }
  const buf = await file.arrayBuffer(), meta = readPngText(buf);
  const bmp = await createImageBitmap(new Blob([buf], { type: file.type || "image/png" }));
  const c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d"); ctx.drawImage(bmp, 0, 0);
  img = { w: bmp.width, h: bmp.height, data: ctx.getImageData(0, 0, bmp.width, bmp.height).data, bmp };
  pngSeed = null;
  if (meta.Seed) {
    const m = /^(\d{1,12})(\s*\(edited\))?/.exec(meta.Seed.trim());
    if (m) pngSeed = { seed: Number(m[1]), edited: !!m[2], label: FC.pad12(Number(m[1])) + (m[2] ? "（編集あり）" : "") };
  }
  const isGrid = img.w === img.h && (img.w >= 900 || /2D ver\.2/.test(meta.Title || ""));
  $("grid3").checked = isGrid;
  baseName = file.name.replace(/\.[^.]+$/, "");
  if (pngSeed) $("seed").value = FC.pad12(pngSeed.seed);
  $("picker").hidden = false;
  $("useSeed").hidden = !(pngSeed && !pngSeed.edited && isGrid);
  if (pngSeed && pngSeed.edited) message("この PNG はシードから編集されているため、シードからは同じ花を再現できない可能性があります。PNG の色から作ります。");
  drawPicker(); runImage();
}
function cellRect() {
  if (!$("grid3").checked) { const s = Math.min(img.w, img.h); return [Math.floor((img.w - s) / 2), Math.floor((img.h - s) / 2), s, s]; }
  const cw = Math.floor(img.w / 3), ch = Math.floor(img.h / 3), i = +$("flower").value;
  return [(i % 3) * cw, Math.floor(i / 3) * ch, cw, ch];
}
function drawPicker() {
  const c = $("pick"), ctx = c.getContext("2d"), S = c.width;
  ctx.clearRect(0, 0, S, S); ctx.drawImage(img.bmp, 0, 0, S, S);
  if (!$("grid3").checked) return;
  const i = +$("flower").value, s = S / 3;
  ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = 1;
  for (let k = 1; k < 3; k++) { ctx.beginPath(); ctx.moveTo(k * s, 0); ctx.lineTo(k * s, S); ctx.moveTo(0, k * s); ctx.lineTo(S, k * s); ctx.stroke(); }
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#f07aa8"; ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 4]); ctx.strokeRect((i % 3) * s + 2, Math.floor(i / 3) * s + 2, s - 4, s - 4); ctx.setLineDash([]);
}
function runImage() {
  mode = "image";
  if (P.threshold === CP.DEFAULTS.threshold) { P.threshold = PNG_THRESHOLD; shows.forEach(f => f()); }
  const rect = cellRect(), base = baseName.replace(/_f\d$/, "");
  baseName = $("grid3").checked ? `${base}_f${+$("flower").value + 1}` : base;
  const copy = new Uint8ClampedArray(img.data);
  worker.postMessage({ type: "image", rgba: copy.buffer, w: img.w, h: img.h, rect }, [copy.buffer]);
}
$("pick").addEventListener("click", e => {
  if (!img || !$("grid3").checked) return;
  const r = e.currentTarget.getBoundingClientRect();
  const i = Math.min(2, Math.floor((e.clientX - r.left) / r.width * 3)), j = Math.min(2, Math.floor((e.clientY - r.top) / r.height * 3));
  $("flower").value = String(j * 3 + i); drawPicker(); runImage();
});
$("grid3").addEventListener("change", () => { if (img) { drawPicker(); runImage(); } });
$("flower").addEventListener("change", () => { if (img && mode === "image") { drawPicker(); runImage(); } });
$("useSeed").addEventListener("click", () => { if (pngSeed) { $("seed").value = FC.pad12(pngSeed.seed); runSeed(pngSeed.seed, +$("flower").value); } });
$("file").addEventListener("change", e => loadFile(e.target.files[0]));
for (const el of [$("drop"), $("stage"), document.body]) {
  el.addEventListener("dragover", e => { e.preventDefault(); (el === document.body ? $("stage") : el).classList.add("dragover"); });
  el.addEventListener("dragleave", () => { $("drop").classList.remove("dragover"); $("stage").classList.remove("dragover"); });
  el.addEventListener("drop", e => { e.preventDefault(); e.stopPropagation(); $("drop").classList.remove("dragover"); $("stage").classList.remove("dragover");
    loadFile(e.dataTransfer.files[0]); });
}

// ───────── 保存 ─────────
// スマホ（タッチ端末）は共有シートで「ファイルに保存」や AirDrop に渡す。使えなければ通常のダウンロード
const touch = matchMedia("(pointer: coarse)").matches;
$("save").addEventListener("click", async () => {
  if (!positions) return;
  const name = `${baseName}_d${Math.round(lastStats.diameter)}.stl`;
  const data = CP.toSTL(positions, `Order&Chaos ${baseName}`);
  if (touch && navigator.canShare) {
    const file = new File([data], name, { type: "application/octet-stream" });
    if (navigator.canShare({ files: [file] })) {
      try { await navigator.share({ files: [file], title: name }); return; }
      catch (e) { if (e.name === "AbortError") return; }      // 失敗したらダウンロードへ
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data], { type: "application/octet-stream" })); a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
});
if (touch) {
  $("drop").querySelector("strong").textContent = "PNG を選ぶ";
  $("drop").querySelector("span").textContent = "写真またはファイルから";
  $("empty").textContent = "右下（下）のパネルで PNG を選ぶか、シードを入力してください";
  $("save").textContent = "STL を保存・共有";
}

// ───────── URL ?seed=…&flower=… ─────────
const q = new URLSearchParams(location.search);
if (q.get("q")) {                                // ?q=n,λ,α,β,γ,ω（Order&Chaos で編集した花）
  const v = q.get("q").split(",").map(Number);
  if (v.length === 6 && v.every(Number.isFinite) && v[0] >= 2 && v[0] <= 16) {
    const p = { n: Math.round(v[0]), lam: v[1], alpha: v[2], beta: v[3], gamma: v[4], omega: v[5] };
    mode = "seed"; baseName = `oc_n${p.n}_${q.get("name") || "edited"}`.replace(/[^\w.-]/g, "_");
    worker.postMessage({ type: "params", q: p, points: +$("points").value });
  } else message("URL のパラメータ（q=n,λ,α,β,γ,ω）を読めませんでした", true);
} else if (q.get("seed")) {
  try {
    const s = FC.parseSeed(q.get("seed")), f = Math.min(9, Math.max(1, +(q.get("flower") || 3))) - 1;
    $("seed").value = FC.pad12(s); $("flower").value = String(f); runSeed(s, f);
  } catch (e) { message(e.message, true); }
}
})();
