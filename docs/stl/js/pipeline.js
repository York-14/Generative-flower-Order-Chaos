/* pipeline.js — Order&Chaos の花 → レース状コースター（構造だけ）→ STL
 *
 *   tone (0..1 の画面トーン) ─▶ structure mask ─▶ mesh / STL
 *
 * 入力は 2 通り:
 *   toneFromSeed : シード + 花番号から写像を直接回して密度場を作る（高精度）
 *   toneFromImage: Order&Chaos が保存した PNG の色をトーンに逆変換する
 * ブラウザ（Worker）と Node の両方で動く。依存は flower-core.js のみ。
 */
(function (root) {
"use strict";
const FC = () => root.FlowerCore;

const DEFAULTS = {
  diameter: 91,      // 外径 (mm)
  flat: true,        // 上面を平らにして輪郭をなめらかな曲線で作る（FDM 向け）。false で凹凸付き
  base: 2.4,         // 厚み (mm)。凹凸ありのときは「どこでも確保する厚み」
  relief: 0.9,       // 凹凸ありのとき、明るさ 1 で足す高さ (mm)
  smooth: 0.20,      // 輪郭のなめらかさ (mm)。上面を平らにするときの角の丸め・ヒゲの除去
  threshold: 0.08,   // 構造とみなすトーン
  grow: 0.30,        // 細い線を太らせる半径 (mm) → 線幅は約 2×grow 以上
  blur: 0.12,        // トーンの平滑化 (mm)
  bridgeWidth: 1.4,  // ブリッジ幅 (mm)
  bridges: true,     // 離れた部分をブリッジでつなぐ
  symmetric: true,   // ブリッジを n 回対称に複製する
  minHole: 1.2,      // これより小さい穴は埋める (mm²)。FDM では直径 1.2 mm 未満の穴はつぶれやすい
  close: 0.15,       // 隙間を閉じる半径 (mm)
  majorFrac: 0.02,   // 最大成分のこの割合以上の島は「つなぐ対象」、未満は捨てる
  cell: 0.10,        // 格子 (mm)。凹凸ありはこのマス目でメッシュ、平らなら輪郭を取る細かさ
  gamma: 0.80,       // シード入力時のトーンカーブ（ページの既定値）
  points: 3e8,       // シード入力時の軌道の点数
  res: 1400,         // シード入力時の密度場の解像度 (px)
};

// トーンカーブの基準。最初に作ったコースター（シード 011291810887 の #3）の見た目を
// 「ページ 1px あたりの点数 × DENSITY_REF」で再現する定数（= (442/1400)^4）
const DENSITY_REF = Math.pow(442 / 1400, 4);
const pageCell = () => Math.round(FC().CANVAS / 3 * FC().CELL_FRAC);

// ───────────────────────────── 画像処理の基本 ─────────────────────────────
function gaussianBlur(src, W, H, sigma) {
  if (!(sigma > 0.3)) return Float32Array.from(src);
  const r = Math.ceil(sigma * 3), k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-i * i / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const o = y * W;
    for (let x = 0; x < W; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < W) a += src[o + xx] * k[i + r]; }
      tmp[o + x] = a;
    }
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < H) a += tmp[yy * W + x] * k[i + r]; }
    out[y * W + x] = a;
  }
  return out;
}

function bilinear(img, W, H, x, y) {            // x: column, y: row（範囲外は 0）
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const g = (i, j) => (i >= 0 && i < W && j >= 0 && j < H) ? img[j * W + i] : 0;
  return (g(x0, y0) * (1 - fx) + g(x0 + 1, y0) * fx) * (1 - fy) + (g(x0, y0 + 1) * (1 - fx) + g(x0 + 1, y0 + 1) * fx) * fy;
}

function resample(img, W, H, f) {               // 等倍率の拡大（双線形）
  const W2 = Math.round(W * f), H2 = Math.round(H * f), out = new Float32Array(W2 * H2);
  for (let y = 0; y < H2; y++) for (let x = 0; x < W2; x++)
    out[y * W2 + x] = bilinear(img, W, H, (x + 0.5) / f - 0.5, (y + 0.5) / f - 0.5);
  return { img: out, W: W2, H: H2 };
}

// 2 乗ユークリッド距離変換（Felzenszwalb）。最寄りの true 画素のインデックスも返す
function edt(mask, W, H) {
  const INF = 1e20, N = Math.max(W, H);
  const d = new Float64Array(W * H), src = new Int32Array(W * H);
  const f = new Float64Array(N), z = new Float64Array(N + 1), v = new Int32Array(N), dd = new Float64Array(N), arg = new Int32Array(N);
  function dt1(n) {
    let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < n; q++) {
      let s;
      for (;;) {
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        if (s <= z[k] && k > 0) k--; else break;
      }
      if (s <= z[k]) { v[0] = q; z[0] = -INF; z[1] = INF; k = 0; continue; }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; dd[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; arg[q] = v[k]; }
  }
  const rowOf = new Int32Array(W * H);
  for (let x = 0; x < W; x++) {                  // 列方向
    for (let y = 0; y < H; y++) f[y] = mask[y * W + x] ? 0 : INF;
    dt1(H);
    for (let y = 0; y < H; y++) { d[y * W + x] = dd[y]; rowOf[y * W + x] = arg[y]; }
  }
  for (let y = 0; y < H; y++) {                  // 行方向
    for (let x = 0; x < W; x++) f[x] = d[y * W + x];
    dt1(W);
    for (let x = 0; x < W; x++) { const sx = arg[x]; d[y * W + x] = dd[x]; src[y * W + x] = rowOf[y * W + sx] * W + sx; }
  }
  return { d, src };
}

const dilate = (m, W, H, r) => { if (r <= 0) return Uint8Array.from(m); const { d } = edt(m, W, H), o = new Uint8Array(W * H), r2 = r * r; for (let i = 0; i < o.length; i++) o[i] = d[i] <= r2; return o; };
const invert = m => { const o = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) o[i] = !m[i]; return o; };
const erode = (m, W, H, r) => invert(dilate(invert(m), W, H, r));
const closing = (m, W, H, r) => erode(dilate(m, W, H, r), W, H, r);

// 連結成分（conn8 = 8 近傍）。lab は 1 始まり、sizes[0] は未使用
function label(mask, W, H, conn8 = false) {
  const lab = new Int32Array(W * H), sizes = [0], st = new Int32Array(W * H);
  const nb = conn8 ? [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]] : [[1, 0], [-1, 0], [0, 1], [0, -1]];
  let n = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i] || lab[i]) continue;
    n++; let sp = 0, cnt = 0; st[sp++] = i; lab[i] = n;
    while (sp) {
      const p = st[--sp], px = p % W, py = (p - px) / W; cnt++;
      for (const [dx, dy] of nb) {
        const x = px + dx, y = py + dy;
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        const q = y * W + x;
        if (mask[q] && !lab[q]) { lab[q] = n; st[sp++] = q; }
      }
    }
    sizes.push(cnt);
  }
  return { lab, sizes, n };
}
const largestId = sizes => { let b = 1; for (let i = 2; i < sizes.length; i++) if (sizes[i] > sizes[b]) b = i; return b; };

function fillSmallHoles(mask, W, H, maxArea) {
  const { lab, sizes } = label(invert(mask), W, H, false), touches = new Uint8Array(sizes.length), o = Uint8Array.from(mask);
  for (let x = 0; x < W; x++) { touches[lab[x]] = 1; touches[lab[(H - 1) * W + x]] = 1; }
  for (let y = 0; y < H; y++) { touches[lab[y * W]] = 1; touches[lab[y * W + W - 1]] = 1; }
  for (let i = 0; i < o.length; i++) { const l = lab[i]; if (l && !touches[l] && sizes[l] < maxArea) o[i] = 1; }
  return o;
}

// 最小コスト経路（8 近傍・幾何的コスト）。window = [x0, y0, x1, y1]
function cheapestPath(cost, W, start, goal, win) {
  const [x0, y0, x1, y1] = win, w = x1 - x0, h = y1 - y0, N = w * h;
  const dist = new Float64Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1);
  const heapK = [], heapV = [];
  const push = (k, v) => { heapK.push(k); heapV.push(v); let i = heapK.length - 1;
    while (i) { const p = (i - 1) >> 1; if (heapK[p] <= heapK[i]) break; [heapK[p], heapK[i]] = [heapK[i], heapK[p]]; [heapV[p], heapV[i]] = [heapV[i], heapV[p]]; i = p; } };
  const pop = () => { const v = heapV[0], k = heapK[0], lk = heapK.pop(), lv = heapV.pop();
    if (heapK.length) { heapK[0] = lk; heapV[0] = lv; let i = 0;
      for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < heapK.length && heapK[l] < heapK[m]) m = l; if (r < heapK.length && heapK[r] < heapK[m]) m = r;
        if (m === i) break; [heapK[m], heapK[i]] = [heapK[i], heapK[m]]; [heapV[m], heapV[i]] = [heapV[i], heapV[m]]; i = m; } }
    return [k, v]; };
  const L = (x, y) => (y - y0) * w + (x - x0), C = (x, y) => cost[y * W + x];
  const s = L(start[0], start[1]), g = L(goal[0], goal[1]);
  dist[s] = 0; push(0, s);
  while (heapK.length) {
    const [k, p] = pop();
    if (k > dist[p]) continue;
    if (p === g) break;
    const px = p % w + x0, py = Math.floor(p / w) + y0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const x = px + dx, y = py + dy;
      if (x < x0 || y < y0 || x >= x1 || y >= y1) continue;
      const q = L(x, y), nd = k + Math.hypot(dx, dy) * (C(px, py) + C(x, y)) / 2;
      if (nd < dist[q]) { dist[q] = nd; prev[q] = p; push(nd, q); }
    }
  }
  const path = [];
  for (let p = g; p !== -1; p = prev[p]) path.push([p % w + x0, Math.floor(p / w) + y0]);
  return path;
}

// ───────────────────────────── 入力 → トーン ─────────────────────────────
// シード + 花番号（0〜8）。onProgress(0..1)
function toneFromSeed(seed, idx, P = {}, onProgress) {
  const q = FC().flowerParams(seed, idx);
  if (!q) throw new Error("この花はシードから再現できませんでした");
  return toneFromParams(q, P, onProgress);
}
// 写像のパラメータ {n, lam, alpha, beta, gamma, omega} から直接（編集した花も作れる）
function toneFromParams(q, P = {}, onProgress) {
  P = { ...DEFAULTS, ...P };
  const F = FC(), a = F.analyze(q);
  if (!a) throw new Error("この花は軌道が発散するか 1 点に収束するため、形になりません。パラメータを少し戻してください。");
  if (a.D < 1.2 || a.fill < 0.03)
    throw new Error("この花は軌道が少数の点を回るだけ（周期軌道）で、つながった形になりません。パラメータを少し戻してください。");
  const hs = F.histogram(q, P.res, P.points, onProgress);
  if (!hs) throw new Error("軌道が発散しました");
  // ページ 1px あたりの点数に換算 → 基準カーブ（DENSITY_REF）を掛ける
  const C = hs.C, cBrowser = pageCell(), k = (F.PAGE_POINTS / hs.total) * (C / cBrowser) ** 2 * DENSITY_REF;
  const d = new Float32Array(C * C);
  for (let i = 0; i < d.length; i++) d[i] = hs.h[i] * k;
  return { kind: "density", density: d, W: C, H: C, n: q.n, params: q, gamma: P.gamma };
}

// 密度場 → トーン（ブラーは mm 単位なので ppm が決まってから）
// 正規化の基準は「0.35mm でぼかした密度の最大値」。解像度（シード／PNG）に左右されにくい。
// NORM_K は基準のコースター（シード 011291810887 #3）で元の見た目と一致させる係数
const NORM_BLUR_MM = 0.35, NORM_K = Math.log1p(10.3) / Math.log1p(4.12);
function densityToTone(d, W, H, sigmaPx, gamma, ppm) {
  const b = gaussianBlur(d, W, H, sigmaPx), nb = gaussianBlur(d, W, H, NORM_BLUR_MM * ppm);
  let mx = 0; for (let i = 0; i < nb.length; i++) if (nb[i] > mx) mx = nb[i];
  const lm = NORM_K * Math.log1p(mx), t = new Float32Array(b.length);
  for (let i = 0; i < b.length; i++) t[i] = Math.min(1, Math.pow(Math.log1p(b[i]) / lm, gamma));
  return t;
}

// PNG（RGBA）→ トーン。rect = [x, y, w, h] は使う花の範囲（3×3 なら 1 マス）
function toneFromImage(rgba, imgW, imgH, rect) {
  const F = FC(), [rx, ry, rw, rh] = rect;
  const px = (x, y) => { const o = ((ry + y) * imgW + rx + x) * 4; return [rgba[o], rgba[o + 1], rgba[o + 2]]; };
  // 背景色 = 枠の最頻色
  const counts = new Map();
  for (let i = 0; i < rw; i++) for (const y of [0, 1, rh - 2, rh - 1]) { const c = px(i, y), k = c[0] << 16 | c[1] << 8 | c[2]; counts.set(k, (counts.get(k) || 0) + 1); }
  for (let j = 0; j < rh; j++) for (const x of [0, 1, rw - 2, rw - 1]) { const c = px(x, j), k = c[0] << 16 | c[1] << 8 | c[2]; counts.set(k, (counts.get(k) || 0) + 1); }
  const bgk = [...counts].sort((a, b) => b[1] - a[1])[0][0], bg = [bgk >> 16 & 255, bgk >> 8 & 255, bgk & 255];
  const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  // 色の系統と色相をサンプルで当てる（ページの paint と同じ合成式）
  const samples = [];
  for (let j = 0, s = 0; j < rh; j += 3) for (let i = (j * 7) % 5; i < rw; i += 5) { const c = px(i, j); if (dist2(c, bg) > 60) samples.push(c); }
  const stride = Math.max(1, Math.floor(samples.length / 900)), S = samples.filter((_, i) => i % stride === 0);
  const table = (LUT, step) => { const T = []; for (let t = 0; t < 256; t += step) { const v = t / 255, al = Math.min(1, v * 2.5);
    T.push([bg[0] + (LUT[t * 3] - bg[0]) * al, bg[1] + (LUT[t * 3 + 1] - bg[1]) * al, bg[2] + (LUT[t * 3 + 2] - bg[2]) * al, v]); } return T; };
  const fitErr = T => { let e = 0; for (const c of S) { let m = Infinity; for (const t of T) { const d = (c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 + (c[2] - t[2]) ** 2; if (d < m) m = d; } e += m; } return e / Math.max(1, S.length); };
  let best = { err: Infinity };
  for (const key of Object.keys(F.PALETTES)) {
    const stops = F.PALETTES[key][1]; if (!stops) continue;
    for (let hue = -180; hue < 180; hue += 4) {
      const err = fitErr(table(F.paletteLUT(stops, hue), 4));
      if (err < best.err) best = { err, key, hue };
    }
  }
  for (let hue = best.hue - 3; hue <= best.hue + 3; hue++) {            // 細かく詰める
    const err = fitErr(table(F.paletteLUT(F.PALETTES[best.key][1], hue), 4));
    if (err < best.err) best = { ...best, err, hue };
  }
  const tone = new Float32Array(rw * rh);
  let method;
  if (best.err < 120) {
    method = `${F.PALETTES[best.key][0]}（色相 ${best.hue}°）`;
    const T = table(F.paletteLUT(F.PALETTES[best.key][1], best.hue), 1), cache = new Map();
    for (let j = 0; j < rh; j++) for (let i = 0; i < rw; i++) {
      const c = px(i, j), k = c[0] << 16 | c[1] << 8 | c[2];
      let v = cache.get(k);
      if (v === undefined) { let m = Infinity; v = 0; for (const t of T) { const d = (c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 + (c[2] - t[2]) ** 2; if (d < m) { m = d; v = t[3]; } } cache.set(k, v); }
      tone[j * rw + i] = v;
    }
  } else {                                                              // カスタム色など：背景との差で代用
    method = "背景との明るさの差（色の系統を特定できず）";
    const lum = c => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2], lb = lum(bg);
    let mx = 1;
    for (let j = 0; j < rh; j++) for (let i = 0; i < rw; i++) { const v = Math.abs(lum(px(i, j)) - lb); tone[j * rw + i] = v; if (v > mx) mx = v; }
    for (let i = 0; i < tone.length; i++) tone[i] /= mx;
  }
  if (best.err >= 120) return { kind: "tone", tone, W: rw, H: rh, method, bg };
  // 画面トーン v = (log1p(d)/log1p(max))^γ を点の数 d（ページ 1px あたり）に戻す。
  // 1 点だけ当たった画素が最小の明るさ → max を推定（γ はページ既定の 0.8 とみなす）
  const g = 0.8, nz = [];
  for (const v of tone) if (v > 0.004) nz.push(v);
  nz.sort((a, b) => a - b);
  const vmin = nz.length ? nz[Math.floor(nz.length * 0.002)] : 0.1;
  const lmax = Math.log(2) / Math.pow(vmin, 1 / g);
  const d = Float32Array.from(tone, v => v > 0 ? Math.expm1(Math.pow(v, 1 / g) * lmax) * DENSITY_REF : 0);
  // 3 倍に拡大（単位はページ 1px あたりのまま）
  const up = resample(d, rw, rh, 3);
  return { kind: "density", density: up.img, W: up.W, H: up.H, gamma: g, minBlur: 0.35, method, bg, estMax: Math.expm1(lmax) };
}

// n 回対称の推定（回転して重なる度合い）
function sourceTone(src) {                      // 表示・対称推定用のおおまかなトーン
  if (src.kind !== "density") return src.tone;
  let mx = 0; for (const v of src.density) if (v > mx) mx = v;
  const lm = Math.log1p(mx); return Float32Array.from(src.density, v => Math.pow(Math.log1p(v) / lm, src.gamma));
}
function detectSymmetry(src) {
  const tone = sourceTone(src), W = src.W, H = src.H;
  const cx = W / 2, cy = H / 2, pts = [];
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) { const v = tone[y * W + x]; if (v > 0.05) pts.push([x - cx, y - cy, v]); }
  const score = n => { const a = 2 * Math.PI / n, c = Math.cos(a), s = Math.sin(a); let num = 0, den = 0;
    for (const [x, y, v] of pts) { num += v * bilinear(tone, W, H, cx + c * x - s * y, cy + s * x + c * y); den += v * v; } return num / den; };
  const sc = []; for (let n = 2; n <= 12; n++) sc.push([n, score(n)]);
  const mx = Math.max(...sc.map(s => s[1]));
  return sc.filter(s => s[1] >= 0.9 * mx).pop()[0];
}

// ───────────────────────────── トーン → 構造 ─────────────────────────────
// src: toneFromSeed / toneFromImage の戻り値。P: DEFAULTS の上書き
function buildStructure(src, P = {}, n = 0) {
  P = { ...DEFAULTS, ...P };
  let { W, H } = src, tone;
  // 1. 外径から ppm（px/mm）を決める。太らせ分 2×grow を差し引く
  const extent = t => { const { lab, sizes } = label(Uint8Array.from(t, v => v > P.threshold), W, H, true);
    const keep = new Uint8Array(sizes.length), big = sizes[largestId(sizes)];
    for (let i = 1; i < sizes.length; i++) keep[i] = sizes[i] >= big * P.majorFrac;
    let r = 0; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (keep[lab[y * W + x]]) r = Math.max(r, Math.hypot(x + 0.5 - W / 2, y + 0.5 - H / 2));
    return r; };
  let ppm;
  if (src.kind === "density") {
    // 外径の見積もりは仮の ppm（R 枠から）でトーンを作って行い、その後に正式な ppm で作り直す
    const ppm0 = 2 * (W / 2 - W / 90) / 1.02 / P.diameter;
    const t0 = densityToTone(src.density, W, H, 0, src.gamma, ppm0);
    ppm = 2 * extent(t0) / (P.diameter - 2 * P.grow);
    tone = densityToTone(src.density, W, H, Math.max(P.blur, src.minBlur || 0) * ppm, src.gamma, ppm);
  } else {
    // PNG は解像度が低い（約 5px/mm）ので 3 倍に拡大してから扱う
    const f = 3, up = resample(src.tone, W, H, f); W = up.W; H = up.H;
    tone = gaussianBlur(up.img, W, H, 1.0);
    ppm = 2 * extent(tone) / (P.diameter - 2 * P.grow);
  }
  const px = mm => mm * ppm;
  // 2. しきい値 → 細線を太らせる
  let m = dilate(Uint8Array.from(tone, v => v > P.threshold), W, H, px(P.grow));
  let { lab, sizes } = label(m, W, H, false);
  const main = largestId(sizes), majors = [];
  for (let i = 1; i < sizes.length; i++) if (i !== main && sizes[i] >= sizes[main] * P.majorFrac) majors.push(i);
  const keep = new Uint8Array(sizes.length); keep[main] = 1; majors.forEach(i => keep[i] = 1);
  m = Uint8Array.from(lab, l => keep[l]);
  // 3. ブリッジ：本体から最も近い島へ、トーンの明るい筋をたどる経路を引き、n 回対称に複製
  const bridges = new Uint8Array(W * H), cost = Float32Array.from(tone, v => 1 / (v + 0.02));
  const sym = P.symmetric && n >= 2 ? n : 1, cx = W / 2, cy = H / 2;
  let nBridges = 0;
  if (P.bridges) for (let iter = 0; iter < 12; iter++) {
    ({ lab, sizes } = label(m, W, H, false));
    const mainL = largestId(sizes);
    if (sizes.filter((s, i) => i && s >= sizes[mainL] * P.majorFrac).length <= 1) break;
    const { d, src: near } = edt(Uint8Array.from(lab, l => l === mainL), W, H);
    let bi = -1, bd = Infinity;
    for (let i = 0; i < lab.length; i++) { const l = lab[i]; if (l && l !== mainL && sizes[l] >= sizes[mainL] * P.majorFrac && d[i] < bd) { bd = d[i]; bi = i; } }
    if (bi < 0) break;
    const s = [bi % W, Math.floor(bi / W)], g = [near[bi] % W, Math.floor(near[bi] / W)], pad = Math.ceil(Math.sqrt(bd) + px(4));
    const win = [Math.max(0, Math.min(s[0], g[0]) - pad), Math.max(0, Math.min(s[1], g[1]) - pad),
                 Math.min(W, Math.max(s[0], g[0]) + pad + 1), Math.min(H, Math.max(s[1], g[1]) + pad + 1)];
    const path = cheapestPath(cost, W, s, g, win), line = new Uint8Array(W * H);
    for (let k = 0; k < sym; k++) {
      const a = 2 * Math.PI * k / sym, c = Math.cos(a), sn = Math.sin(a);
      for (const [x, y] of path) {
        const X = Math.round(cx + c * (x + 0.5 - cx) - sn * (y + 0.5 - cy) - 0.5), Y = Math.round(cy + sn * (x + 0.5 - cx) + c * (y + 0.5 - cy) - 0.5);
        if (X >= 0 && Y >= 0 && X < W && Y < H) line[Y * W + X] = 1;
      }
    }
    const b = dilate(line, W, H, px(P.bridgeWidth / 2));
    for (let i = 0; i < b.length; i++) if (b[i]) { bridges[i] = 1; m[i] = 1; }
    nBridges += sym;
  }
  // 4. 印刷できる形に整える
  m = closing(m, W, H, px(P.close));
  m = fillSmallHoles(m, W, H, P.minHole * ppm * ppm);
  ({ lab, sizes } = label(m, W, H, false));
  const mainF = largestId(sizes), pieces = sizes.length - 1;
  m = Uint8Array.from(lab, l => l === mainF);
  let area = 0, barea = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) { area++; if (bridges[i]) barea++; }
  if (area < (P.diameter * ppm) ** 2 * 0.01)
    throw new Error("構造がほとんど残りませんでした。しきい値を下げるか、線の太さを上げてください。");
  return { mask: m, bridges, tone, W, H, ppm, n, nBridges, pieces, bridgeFrac: barea / area, P };
}


// ───────────────────────────── 構造 → なめらかな輪郭 → 平らなメッシュ（FDM 向け） ─────────────────────────────
// マスクをぼかした場を 0.5 で等高線にし（marching squares）、Douglas–Peucker で間引いた
// 閉曲線（外周は反時計回り・穴は時計回り）を earcut で三角形分割して、厚み base で押し出す。
function simplifyRing(pts, tol) {               // 閉曲線の Douglas–Peucker
  const n = pts.length; if (n < 8) return pts;
  let far = 0, fd = -1;
  for (let i = 1; i < n; i++) { const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2; if (d > fd) { fd = d; far = i; } }
  const keep = new Uint8Array(n); keep[0] = keep[far] = 1;
  const stack = [[0, far], [far, n]], t2 = tol * tol;
  while (stack.length) {
    const [a, b] = stack.pop(); if (b - a < 2) continue;
    const A = pts[a], B = pts[b % n], dx = B[0] - A[0], dy = B[1] - A[1], L2 = dx * dx + dy * dy || 1e-12;
    let mi = -1, md = -1;
    for (let i = a + 1; i < b; i++) {
      const P = pts[i], t = Math.max(0, Math.min(1, ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / L2));
      const d = (P[0] - A[0] - t * dx) ** 2 + (P[1] - A[1] - t * dy) ** 2;
      if (d > md) { md = d; mi = i; }
    }
    if (md > t2) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
  }
  const out = []; for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[i]);
  return out;
}
const ringArea = r => { let a = 0; for (let i = 0, j = r.length - 1; i < r.length; j = i++) a += (r[j][0] - r[i][0]) * (r[j][1] + r[i][1]); return a / 2; };
function pointInRing(p, r) {
  let inside = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++)
    if ((r[i][1] > p[1]) !== (r[j][1] > p[1]) && p[0] < (r[j][0] - r[i][0]) * (p[1] - r[i][1]) / (r[j][1] - r[i][1]) + r[i][0]) inside = !inside;
  return inside;
}

function buildMeshFlat(st) {
  const { mask, W, H, ppm, P } = st, cx = W / 2, cy = H / 2;
  if (typeof root.earcut !== "function") throw new Error("earcut が読み込まれていません");
  let rmax = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) rmax = Math.max(rmax, Math.hypot(x + 0.5 - cx, y + 0.5 - cy));
  const c = P.cell, R = rmax / ppm + 1 + 3 * P.smooth, G1 = Math.ceil(2 * R / c) + 1;   // 格子点 G1 × G1
  const col = xmm => cx + xmm * ppm - 0.5, row = ymm => cy - ymm * ppm - 0.5;
  const mf = Float32Array.from(mask);
  let f = new Float32Array(G1 * G1);
  for (let i = 0; i < G1; i++) for (let j = 0; j < G1; j++) f[i * G1 + j] = bilinear(mf, W, H, col(j * c - R), row(i * c - R));
  f = gaussianBlur(f, G1, G1, P.smooth / c);
  for (let k = 0; k < G1; k++) f[k] = f[(G1 - 1) * G1 + k] = f[k * G1] = f[k * G1 + G1 - 1] = 0;   // 外周は必ず外側
  // ぼかしで切れた小片は捨てて 1 ピースに
  { const { lab, sizes } = label(Uint8Array.from(f, v => v > 0.5), G1, G1, false), b = largestId(sizes);
    for (let k = 0; k < f.length; k++) if (lab[k] !== b) f[k] = Math.min(f[k], 0.49); }

  // marching squares（内側を左に見る向き）。辺 id: 横 = 2·node, 縦 = 2·node + 1
  const inside = k => f[k] > 0.5, node = (i, j) => i * G1 + j;
  const edgePt = new Map(), next = new Map();
  const pt = (id) => {
    let p = edgePt.get(id); if (p) return p;
    const nd = id >> 1, i = Math.floor(nd / G1), j = nd % G1, a = f[nd], b = (id & 1) ? f[nd + G1] : f[nd + 1];
    const t = (0.5 - a) / (b - a);
    p = (id & 1) ? [j * c - R, (i + t) * c - R] : [(j + t) * c - R, i * c - R];
    edgePt.set(id, p); return p;
  };
  for (let i = 0; i < G1 - 1; i++) for (let j = 0; j < G1 - 1; j++) {
    const n00 = node(i, j), n01 = n00 + 1, n11 = n00 + G1 + 1, n10 = n00 + G1;
    const C = [n00, n01, n11, n10], ins = C.map(inside);
    if (ins[0] === ins[1] && ins[1] === ins[2] && ins[2] === ins[3]) continue;
    const E = [2 * n00, 2 * n01 + 1, 2 * n10, 2 * n00 + 1];   // 下・右・上・左（反時計回り）
    const cross = [];                                        // [辺 k, exit?]
    for (let k = 0; k < 4; k++) if (ins[k] !== ins[(k + 1) % 4]) cross.push([k, ins[k]]);
    const connected = cross.length === 4 && (f[n00] + f[n01] + f[n11] + f[n10]) / 4 > 0.5;
    for (let q = 0; q < cross.length; q++) {
      if (!cross[q][1]) continue;                            // exit から entry へ線を引く
      // 鞍点：中心が内側なら内側どうしがつながる → 次の entry、そうでなければ前の entry と組む
      const m = cross.length, partner = connected ? cross[(q + 1) % m] : cross[(q - 1 + m) % m];
      const pk = cross.length === 2 ? cross[(q + 1) % 2] : partner;
      next.set(E[cross[q][0]], E[pk[0]]);
    }
  }
  const rings = [], used = new Set();
  for (const start of next.keys()) {
    if (used.has(start)) continue;
    const r = []; let e = start;
    while (!used.has(e)) { used.add(e); r.push(pt(e)); e = next.get(e); if (e === undefined) break; }
    if (r.length >= 3) rings.push(r);
  }
  // 間引き・小さすぎる輪の除去・外周と穴の対応づけ
  const tol = Math.min(0.02, c * 0.2);
  const outers = [], holes = [];
  for (let r of rings) {
    r = simplifyRing(r, tol);
    if (r.length < 3) continue;
    const a = ringArea(r);
    if (a > 0.5) outers.push({ r, a, holes: [] });
    else if (a < -0.05) holes.push(r);
  }
  if (!outers.length) throw new Error("輪郭を作れませんでした。線の太さを上げてください。");
  outers.sort((x, y) => y.a - x.a);
  for (const h of holes) { const o = outers.find(o => pointInRing(h[0], o.r)) || outers[0]; o.holes.push(h); }

  // 頂点（上面 0..N-1、底面 N..2N-1）と三角形
  const verts = [], tris = [];
  for (const o of outers) {
    const loops = [o.r, ...o.holes], flat = [], holeIdx = [], base = verts.length / 2;
    for (const L of loops) { if (L !== o.r) holeIdx.push(flat.length / 2); for (const p of L) flat.push(p[0], p[1]); }
    for (let k = 0; k < flat.length; k++) verts.push(flat[k]);
    const T = root.earcut(flat, holeIdx, 2);
    for (let t = 0; t < T.length; t += 3) {
      let [a, b, d] = [T[t], T[t + 1], T[t + 2]];
      const ar = (flat[2 * b] - flat[2 * a]) * (flat[2 * d + 1] - flat[2 * a + 1]) - (flat[2 * d] - flat[2 * a]) * (flat[2 * b + 1] - flat[2 * a + 1]);
      if (ar < 0) [b, d] = [d, b];
      tris.push(['t', base + a, base + b, base + d]);
    }
    let off = base;
    for (const L of loops) {                                // 壁：辺 a→b（内側が左）の右側が外向き
      for (let k = 0; k < L.length; k++) tris.push(['w', off + k, off + (k + 1) % L.length]);
      off += L.length;
    }
  }
  const N = verts.length / 2, Z = P.base, idx = [];
  for (const t of tris) {
    if (t[0] === 't') { idx.push(t[1], t[2], t[3]); idx.push(t[1] + N, t[3] + N, t[2] + N); }
    else { const a = t[1], b = t[2]; idx.push(a + N, b + N, b, a + N, b, a); }
  }
  const nT = idx.length / 3, pos = new Float32Array(nT * 9);
  const vxyz = v => v < N ? [verts[2 * v], verts[2 * v + 1], Z] : [verts[2 * (v - N)], verts[2 * (v - N) + 1], 0];
  let vol = 0;
  for (let t = 0; t < nT; t++) {
    const p = [vxyz(idx[3 * t]), vxyz(idx[3 * t + 1]), vxyz(idx[3 * t + 2])];
    for (let k = 0; k < 3; k++) pos.set(p[k], t * 9 + k * 3);
    const [A, B, D] = p;
    vol += (A[0] * (B[1] * D[2] - B[2] * D[1]) - A[1] * (B[0] * D[2] - B[2] * D[0]) + A[2] * (B[0] * D[1] - B[1] * D[0])) / 6;
  }
  let area = 0, dia = 0;
  for (const o of outers) { area += o.a; for (const h of o.holes) area += ringArea(h); for (const p of o.r) dia = Math.max(dia, 2 * Math.hypot(p[0], p[1])); }
  return { positions: pos, indices: Uint32Array.from(idx), nVerts: 2 * N, triangles: nT, volume: vol, area,
           diameter: dia, height: Z, rings: { outers: outers.length, holes: holes.length, points: N }, flat: true };
}

// ───────────────────────────── 構造 → メッシュ / STL ─────────────────────────────
function buildMesh(st) {
  if (st.P.flat) {
    const m = buildMeshFlat(st);
    if (checkWatertight(m.indices).ok) return m;
    // 三角形分割が閉じなかったとき（まれ）は格子のメッシュで作り直す
    st = { ...st, P: { ...st.P, relief: 0 } };
    const g = buildMeshGrid(st); g.fallback = true; return g;
  }
  return buildMeshGrid(st);
}
function buildMeshGrid(st) {
  const { mask, bridges, tone, W, H, ppm, P } = st, cx = W / 2, cy = H / 2;
  // 構造の外接半径から格子の範囲を決める
  let rmax = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) rmax = Math.max(rmax, Math.hypot(x + 0.5 - cx, y + 0.5 - cy));
  const R = rmax / ppm + 1, CELL = P.cell, G = Math.ceil(2 * R / CELL), G1 = G + 1;
  const col = xmm => cx + xmm * ppm - 0.5, row = ymm => cy - ymm * ppm - 0.5;
  const mf = Float32Array.from(mask), bf = Float32Array.from(bridges);
  let M = new Uint8Array(G * G);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const xm = (j + 0.5) * CELL - R, ym = (i + 0.5) * CELL - R;
    M[i * G + j] = bilinear(mf, W, H, col(xm), row(ym)) > 0.5;
  }
  // 最大の 8 連結だけ残し、斜めだけで接する角（非多様体）を埋める
  { const { lab, sizes } = label(M, G, G, true), b = largestId(sizes); M = Uint8Array.from(lab, l => l === b); }
  for (let it = 0; it < 2000; it++) {
    let changed = false;
    for (let i = 0; i < G - 1; i++) for (let j = 0; j < G - 1; j++) {
      const a = M[i * G + j], b = M[i * G + j + 1], c = M[(i + 1) * G + j], d = M[(i + 1) * G + j + 1];
      if (a && d && !b && !c) { M[i * G + j + 1] = 1; changed = true; }
      else if (b && c && !a && !d) { M[i * G + j] = 1; changed = true; }
    }
    if (!changed) break;
  }
  // 角の高さ
  const Z = new Float32Array(G1 * G1);
  for (let i = 0; i < G1; i++) for (let j = 0; j < G1; j++) {
    const xm = j * CELL - R, ym = i * CELL - R;
    const v = Math.max(bilinear(tone, W, H, col(xm), row(ym)), 0.5 * bilinear(bf, W, H, col(xm), row(ym)));
    Z[i * G1 + j] = P.base + P.relief * Math.min(1, Math.max(0, v));
  }
  const NV = G1 * G1, vx = k => (k % G1) * CELL - R, vy = k => Math.floor(k / G1) * CELL - R, vz = k => k < NV ? Z[k] : 0;
  const tris = [];
  const has = (i, j) => i >= 0 && j >= 0 && i < G && j < G && M[i * G + j];
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    if (!M[i * G + j]) continue;
    const c00 = i * G1 + j, c01 = c00 + 1, c11 = c00 + G1 + 1, c10 = c00 + G1;
    tris.push(c00, c01, c11, c00, c11, c10, c00 + NV, c11 + NV, c01 + NV, c00 + NV, c10 + NV, c11 + NV);
    const wall = (a, b) => tris.push(b, a, a + NV, b, a + NV, b + NV);
    if (!has(i - 1, j)) wall(c00, c01);
    if (!has(i, j + 1)) wall(c01, c11);
    if (!has(i + 1, j)) wall(c11, c10);
    if (!has(i, j - 1)) wall(c10, c00);
  }
  const nT = tris.length / 3, pos = new Float32Array(nT * 9);
  let vol = 0;
  for (let t = 0; t < nT; t++) {
    const p = [0, 1, 2].map(k => { const v = tris[t * 3 + k]; return [vx(v % NV + 0), vy(v % NV), vz(v)]; });
    for (let k = 0; k < 3; k++) pos.set(p[k], t * 9 + k * 3);
    const [a, b, c] = p;
    vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  let area = 0; for (let i = 0; i < M.length; i++) area += M[i];
  let dia = 0;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) if (M[i * G + j]) dia = Math.max(dia, 2 * Math.hypot((j + 0.5) * CELL - R, (i + 0.5) * CELL - R) + CELL);
  return { positions: pos, indices: Uint32Array.from(tris), nVerts: 2 * NV, triangles: nT, volume: vol, area: area * CELL * CELL,
           diameter: dia, height: P.base + P.relief, grid: { M, G, R, CELL } };
}

function toSTL(positions, name = "order-chaos-stl") {
  const nT = positions.length / 9, buf = new ArrayBuffer(84 + nT * 50), dv = new DataView(buf);
  const hdr = new TextEncoder().encode(name.slice(0, 80));
  new Uint8Array(buf, 0, 80).set(hdr);
  dv.setUint32(80, nT, true);
  for (let t = 0, o = 84; t < nT; t++, o += 50) {
    const p = positions.subarray(t * 9, t * 9 + 9);
    const ux = p[3] - p[0], uy = p[4] - p[1], uz = p[5] - p[2], wx = p[6] - p[0], wy = p[7] - p[1], wz = p[8] - p[2];
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx; const l = Math.hypot(nx, ny, nz) || 1;
    dv.setFloat32(o, nx / l, true); dv.setFloat32(o + 4, ny / l, true); dv.setFloat32(o + 8, nz / l, true);
    for (let k = 0; k < 9; k++) dv.setFloat32(o + 12 + k * 4, p[k], true);
  }
  return buf;
}

// すべての辺がちょうど 2 回・逆向きに使われているか（テスト用）
function checkWatertight(indices) {
  const seen = new Set();
  for (let t = 0; t < indices.length; t += 3) for (let k = 0; k < 3; k++) {
    const a = indices[t + k], b = indices[t + (k + 1) % 3], key = a * 4294967296 + b;
    if (seen.has(key)) return { ok: false, reason: "同じ向きの辺が重複（非多様体）" };
    seen.add(key);
  }
  for (const key of seen) { const a = Math.floor(key / 4294967296), b = key % 4294967296; if (!seen.has(b * 4294967296 + a)) return { ok: false, reason: "開いた辺" }; }
  return { ok: true };
}

root.CoasterPipeline = { DEFAULTS, toneFromSeed, toneFromParams, toneFromImage, detectSymmetry, sourceTone, buildStructure, buildMesh, toSTL, checkWatertight,
                         _internal: { edt, label, dilate, gaussianBlur, cheapestPath, resample } };
})(typeof globalThis !== "undefined" ? globalThis : this);
