/* flower-core.js — Order&Chaos 2D ver.2 と同じ写像・シード乱数・パレット
 * 出典: github.com/York-14/Generative-flower-Order-Chaos (docs/2d-v2.html) からそのまま移植。
 * ブラウザ（window / Worker）と Node の両方で使える。 */
(function (root) {
"use strict";
  const SEED_SPACE = 1e12;
  const pad12 = s => String(s).padStart(12, "0");
  function splitmix32(a) {
    return () => {
      a = a + 0x9E3779B9 | 0;
      let z = a;
      z = Math.imul(z ^ z >>> 16, 0x85EBCA6B); z = Math.imul(z ^ z >>> 13, 0xC2B2AE35);
      return (z ^ z >>> 16) >>> 0;
    };
  }
  // (hi, lo, k) → 4 ワード。k を固定すると (hi, lo) に対して単射（各 splitmix32 の初回出力が種に対して全単射のため）
  function subSeed(hi, lo, k) {
    const a = splitmix32(lo), b = splitmix32(Math.imul(hi, 0x9E3779B1) ^ 0x6A09E667),
          c = splitmix32(Math.imul(k + 1, 0x85EBCA77) ^ 0xBB67AE85);
    const a0 = a(), b0 = b(), c0 = c();
    return [a0 ^ c0, b0 ^ a(), c0 ^ b(), a() ^ b() ^ c()].map(x => x >>> 0);
  }
  function sfc32(a, b, c, d) {
    const next = () => {
      const t = (a + b | 0) + d | 0;
      d = d + 1 | 0; a = b ^ b >>> 9; b = c + (c << 3) | 0; c = (c << 21 | c >>> 11) + t | 0;
      return (t >>> 0) / 4294967296;
    };
    for (let i = 0; i < 12; i++) next();       // 初期状態の偏りを捨てる
    return next;
  }
  const seedRng = (s, k) => sfc32(...subSeed(Math.floor(s / 4294967296), s % 4294967296, k));
  // 64bit を棄却サンプリングして 0〜10^12−1 の一様乱数にする
  function newSeed() {
    const N = 1000000000000n, LIMIT = (1n << 64n) - (1n << 64n) % N, w = new Uint32Array(2);
    for (;;) {
      crypto.getRandomValues(w);
      const v = BigInt(w[0]) << 32n | BigInt(w[1]);
      if (v < LIMIT) return Number(v % N);
    }
  }
  const freshRng = () => seedRng(newSeed(), 0);   // 再抽選・変奏・色の抽選など、シードに紐づかない乱数
  // 1〜12 桁の数字はそのまま、13 桁以上はエラー、それ以外の文字列は 64bit FNV-1a で 10^12 に収める
  function parseSeed(str) {
    str = String(str ?? "").trim();
    if (!str) return null;
    if (/^\d+$/.test(str)) {
      if (str.length > 12) throw new RangeError("シードは 12 桁まで（000000000000〜999999999999）です");
      return Number(str);
    }
    let h = 0xCBF29CE484222325n;
    for (const b of new TextEncoder().encode(str)) h = (h ^ BigInt(b)) * 0x100000001B3n & 0xFFFFFFFFFFFFFFFFn;
    return Number(h % 1000000000000n);
  }
  const dateSeed = d => Number(`${d.getFullYear()}${[d.getMonth() + 1, d.getDate(), d.getHours(), d.getMinutes()]
    .map(v => String(v).padStart(2, "0")).join("")}`);
  const rnd = (rng, lo, hi) => lo + rng() * (hi - lo);
  const pick = (rng, a) => a[Math.floor(rng() * a.length)];
  const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let NX = 0, NY = 0;
  function step(q, x, y) {
    const zz = x * x + y * y;
    let zr = x, zi = y;                         // z^(n-1)
    for (let i = 1; i < q.n - 1; i++) { const t = zr * x - zi * y; zi = zr * y + zi * x; zr = t; }
    const ren = zr * x - zi * y;                // Re(z^n)
    const p = q.lam + q.alpha * zz + q.beta * ren;
    NX = p * x - q.omega * y + q.gamma * zr;
    NY = p * y + q.omega * x - q.gamma * zi;
  }

  // 過渡を捨てて境界半径を推定。発散・固定点なら null
  function probe(q, N = 20000) {
    let x = 0.01, y = 0.003, R = 0;
    for (let i = 0; i < N; i++) {
      step(q, x, y); x = NX; y = NY;
      if (!(Math.abs(x) + Math.abs(y) < 1e4)) return null;
      if (i > 1000) { const r = x * x + y * y; if (r > R) R = r; }
    }
    R = Math.sqrt(R);
    return R > 1e-6 ? { R, x, y } : null;
  }

  // 軌道を 128×128 に落としてボックス次元・余白・Lyapunov を測る
  function analyze(q) {
    const pr = probe(q);
    if (!pr) return null;
    const G = 128, h = new Uint8Array(G * G);
    let x = pr.x, y = pr.y;
    for (let i = 0; i < 80000; i++) {
      step(q, x, y); x = NX; y = NY;
      const gx = ((x / pr.R) * 0.5 + 0.5) * (G - 1) | 0, gy = ((y / pr.R) * 0.5 + 0.5) * (G - 1) | 0;
      if (gx >= 0 && gx < G && gy >= 0 && gy < G) h[gy * G + gx] = 1;
    }
    const occ32 = new Uint8Array(32 * 32), c = (G - 1) / 2;
    let c128 = 0, c32 = 0, occDisk = 0, disk = 0;
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const dx = i - c, dy = j - c, inD = dx * dx + dy * dy <= (G / 2) ** 2;
      if (inD) disk++;
      if (!h[j * G + i]) continue;
      c128++; if (inD) occDisk++;
      const o = (j >> 2) * 32 + (i >> 2); if (!occ32[o]) { occ32[o] = 1; c32++; }
    }
    const d0 = 1e-8; let x2 = x + d0, y2 = y, lsum = 0;
    for (let i = 0; i < 3000; i++) {
      step(q, x, y); const ax = NX, ay = NY;
      step(q, x2, y2);
      const dx = NX - ax, dy = NY - ay, dd = Math.hypot(dx, dy) || 1e-300;
      lsum += Math.log(dd / d0);
      x = ax; y = ay; x2 = ax + dx * d0 / dd; y2 = ay + dy * d0 / dd;
    }
    return { D: Math.log2(c128 / Math.max(c32, 1)) / 2, fill: occDisk / disk, lyap: lsum / 3000 };
  }
  const isGood = a => a && a.D > 1.45 && a.D < 1.9 && a.fill > 0.15 && a.fill < 0.7 && a.lyap > 0.01;

  function randomParams(rng) {
    return { n: 3 + Math.floor(rng() * 8), lam: rnd(rng, -2.8, 2.8), alpha: rnd(rng, -5, 5), beta: rnd(rng, -1.5, 1.5),
             gamma: rnd(rng, -1.2, 1.2), omega: rng() < 0.5 ? 0 : rnd(rng, -0.3, 0.3) };
  }
  const PALETTES = {
    magma:   ["マグマ", [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]]],
    inferno: ["インフェルノ", [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]]],
    viridis: ["ビリディス", [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]],
    ocean:   ["氷・海", [[2, 6, 20], [10, 40, 90], [30, 110, 170], [120, 200, 230], [240, 252, 255]]],
    sakura:  ["桜", [[10, 4, 10], [90, 30, 70], [200, 90, 140], [250, 170, 200], [255, 240, 245]]],
    gold:    ["金箔", [[8, 5, 0], [80, 45, 5], [170, 110, 20], [235, 190, 80], [255, 250, 220]]],
    emerald: ["翡翠", [[0, 8, 6], [5, 60, 45], [20, 140, 100], [110, 220, 170], [235, 255, 245]]],
    mono:    ["モノクロ", [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]]],
    ink:     ["墨（白背景向け）", [[255, 255, 255], [180, 180, 190], [110, 110, 125], [50, 50, 60], [0, 0, 0]]],
    custom:  ["カスタム（3 色）", null],
  };

  // 9 つの花のパラメータ（ページの generate と同じ：花 i は系列 i）
  function flowerParams(seed, idx, maxTries = 1500) {
    const rng = seedRng(seed, idx);
    for (let t = 0; t < maxTries; t++) { const q = randomParams(rng); if (isGood(analyze(q))) return q; }
    return null;
  }
  // 色相回転込みの LUT（ページの buildLUT と同じ計算）
  function paletteLUT(stops, hue) {
    const LUT = new Float32Array(256 * 3);
    const a = Math.cos(hue * Math.PI / 180), b = Math.sin(hue * Math.PI / 180);
    const M = [.213 + .787 * a - .213 * b, .715 - .715 * a - .715 * b, .072 - .072 * a + .928 * b,
               .213 - .213 * a + .143 * b, .715 + .285 * a + .140 * b, .072 - .072 * a - .283 * b,
               .213 - .213 * a - .787 * b, .715 - .715 * a + .715 * b, .072 + .928 * a + .072 * b];
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * (stops.length - 1), j = Math.min(stops.length - 2, Math.floor(t)), f = t - j;
      const c = [0, 1, 2].map(k => stops[j][k] + f * (stops[j + 1][k] - stops[j][k]));
      for (let k = 0; k < 3; k++) LUT[i * 3 + k] = clip(M[k * 3] * c[0] + M[k * 3 + 1] * c[1] + M[k * 3 + 2] * c[2], 0, 255);
    }
    return LUT;
  }
  // 軌道のヒストグラム（ページの render と同じ写像・同じ枠取り）。onProgress(0..1) は任意
  function histogram(q, C, total, onProgress) {
    const pr = probe(q);
    if (!pr) return null;
    const scale = (C / 2 - C / 90) / (pr.R * 1.02), h = new Float32Array(C * C);
    let x = pr.x, y = pr.y;
    const CH = 5e6;
    for (let done = 0; done < total; done += CH) {
      const m = Math.min(CH, total - done);
      for (let i = 0; i < m; i++) {
        step(q, x, y); x = NX; y = NY;
        const px = (C / 2 + x * scale) | 0, py = (C / 2 - y * scale) | 0;
        if (px >= 0 && px < C && py >= 0 && py < C) h[py * C + px]++;
      }
      if (onProgress) onProgress(Math.min(1, (done + m) / total));
    }
    return { h, C, total, Rpx: pr.R * scale };
  }
  root.FlowerCore = { parseSeed, pad12, seedRng, step, probe, analyze, isGood, randomParams, flowerParams,
                      PALETTES, paletteLUT, histogram,
                      CANVAS: 1440, CELL_FRAC: 0.92, PAGE_POINTS: 2e6 };
})(typeof globalThis !== "undefined" ? globalThis : this);
