/* worker.js — 重い計算（軌道・画像処理・メッシュ）を UI と別スレッドで行う */
importScripts("../vendor/earcut.js", "flower-core.js", "pipeline.js");
const CP = self.CoasterPipeline;
let src = null, srcN = 0;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

function tonePreview(s, size = 360) {
  const t = CP.sourceTone(s), img = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const v = t[Math.min(s.H - 1, Math.floor((y + 0.5) * s.H / size)) * s.W + Math.min(s.W - 1, Math.floor((x + 0.5) * s.W / size))];
    const o = (y * size + x) * 4, a = Math.min(1, v * 2.5);
    img[o] = 11 + (120 * v - 11) * a; img[o + 1] = 11 + (200 * v - 11) * a; img[o + 2] = 16 + (255 * v - 16) * a; img[o + 3] = 255;
  }
  return img;
}

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.type === "seed" || m.type === "params") {
      post({ type: "status", text: "軌道を計算中… 0%" });
      const prog = p => post({ type: "status", text: `軌道を計算中… ${Math.round(p * 100)}%` });
      src = m.type === "seed" ? CP.toneFromSeed(m.seed, m.idx, { points: m.points }, prog)
                              : CP.toneFromParams(m.q, { points: m.points }, prog);
      srcN = src.n;
      const pv = tonePreview(src);
      post({ type: "source", mode: "seed", n: srcN, params: src.params, preview: pv }, [pv.buffer]);
    } else if (m.type === "image") {
      post({ type: "status", text: "色からトーンを復元中…" });
      src = CP.toneFromImage(new Uint8ClampedArray(m.rgba), m.w, m.h, m.rect);
      srcN = CP.detectSymmetry(src);
      const pv = tonePreview(src);
      post({ type: "source", mode: "image", n: srcN, method: src.method, preview: pv }, [pv.buffer]);
    } else if (m.type === "build") {
      if (!src) return;
      post({ type: "status", text: "構造を作成中…" });
      const n = m.n > 0 ? m.n : srcN;
      const st = CP.buildStructure(src, m.params, n);
      post({ type: "status", text: "メッシュを作成中…" });
      const mesh = CP.buildMesh(st), pv = CP.previewRGBA(st, mesh);
      post({ type: "result", job: m.job, preview: pv, positions: mesh.positions,
             stats: { triangles: mesh.triangles, volume: mesh.volume, area: mesh.area, diameter: mesh.diameter,
                      height: mesh.height, flat: !!mesh.flat, bridges: st.nBridges, bridgeFrac: st.bridgeFrac, n } },
           [pv.buffer, mesh.positions.buffer]);
    }
  } catch (err) {
    post({ type: "error", text: err.message || String(err) });
  }
};
