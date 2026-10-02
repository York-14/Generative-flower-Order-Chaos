"""
shape_map.py
============
symmetric_icons.py の探索結果 (catalog.csv + thumbs.npy) から
  1) 回転・鏡映不変な形状特徴を作り、2次元埋め込み（UMAP / なければ t-SNE）と
     クラスタリング（HDBSCAN）で「形の地図」を作る
  2) 美しさの評価関数  Beauty = Order^wO × Complexity^wC × Contrast^wK  を計算する
  3) 人間の一対比較データから評価関数の重みを較正する（Bradley–Terry）

使い方
  python shape_map.py --run out/run1                    # 全形の地図 + 美しさ
  python shape_map.py --run out/run1 --mode petal       # 対称次数で正規化した「花びら」地図

特徴量の2つのモード
  full  : 角度スペクトル |FFT_m| (m = 1..M) を半径ごとに → 対称次数 n も形の一部として扱う。
          地図は主に「何回対称か」で大きく分かれ、その中で形が並ぶ。
  petal : m = k, 2k, ..., Jk（k = 検出された回転次数）だけを取り出す。
          5回対称の花と 8回対称の花が「花びらの描き方」が同じなら近くに来る。
          ＝ 対称性（order）を割り算して、基本領域の形（chaos 側の個性）だけを比べる地図。
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

# --------------------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------------------

NUM_COLS = ("lam", "alpha", "beta", "gamma", "omega", "lyap", "R", "alive", "boxdim",
            "entropy", "fill", "mirror_score", "angular_contrast", "breaking_amp")


def load_run(run):
    with open(os.path.join(run, "catalog.csv")) as f:
        rows = list(csv.DictReader(f))
    thumbs = np.load(os.path.join(run, "thumbs.npy"))
    cat = {c: np.array([float(r[c]) for r in rows]) for c in NUM_COLS if c in rows[0]}
    for c in ("n", "k", "k_orbit", "id"):
        cat[c] = np.array([int(r[c]) for r in rows])
    for c in ("symmetry_broken", "degenerate_1d", "extra_mirror"):
        cat[c] = np.array([r.get(c) == "True" for r in rows])
    for c in ("group", "map_group"):
        cat[c] = np.array([r[c] for r in rows])
    return cat, thumbs

# --------------------------------------------------------------------------------------
# 形状特徴（回転・鏡映不変）
# --------------------------------------------------------------------------------------

def polar_resample(thumbs, n_r=16, n_th=256):
    """サムネイル (N,H,H)（中心=原点）を極座標 (N,n_r,n_th) に双線形補間で再標本化。"""
    N, H, _ = thumbs.shape
    c = (H - 1) / 2
    r = (np.arange(n_r) + 0.5) / n_r * (H / 2 - 0.5)
    th = np.arange(n_th) / n_th * 2 * np.pi
    yy = c + r[:, None] * np.sin(th)[None, :]
    xx = c + r[:, None] * np.cos(th)[None, :]
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    fy = yy - y0; fx = xx - x0
    y1 = np.clip(y0 + 1, 0, H - 1); x1 = np.clip(x0 + 1, 0, H - 1)
    T = thumbs.astype(np.float32) / 255.0
    return ((1 - fy) * (1 - fx) * T[:, y0, x0] + (1 - fy) * fx * T[:, y0, x1]
            + fy * (1 - fx) * T[:, y1, x0] + fy * fx * T[:, y1, x1])


def shape_features(thumbs, k=None, mode="full", n_r=16, n_th=256, M=48, J=10):
    """回転不変（|FFT| は回転で位相しか変わらない）かつ鏡映不変（実信号の |FFT| は反転で不変）。"""
    P = polar_resample(thumbs, n_r=n_r, n_th=n_th)        # (N, n_r, n_th)
    radial = P.mean(2)                                    # 半径方向プロファイル
    F = np.abs(np.fft.rfft(P, axis=2))                    # (N, n_r, n_th/2+1)
    dc = F[:, :, :1] + 1e-6
    if mode == "full":
        spec = F[:, :, 1:M + 1] / dc
    elif mode == "petal":
        assert k is not None
        idx = np.clip(np.asarray(k)[:, None] * np.arange(1, J + 1)[None, :], 0, F.shape[2] - 1)
        spec = np.take_along_axis(F, idx[:, None, :].repeat(n_r, 1), axis=2) / dc
    else:
        raise ValueError(mode)
    spec = np.log1p(10 * spec)                             # 強い調和成分の支配を緩和
    return np.concatenate([radial, spec.reshape(len(thumbs), -1)], axis=1)


def image_stats(thumbs):
    """サムネイルから「コントラスト」系の統計量。"""
    I = thumbs.astype(np.float32) / 255.0
    gy, gx = np.gradient(I, axis=(1, 2))
    g = np.sqrt(gx * gx + gy * gy)
    occ = I > 0.02
    n_occ = np.maximum(occ.sum((1, 2)), 1)
    mean_I = (I * occ).sum((1, 2)) / n_occ
    sharp = (g * occ).sum((1, 2)) / n_occ / (mean_I + 1e-6)       # フィラメント・輪郭の鋭さ
    var_I = ((I - mean_I[:, None, None]) ** 2 * occ).sum((1, 2)) / n_occ
    dyn = np.sqrt(var_I) / (mean_I + 1e-6)                        # 明暗のダイナミックレンジ
    # 円盤内の「空白（negative space）」率：外接円内でアトラクタが占めない割合
    H = I.shape[1]
    yy, xx = np.mgrid[:H, :H]
    disk = ((yy - (H - 1) / 2) ** 2 + (xx - (H - 1) / 2) ** 2) <= (H / 2) ** 2
    negative = 1 - (occ & disk).sum((1, 2)) / disk.sum()
    # 「もや」率：占有画素のうち暗い画素の割合。明るい細線＋一面の淡い塵（スポーク型）で大きい
    haze = ((I > 0.02) & (I < 0.4)).sum((1, 2)) / n_occ
    # 斑点率：4近傍の半分以上が空の孤立画素の割合（ノイズっぽさ）
    nb = sum(np.roll(occ, s, a) for s in (1, -1) for a in (1, 2))
    speckle = (occ & (nb <= 2)).sum((1, 2)) / n_occ
    return dict(sharpness=sharp, dynamic_range=dyn, negative_space=negative,
                haze=haze, speckle=speckle)

# --------------------------------------------------------------------------------------
# 美しさの評価関数
# --------------------------------------------------------------------------------------

def bell(x, mu, sigma):
    """逆U字（Wundt 曲線）：中庸で最大 1。"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigmoid(x, x0, s):
    return 1 / (1 + np.exp(-(x - x0) / s))


DEFAULT_BEAUTY = dict(
    # Order
    k_mu=math.log(6), k_sigma=0.55,      # 回転次数は 4〜10 回あたりを好む（log k 上の釣鐘）
    mirror_w=0.35,                       # 鏡映対称の加点の大きさ
    # Complexity（逆U字）
    D_mu=1.72, D_sigma=0.11,             # ボックス次元：線(1)でも塗りつぶし(2)でもない中庸
    lyap_mu=math.log(0.35), lyap_sigma=0.7,
    # Contrast
    neg_mu=0.45, neg_sigma=0.22,         # 円内の空白率：適度な余白
    haze_x0=0.5, haze_s=0.1, haze_w=0.6,     # 図と地の明瞭さ：淡いもやが多いと減点
    speckle_x0=0.2, speckle_s=0.04, speckle_w=0.7,  # 孤立点ノイズの減点
    actr_x0=0.25, actr_s=0.07,           # 角度コントラスト：模様が角度方向に読み取れるか
    sharp_x0=None, sharp_s=None,         # 鋭さ：データの分布から自動設定（中央値, IQR/2）
    # 重み（幾何平均の指数）
    wO=1.0, wC=1.0, wK=1.0,
    degenerate_penalty=0.25,             # 直線・スポーク状（1次元的）への減点
)


def beauty_components(cat, stats, p=None):
    p = dict(DEFAULT_BEAUTY, **(p or {}))
    k = np.maximum(cat["k"], 1).astype(float)
    # ---- Order: 対称の「次数」と「鏡映」と「規則性の質」
    o_k = bell(np.log(k), p["k_mu"], p["k_sigma"])
    o_m = 1 - p["mirror_w"] + p["mirror_w"] * np.clip(cat["mirror_score"], 0, 1)
    iso = cat["group"] == "O2~"                            # ほぼ円対称 = 秩序が読み取れない
    o_a = sigmoid(cat["angular_contrast"], p["actr_x0"], p["actr_s"])   # 円環に近いと低い
    O = np.where(iso, 0.3, o_k * o_m) * (0.3 + 0.7 * o_a)
    O = O * np.where(cat["degenerate_1d"], p["degenerate_penalty"], 1.0)
    # ---- Complexity: フラクタル次元と Lyapunov 指数の逆U字
    c_D = bell(cat["boxdim"], p["D_mu"], p["D_sigma"])
    c_L = bell(np.log(np.maximum(cat["lyap"], 1e-4)), p["lyap_mu"], p["lyap_sigma"])
    C = np.sqrt(c_D * c_L)
    # ---- Contrast: 余白 × 輪郭の鋭さ
    sh = stats["sharpness"]
    x0 = p["sharp_x0"] if p["sharp_x0"] is not None else np.median(sh)
    s = p["sharp_s"] if p["sharp_s"] is not None else max(np.subtract(*np.percentile(sh, [75, 25])) / 2, 1e-6)
    k_sh = sigmoid(sh, x0, s)
    k_neg = bell(stats["negative_space"], p["neg_mu"], p["neg_sigma"])
    k_clar = (1 - p["haze_w"] * sigmoid(stats["haze"], p["haze_x0"], p["haze_s"])) * \
             (1 - p["speckle_w"] * sigmoid(stats["speckle"], p["speckle_x0"], p["speckle_s"]))
    K = np.cbrt(k_sh * k_neg * k_clar)
    B = O ** p["wO"] * C ** p["wC"] * K ** p["wK"]
    return dict(beauty=B, order=O, complexity=C, contrast=K,
                o_k=o_k, o_mirror=o_m, o_angular=o_a, c_D=c_D, c_lyap=c_L,
                k_sharp=k_sh, k_neg=k_neg, k_clarity=k_clar)


def fit_weights_from_pairs(comp, pairs):
    """一対比較 [(i_win, j_lose), ...] から (wO, wC, wK) を推定（Bradley–Terry）。

    log Beauty = wO log O + wC log C + wK log K を効用とし、
    P(i > j) = σ(u_i - u_j) をロジスティック回帰（切片なし）で当てる。
    """
    from sklearn.linear_model import LogisticRegression
    L = np.stack([np.log(comp[c] + 1e-9) for c in ("order", "complexity", "contrast")], 1)
    pairs = np.asarray(pairs)
    X = L[pairs[:, 0]] - L[pairs[:, 1]]
    X = np.concatenate([X, -X]); y = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    m = LogisticRegression(fit_intercept=False, C=10.0).fit(X, y)
    w = np.maximum(m.coef_[0], 0)
    return dict(wO=w[0], wC=w[1], wK=w[2])

# --------------------------------------------------------------------------------------
# 埋め込みとクラスタリング
# --------------------------------------------------------------------------------------

def embed(cat, thumbs, mode="full", seed=0, scalar_weight=0.5):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Fs = shape_features(thumbs, k=cat["k"], mode=mode)
    Fs = StandardScaler().fit_transform(Fs)
    Fs = PCA(n_components=min(40, Fs.shape[1], len(Fs) - 1), random_state=seed).fit_transform(Fs)
    Fs /= np.sqrt((Fs ** 2).sum(1).mean())
    sc = np.stack([cat["boxdim"], np.log(cat["lyap"]), cat["entropy"], cat["fill"],
                   cat["mirror_score"].clip(0, 1.2)], 1)
    sc = StandardScaler().fit_transform(sc)
    sc /= np.sqrt((sc ** 2).sum(1).mean())
    Z = np.concatenate([Fs, scalar_weight * sc], 1)      # 形が主、指標は従
    try:
        import umap  # noqa
        Y = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=seed).fit_transform(Z)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Y = TSNE(n_components=2, perplexity=min(40, max(5, len(Z) // 50)), init="pca",
                 random_state=seed).fit_transform(Z)
        method = "t-SNE"
    return Z, Y, method


def cluster(Z, Y=None, min_size=None, space="map"):
    """HDBSCAN。space="map" は2次元地図上の密度で切る（地図と見た目が一致する）。
    space="feature" は特徴空間（PCA 10次元）で切る（距離の意味はこちらが忠実）。"""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    if space == "map" and Y is not None:
        Zr = (Y - Y.mean(0)) / Y.std(0)
    else:
        Zr = PCA(n_components=min(10, Z.shape[1]), random_state=0).fit_transform(Z)
    min_size = min_size or max(10, len(Z) // 80)
    lab = HDBSCAN(min_cluster_size=min_size, min_samples=8, copy=True,
                  cluster_selection_method="leaf").fit_predict(Zr)
    # 外れ値 (-1) は最も近いクラスタ重心へ割り当てる（地図上は区別して表示）
    out = lab < 0
    if out.any() and (~out).any():
        ids = np.unique(lab[~out])
        cent = np.stack([Zr[lab == c].mean(0) for c in ids])
        near = ids[np.argmin(((Zr[out][:, None] - cent[None]) ** 2).sum(-1), 1)]
        lab2 = lab.copy(); lab2[out] = near
    else:
        lab2 = lab
    return lab2, out

# --------------------------------------------------------------------------------------
# 可視化
# --------------------------------------------------------------------------------------

def _thumb_rgba(t, cmap):
    import matplotlib.pyplot as plt
    rgba = plt.get_cmap(cmap)(t[::-1] / 255.0)
    rgba[..., 3] = np.clip(t[::-1] / 255.0 * 4, 0, 1)    # 背景を透過
    return rgba


def plot_thumbnail_map(Y, thumbs, score, path, grid=22, title="", cmap="magma"):
    """埋め込み平面をグリッドに切り、各セルで score 最大の個体のサムネイルを置く。"""
    import matplotlib.pyplot as plt
    lo, hi = Y.min(0), Y.max(0)
    G = ((Y - lo) / (hi - lo + 1e-9) * (grid - 1e-6)).astype(int)
    fig, ax = plt.subplots(figsize=(13, 13))
    fig.patch.set_facecolor("#0b0b10"); ax.set_facecolor("#0b0b10")
    ax.scatter(G[:, 0] + 0.5 + (Y[:, 0] * 0) , G[:, 1] + 0.5, s=1, c="#333")
    best = {}
    for i, cell in enumerate(map(tuple, G)):
        if cell not in best or score[i] > score[best[cell]]:
            best[cell] = i
    for (gx, gy), i in best.items():
        ax.imshow(_thumb_rgba(thumbs[i], cmap), extent=(gx + 0.04, gx + 0.96, gy + 0.04, gy + 0.96))
    ax.set_xlim(0, grid); ax.set_ylim(0, grid); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color="w", fontsize=13)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(Y, lab, out, cat, beauty, path, method):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    ids = np.unique(lab)
    cmap = plt.get_cmap("tab20")
    ax = axes[0]
    for c in ids:
        m = lab == c
        ax.scatter(Y[m & ~out, 0], Y[m & ~out, 1], s=5, color=cmap(c % 20), alpha=0.8)
        ax.scatter(Y[m & out, 0], Y[m & out, 1], s=3, color=cmap(c % 20), alpha=0.25)
        cx, cy = np.median(Y[m], 0)
        ax.text(cx, cy, str(c), fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8))
    ax.set_title(f"clusters (HDBSCAN, {len(ids)})  — {method}")
    ax = axes[1]
    sc = ax.scatter(Y[:, 0], Y[:, 1], s=5, c=np.minimum(cat["n"], 12), cmap="turbo")
    plt.colorbar(sc, ax=ax, fraction=0.04); ax.set_title("map degree n")
    ax = axes[2]
    o = np.argsort(beauty)
    sc = ax.scatter(Y[o, 0], Y[o, 1], s=5, c=beauty[o], cmap="viridis")
    plt.colorbar(sc, ax=ax, fraction=0.04); ax.set_title("beauty score")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_cluster_gallery(lab, thumbs, cat, beauty, path, per=8, max_clusters=24, cmap="magma"):
    """クラスタごとに 美しさ上位 per 個を1行に並べ、行頭に代表的な指標を書く。"""
    import matplotlib.pyplot as plt
    ids = sorted(np.unique(lab), key=lambda c: -np.median(beauty[lab == c]))[:max_clusters]
    fig, axes = plt.subplots(len(ids), per + 1, figsize=((per + 1) * 1.25, len(ids) * 1.3))
    axes = np.atleast_2d(axes)
    for row, c in zip(axes, ids):
        m = np.flatnonzero(lab == c)
        top = m[np.argsort(-beauty[m])][:per]
        ns, cnt = np.unique(cat["n"][m], return_counts=True)
        txt = (f"C{c}  N={len(m)}\nB̃={np.median(beauty[m]):.2f}\n"
               f"D̃={np.median(cat['boxdim'][m]):.2f}\nλ̃={np.median(cat['lyap'][m]):.2f}\n"
               f"n: {','.join(map(str, ns[np.argsort(-cnt)][:3]))}")
        row[0].text(0, 0.5, txt, fontsize=6.5, va="center"); row[0].axis("off")
        for ax in row[1:]:
            ax.axis("off")
        for ax, i in zip(row[1:], top):
            ax.imshow(thumbs[i][::-1], cmap=cmap)
    plt.tight_layout(pad=0.2); fig.savefig(path, dpi=120); plt.close(fig)


def plot_beauty_ranking(thumbs, cat, comp, path, top=40, cmap="magma"):
    import matplotlib.pyplot as plt
    o = np.argsort(-comp["beauty"])[:top]
    nc = 8; nr = math.ceil(len(o) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 1.7, nr * 1.95))
    for ax in axes.ravel():
        ax.axis("off")
    for r, (ax, i) in enumerate(zip(axes.ravel(), o)):
        ax.imshow(thumbs[i][::-1], cmap=cmap)
        ax.set_title(f"{r+1}. #{cat['id'][i]} {cat['group'][i]}\nB={comp['beauty'][i]:.2f} "
                     f"O{comp['order'][i]:.2f} C{comp['complexity'][i]:.2f} K{comp['contrast'][i]:.2f}",
                     fontsize=6)
    fig.suptitle("Beauty = Order × Complexity × Contrast  (top)", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_wundt(cat, comp, path):
    """美しさ（O×K のみ）と複雑性指標の関係：逆U字が出るかの確認用。"""
    import matplotlib.pyplot as plt
    ok = comp["order"] * comp["contrast"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, lab in ((axes[0], cat["boxdim"], "box dimension D"),
                       (axes[1], np.log10(cat["lyap"]), "log10 Lyapunov λ₁")):
        ax.scatter(x, ok, s=3, alpha=0.4)
        bins = np.quantile(x, np.linspace(0, 1, 13))
        idx = np.clip(np.digitize(x, bins) - 1, 0, 11)
        med = [np.median(ok[idx == b]) if (idx == b).any() else np.nan for b in range(12)]
        ax.plot((bins[:-1] + bins[1:]) / 2, med, "r-o", ms=3)
        ax.set_xlabel(lab); ax.set_ylabel("Order × Contrast")
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

# --------------------------------------------------------------------------------------

def run(run_dir, mode="full", out=None, seed=0):
    out = out or os.path.join(run_dir, f"map_{mode}")
    os.makedirs(out, exist_ok=True)
    cat, thumbs = load_run(run_dir)
    stats = image_stats(thumbs)
    comp = beauty_components(cat, stats)
    Z, Y, method = embed(cat, thumbs, mode=mode, seed=seed)
    lab, outl = cluster(Z, Y)
    B = comp["beauty"]
    plot_thumbnail_map(Y, thumbs, B, os.path.join(out, "shape_map.png"),
                       title=f"Shape map ({mode}, {method}, N={len(B)}) — best-scoring icon per cell")
    plot_cluster_map(Y, lab, outl, cat, B, os.path.join(out, "clusters.png"), method)
    plot_cluster_gallery(lab, thumbs, cat, B, os.path.join(out, "cluster_gallery.png"))
    plot_beauty_ranking(thumbs, cat, comp, os.path.join(out, "beauty_top.png"))
    plot_wundt(cat, comp, os.path.join(out, "wundt_check.png"))
    np.savez_compressed(os.path.join(out, "map.npz"), Y=Y, cluster=lab, outlier=outl,
                        id=cat["id"], **comp, **stats)
    # catalog に追記した CSV
    with open(os.path.join(run_dir, "catalog.csv")) as f:
        rows = list(csv.DictReader(f))
    extra = dict(x=Y[:, 0], y=Y[:, 1], cluster=lab, **{k: comp[k] for k in
                 ("beauty", "order", "complexity", "contrast")}, **stats)
    with open(os.path.join(out, "catalog_scored.csv"), "w") as f:
        cols = list(rows[0].keys()) + list(extra.keys())
        f.write(",".join(cols) + "\n")
        for i, r in enumerate(rows):
            f.write(",".join([r[c] for c in rows[0].keys()] +
                             [f"{extra[k][i]:.5g}" for k in extra]) + "\n")
    print(f"{method}: N={len(B)}  clusters={len(np.unique(lab))}  outliers={outl.sum()}  -> {out}")
    return cat, thumbs, comp, Y, lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "petal"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.run, mode=a.mode, out=a.out)
