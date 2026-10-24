# Generative Flower — Order × Chaos

**🌐 Web ページ: https://york-14.github.io/Generative-flower-Order-Chaos/**

| ページ | URL |
|---|---|
| 2D 版 | https://york-14.github.io/Generative-flower-Order-Chaos/ |
| 2D 版 ver.2（3 × 3） | https://york-14.github.io/Generative-flower-Order-Chaos/2d-v2.html |
| 3D 版 | https://york-14.github.io/Generative-flower-Order-Chaos/3d.html |
| 3D 版 ver.2 | https://york-14.github.io/Generative-flower-Order-Chaos/3d-v2.html |
| 3D 版 ver.3 | https://york-14.github.io/Generative-flower-Order-Chaos/3d-v3.html |

対称カオス写像（*symmetric icons*）が描く「花」のような図形を大量に探索し、

1. **形の地図** — 回転・鏡映に不変な形状特徴を 2 次元に埋め込み（UMAP / t-SNE）、HDBSCAN でクラスタに分ける
2. **美しさの評価関数** — `Beauty = Order^wO × Complexity^wC × Contrast^wK`
3. **重みの較正** — 人間の一対比較（A と B どちらが美しいか）から Bradley–Terry モデルで `wO, wC, wK` を推定

を行う解析ツール `shape_map.py` です。

「秩序（対称性）」と「カオス（フラクタル・不安定性）」がちょうど釣り合ったところに美しさがある、という仮説を数値で扱えるようにすることが目的です。

---

## 背景：Symmetric Icons

カタログの列 `lam, alpha, beta, gamma, omega, n` は、Field & Golubitsky の対称カオス写像（複素平面上の写像）

$$
z_{t+1} = \left(\lambda + \alpha |z|^2 + \beta\,\mathrm{Re}(z^n) + i\omega\right) z + \gamma\,\bar{z}^{\,n-1}
$$

のパラメータに対応します。この写像は $n$ 回回転対称（$\omega = 0$ なら鏡映も含む二面体群 $D_n$）を持ち、軌道を長時間プロットするとアトラクタが対称な「花」の模様になります。

`shape_map.py` は、この探索を行う **`symmetric_icons.py`（本リポジトリ外）** の出力を入力として受け取ります。

---

## 処理の流れ

```mermaid
flowchart LR
    A[catalog.csv<br/>thumbs.npy] --> B[load_run]
    B --> C[image_stats<br/>鋭さ・余白・もや・斑点]
    B --> D[shape_features<br/>極座標 FFT]
    C --> E[beauty_components<br/>Order × Complexity × Contrast]
    D --> F[embed<br/>Standardize → PCA40 → UMAP / t-SNE]
    F --> G[cluster<br/>HDBSCAN leaf]
    E --> H[可視化 PNG<br/>map.npz / catalog_scored.csv]
    G --> H
```

### 1. 形状特徴（回転・鏡映不変）

| 関数 | 内容 |
|---|---|
| `polar_resample` | サムネイル `(N,H,H)` を中心まわりの極座標 `(N, n_r=16, n_θ=256)` に双線形補間で再標本化 |
| `shape_features` | 各半径リングで角度方向に FFT。振幅 `\|FFT_m\|` は回転では位相しか変わらず、実信号なので反転でも不変 → **回転・鏡映不変**。DC 成分で正規化し `log1p(10·x)` で圧縮。半径方向の平均プロファイルも連結 |

特徴量の 2 つのモード：

| モード | 使う周波数 | 地図の性格 |
|---|---|---|
| `full` | `m = 1 … 48` | 対称次数 *n* も形の一部。地図は主に「何回対称か」で大きく分かれ、その中で形が並ぶ |
| `petal` | `m = k, 2k, …, 10k`（*k* = 検出された回転次数） | 対称性を「割り算」して、基本領域（花びら 1 枚）の描き方だけを比較。5 回対称と 8 回対称の花でも花びらが似ていれば近くに来る |

### 2. 埋め込みとクラスタリング

- `embed`: 形状特徴を標準化 → PCA（最大 40 次元）→ RMS 正規化。さらにスカラー指標（ボックス次元、log Lyapunov、エントロピー、充填率、鏡映スコア）を重み `0.5` で連結（**形が主、指標は従**）。UMAP（`n_neighbors=30, min_dist=0.15`）があれば UMAP、なければ t-SNE。
- `cluster`: HDBSCAN（`cluster_selection_method="leaf"`, `min_samples=8`, `min_cluster_size = max(10, N/80)`）。
  - `space="map"`（既定）: 2 次元地図上の密度で切る → 地図の見た目と一致
  - `space="feature"`: PCA 10 次元の特徴空間で切る → 距離の意味に忠実
  - 外れ値（ラベル −1）は最寄りクラスタ重心に割り当て、地図上では薄く表示

### 3. 美しさの評価関数

$$
\mathrm{Beauty} = O^{w_O}\cdot C^{w_C}\cdot K^{w_K}\qquad(\text{既定 } w_O=w_C=w_K=1)
$$

「中庸が最も好まれる」という **Wundt 曲線（逆 U 字）** をガウス型 `bell(x, μ, σ)` で、「多いほど良い／悪い」をシグモイドで表しています。

#### Order（秩序）*O*

| 要素 | 式 | 意味 |
|---|---|---|
| `o_k` | `bell(log k, log 6, 0.55)` | 回転次数はおよそ 4〜10 回を好む |
| `o_mirror` | `0.65 + 0.35·clip(mirror_score, 0, 1)` | 鏡映対称があると加点 |
| `o_angular` | `sigmoid(angular_contrast, 0.25, 0.07)` | 角度方向に模様が読み取れるか（円環に近いと低い） |
| 特例 | group が `O2~`（ほぼ円対称）→ `0.3` に固定 / `degenerate_1d`（直線・スポーク状）→ ×0.25 | |

`O = (o_k · o_mirror) · (0.3 + 0.7 · o_angular)`

#### Complexity（複雑さ）*C*

| 要素 | 式 | 意味 |
|---|---|---|
| `c_D` | `bell(boxdim, 1.72, 0.11)` | フラクタル次元：線（1）でも塗りつぶし（2）でもない中庸 |
| `c_lyap` | `bell(log λ₁, log 0.35, 0.7)` | 最大 Lyapunov 指数：弱すぎず強すぎないカオス |

`C = √(c_D · c_lyap)`

#### Contrast（図と地）*K*

`image_stats` がサムネイルから計算する統計量を使います。

| 統計量 | 定義 |
|---|---|
| `sharpness` | 占有画素の平均勾配 ÷ 平均輝度（フィラメント・輪郭の鋭さ） |
| `negative_space` | 外接円内でアトラクタが占めない割合（余白） |
| `haze` | 占有画素のうち暗い画素（0.02 < I < 0.4）の割合（淡いもや） |
| `speckle` | 4 近傍の半分以上が空の孤立画素の割合（ノイズ） |
| `dynamic_range` | 明暗の標準偏差 ÷ 平均（出力のみ、評価には未使用） |

| 要素 | 式 |
|---|---|
| `k_sharp` | `sigmoid(sharpness, 中央値, IQR/2)`（データ分布から自動設定） |
| `k_neg` | `bell(negative_space, 0.45, 0.22)` — 適度な余白 |
| `k_clarity` | `(1 − 0.6·sigmoid(haze, 0.5, 0.1)) · (1 − 0.7·sigmoid(speckle, 0.2, 0.04))` |

`K = ∛(k_sharp · k_neg · k_clarity)`

パラメータはすべて `DEFAULT_BEAUTY` 辞書にまとまっており、`beauty_components(cat, stats, p={...})` で部分的に上書きできます。

### 4. 一対比較からの重み較正（Bradley–Terry）

`fit_weights_from_pairs(comp, pairs)` は、効用を

$$
u_i = \log B_i = w_O \log O_i + w_C \log C_i + w_K \log K_i
$$

とし、$P(i \succ j) = \sigma(u_i - u_j)$ を切片なしロジスティック回帰で当てはめます（対称化のため反転サンプルも追加、負の重みは 0 にクリップ）。

```python
from shape_map import load_run, image_stats, beauty_components, fit_weights_from_pairs

cat, thumbs = load_run("out/run1")
comp = beauty_components(cat, image_stats(thumbs))
pairs = [(12, 40), (7, 3), ...]                 # (勝った index, 負けた index)
w = fit_weights_from_pairs(comp, pairs)         # {'wO': ..., 'wC': ..., 'wK': ...}
comp2 = beauty_components(cat, image_stats(thumbs), p=w)
```

---

## 使い方

```bash
pip install -r requirements.txt

python shape_map.py --run out/run1                 # 全形の地図 + 美しさ（full モード）
python shape_map.py --run out/run1 --mode petal    # 花びら地図
python shape_map.py --run out/run1 --out my_map    # 出力先を指定
```

### 入力（`--run` ディレクトリ）

| ファイル | 内容 |
|---|---|
| `catalog.csv` | 1 行 1 個体。必須列: `id, n, k, k_orbit, group, map_group, lyap, boxdim, entropy, fill, mirror_score, angular_contrast, degenerate_1d`。任意: `lam, alpha, beta, gamma, omega, R, alive, breaking_amp, symmetry_broken, extra_mirror` |
| `thumbs.npy` | `uint8` 配列 `(N, H, H)`。中心が原点の正方形サムネイル（輝度 0–255） |

### 出力（既定: `<run>/map_<mode>/`）

| ファイル | 内容 |
|---|---|
| `shape_map.png` | 埋め込み平面を 22×22 のグリッドに切り、各セルで美しさ最大の個体のサムネイルを配置した「形の地図」 |
| `clusters.png` | 3 面図：HDBSCAN クラスタ / 写像次数 *n* / 美しさスコア |
| `cluster_gallery.png` | クラスタごと（美しさの中央値順、最大 24）に上位 8 個を 1 行に並べたギャラリー |
| `beauty_top.png` | 美しさ上位 40 個と O / C / K の内訳 |
| `wundt_check.png` | `Order × Contrast` とボックス次元・Lyapunov 指数の関係（逆 U 字が出るかの確認用） |
| `map.npz` | 埋め込み座標 `Y`、クラスタ、外れ値フラグ、全スコア成分、画像統計量 |
| `catalog_scored.csv` | 元の `catalog.csv` に `x, y, cluster, beauty, order, complexity, contrast` と画像統計量を追記 |

---

## 動作環境

- Python 3.9+
- `numpy`, `matplotlib`
- `scikit-learn >= 1.3`（`sklearn.cluster.HDBSCAN` を使用）
- `umap-learn`（任意。無ければ t-SNE に自動フォールバック）

## コード解析メモ

- `catalog.csv` の `True` / `False` 文字列は `== "True"` で真偽値化しています。
- `plot_thumbnail_map` 内の `+ (Y[:, 0] * 0)` は値に影響しない項です（削除しても結果は同じ）。
- `catalog_scored.csv` は `csv` モジュールを使わず手書きで出力しているため、元 CSV の値にカンマが含まれると列がずれます。
- 評価関数のパラメータ（`k_mu=log 6`, `D_mu=1.72` など）は経験的な初期値であり、`fit_weights_from_pairs` による較正を前提としています（現状の較正対象は指数 `wO, wC, wK` のみ）。

## Web ページ（GitHub Pages）

`docs/index.html` が Web ページ本体です（ビルド不要の 1 ファイル）。

| 機能 | 内容 |
|---|---|
| 描画スタジオ | パラメータ λ, α, β, γ, ω, n をスライダーで操作し、ブラウザ上でアトラクタを描画。プリセット・ランダム探索・PNG 保存 |
| 色彩の調整 | パレット（マグマ／氷・海／桜／金箔／翡翠／墨 など）、カスタム 3 色グラデーション、色相回転、ガンマ、露出、背景色。軌道を再計算せずに即座に塗り直す |
| 好みの学習 | 2 つの花を並べて好きな方を選ぶ（クリック または ← / →）。選択のたびにブラウザ内で Bradley–Terry モデルを再推定し、出題と「あなた好みの花を描く」に反映 |

### 2D 版 ver.2（`docs/2d-v2.html`）

**🌐 https://york-14.github.io/Generative-flower-Order-Chaos/2d-v2.html**

2D 版（`docs/index.html`）はそのままに、一枚のビジュアル（1 つのキャンバス）の上に花を 3 × 3 で 9 つ配置し、それぞれに異なるパラメータをランダムに適用する版です。色は 9 つ共通の 1 系統です。

| 機能 | 内容 |
|---|---|
| 9 つをランダム探索 | 9 つの花それぞれに独立なランダムパラメータ（λ, α, β, γ, ω, n）を与える。ver.1 のランダム探索と同じ基準（発散しない・ボックス次元と余白が中庸・Lyapunov 指数が正）を満たすものだけを採用 |
| 一枚の画面に合成 | 区切りなしで同じ背景の上に配置。花ごとに密度を正規化し、スクリーン合成してから共通のパレットで塗る。「大きさ」を 1 より大きくすると隣の花と重なって混ざる |
| 色彩（9 つ共通） | パレット・色相・ガンマ・露出・背景は全体で 1 つ。「色を抽選」で系統ごと変更、探索のたびに変えることも可 |
| 花の選択と調整 | 画面上の花をクリック（または矢印キー）で選び、パラメータを編集。選択枠は初期設定で非表示（「選択枠を表示」でオン）。画像には含まれない |
| ロック・再抽選・変奏 | ロックした花は再探索・変奏で置き換えない。選んだ花だけの再抽選、選んだ花の近傍パラメータで残りを埋める変奏も可 |
| シード | シードは 12 桁の数（000000000000〜999999999999、全 10¹² 通り）。日付 YYYYMMDDHHmm（例 202610241756）もそのまま使え、「今の日時で生成」ボタンもある。シードを上位・下位の 32bit に分けて系列番号と混ぜ（splitmix32）、128bit 状態の sfc32 を初期化するので、10¹² 通りがすべて別の系列になる。花 i は系列 i、色は系列 9。シードは画面左下に 12 桁ゼロ埋めで小さく表示し、日付として読めるときは日付も併記する（PNG には含まれない）。入力欄または URL の `?seed=` で指定でき、13 桁以上はエラー、数字以外の文字列は 64bit FNV-1a でハッシュして 12 桁に収める。再抽選・変奏・編集でシードの結果から変わると `*` が付く。色は「探索のたびに色の系統も変える」がオンのときだけシードから決まる |
| 保存 | 一枚の PNG（1440 × 1440）で保存。状態は localStorage に保存され、再読み込みしても残る |

### 3D 版 ver.3（`docs/3d-v3.html`）

**🌐 https://york-14.github.io/Generative-flower-Order-Chaos/3d-v3.html**

ver.2 の全モードに、流体力学に基づく「煙」モードを加えたものです。アトラクタの密度を発生源にして、非圧縮性流体（Stable Fluids：MacCormack 移流・浮力・渦度閉じ込め・圧力の Jacobi 反復・移流拡散）を GPU 上で解き、ver.2 のインク用ボリュームレンダラで描きます。

| 設定 | 内容 |
|---|---|
| 発生源 | 呼吸（初期設定：形 → 横へ拡散 → 混沌 → 収束して元の形へ、を小節に同期して繰り返す）／解放（彫刻がほどける）／継続放出／流線放出（写像のヤコビアンで求めたフィラメント方向に初速） |
| 対称性の保持 | 毎ステップ、ρ と u を対称群で平均した場と補間。0 で乱流、1 で対称な煙の花 |
| 見た目 | 煙（暗い背景・高アルベド）／水中のインク（明るい背景・吸収寄り・色付き）、温度による色づけ |
| 流体 | 浮力 α・β、渦度、乱れ、拡散 κ、減衰、放出量、時間刻み、Jacobi 回数、格子 64³ / 96³、上面の開放 |
| 操作 | 再生・一時停止・リセット。「かき混ぜ」をオンにするとドラッグで流れを起こせる |

### 3D 版 ver.2（`docs/3d-v2.html`）

**🌐 https://york-14.github.io/Generative-flower-Order-Chaos/3d-v2.html**

3D 版の写像を、点群に加えて 3D 密度ボリュームとしても描きます。軌道を n 回回転（ω = 0 なら鏡映も）で複製して 3D ヒストグラム（128³ / 192³）を作り、Web Worker で log 圧縮・ガウスぼかしをかけて WebGL2 の 3D テクスチャに格納します。

| モード | 内容 |
|---|---|
| 点群 | 3D 版と同じ描画 |
| 等値面（ガラス／金属） | レイマーチングで等値面を求め、ガラス（フレネル・屈折・Beer–Lambert 吸収）または金属（水銀・銀・金・銅・ローズゴールド・チタン・黒鉄、任意色）で描く。背景のグラデーション・照明の色・露出も調整可 |
| 流線 | 密度場から作った「形の表面に沿う」流れ v = ∇ρ × d を積分した細い線で描く。間隔を保って配置し、対称像はインスタンス描画で複製。線上をテンポに同期した光の粒が流れる。墨流し・紅と藍・藍染・翡翠・夜光・金糸のスタイル |

プリセットは「回転対称の秩序（Order）と、カオス軌道が刻む細部（Chaos）が同居する自然のかたち」として、クラゲ・薔薇・百合・菊・蓮の花托・イソギンチャク・ウニの殻・ハイビスカス・朝顔・サンゴの 10 種を用意しています。初期表示は水銀の等値面です。

ドラッグ中は低解像度で描き、止まると高品質の画像をジッターの累積で仕上げます。URL に `?mode=iso` / `?mode=flow` / `?mode=points` を付けるとそのモードで、`?p=n,λ,α,β,γ,ω,κ,μ,δ,ε` を付けるとそのパラメータで開きます（`&mat=glass` でガラス）。

### 3D 版（`docs/3d.html`）

**🌐 https://york-14.github.io/Generative-flower-Order-Chaos/3d.html**

平面の写像に実数の高さ *w* を加えた 3 次元の写像を、WebGL2 で立体表示します。

$$
\begin{aligned}
z_{t+1} &= \left(\lambda + \kappa w_t + \alpha |z|^2 + \beta\,\mathrm{Re}(z^n) + i\omega\right) z + \gamma\,\bar{z}^{\,n-1} \\
w_{t+1} &= \mu w_t + \delta\,|z|\cos(n \arg z) + \varepsilon |z|^2
\end{aligned}
$$

| パラメータ | 役割 |
|---|---|
| κ（結合） | 高さが平面の動きに戻る強さ。0 なら平面の花に高さが付くだけ、0 以外では形自体が立体的に変わる |
| μ（記憶） | 高さが過去を覚えている度合い（層の厚さ・起伏のなめらかさ） |
| δ（起伏） | 花びらの向き cos(n·arg z) に応じた上下 → n 枚の花弁が交互に反り返る |
| ε（お椀） | 中心からの距離に応じた上下（負 = お椀、正 = ドーム） |

- *w* の更新式は z を 2π/n 回しても変わらない量だけでできているため、**n 回回転対称が縦軸まわりの対称性として保たれます**（ω = 0 なら鏡映も含む D<sub>n</sub>）。
- 描画は外部ライブラリなし。点を浮動小数点バッファに加算合成して密度を求め、2D 版と同じパレット・ガンマ・露出でトーンマッピングします。
- ドラッグで回転、ホイール／ピンチで拡大縮小、自動回転。色は「高さ」「密度」「中心からの距離」から選択。
- ランダム探索は、発散せず、3D 箱数え次元が中庸で、最大 Lyapunov 指数が正（カオス）のものだけを採用します。
- 色彩設定は 2D 版と共有（同じ localStorage キー）。好みの学習は 2D 版のみです。

### 好みの学習の仕組み

- 各候補について、軌道を 128×128 のヒストグラムに落として **ボックス次元・余白率・もや率・斑点率・角度コントラスト・最大 Lyapunov 指数** を計算し、`shape_map.py` の `Order / Complexity / Contrast` をブラウザ用に近似します。
- 効用は `u = wO·log O + wC·log C + wK·log K + Σ vj·φj`（φ = 対称数・次元・カオスの強さ・余白などの 1 次・2 次項）。
- 初期の評価関数（`w = 1, v = 0`）を事前分布とする L2 正則化付きロジスティック回帰で、選択のたびに全比較データから再推定します。比較が少ないうちは初期関数に近く、増えるほど個人の好みに寄ります。
- 出題の半分は「現在のモデルで最良の候補」を含め、残りはランダム（探索と活用のバランス）。
- 学習データと色彩設定は **そのブラウザの localStorage にのみ保存** され、外部には送信されません（端末・ブラウザごとに別々の学習になります。「学習をリセット」で消去）。
- 学習結果は「初期の評価関数からのずれ」として棒グラフで表示されます（右 = 好き、左 = 苦手）。
