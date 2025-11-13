## Interstellar Kerr 黑洞卡

一張帶有實時著色與科學數據的黑洞資訊卡。前端使用 WebGL2 片段著色器重現《Interstellar》風格的 Kerr 黑洞，並提供輸入參數（自旋、傾角、觀測距離）供使用者調整。計算核心採用 C++ Kerr 光線追蹤器（可編譯為 WebAssembly）以及 Python 驗證腳本以確保推導公式的數值正確。

### 專案結構

- `index.html` / `src/`：卡片 UI、控制面板、WebGL 著色器與 JavaScript 管線
- `wasm/solver.cpp`：Kerr 幾何的 RK4 光線追蹤器，供 Emscripten 編譯為 WASM
- `validation/verify_kerr.py`：核對重力半徑、ISCO、光子球等量測的驗證腳本

### 本地預覽

1. 執行驗證腳本以產生 `validation/report.json`（供卡片顯示驗證狀態）：

   ```bash
   python3 validation/verify_kerr.py --out validation/report.json
   ```

2. 於專案根目錄啟動任何靜態伺服器（需支援 WebGL2）：

   ```bash
   python3 -m http.server 4173
   ```

3. 造訪 `http://localhost:4173`。若 `validation/report.json` 存在，頁面左下角會顯示 PASS/FAIL 與最大相對誤差。

### WASM 求解器

`wasm/solver.cpp` 提供 equatorial Kerr 幾何的光線積分器。若已安裝 Emscripten：

```bash
emcc wasm/solver.cpp -O3 -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_trace_kerr_bundle"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap","_malloc","_free"]' \
  -o public/kerr_solver.js
```

載入瀏覽器端模組後呼叫：

```js
import Module from "./public/kerr_solver.js";
window.attachKerrWasmModule(Module());
```

`BlackHoleCard` 會自動連結到 WASM 桥接器並以求解結果更新 shader 能量權重。

### 計算驗證

`validation/verify_kerr.py` 會：

1. 計算指定質量的重力半徑（SI 單位）。
2. 使用 Bardeen et al. 的解析公式計算 ISCO 與光子球半徑。
3. 與參考值比對並輸出相對誤差（需全部 < 8e-3 才視為 PASS）。

產出的 `validation/report.json` 可直接被前端輪詢顯示最新驗證摘要。

### 後續可行方向

- 將 `wasm/solver.cpp` 編譯為 WASM 並串接 `KerrWasmBridge`，取代目前 shader 內的近似能量。
- 擴充驗證腳本以讀取 WASM 求解資料並與解析解比對。
- 加入更多觀測資料（時間延遲、Doppler 映射、磁場線）強化卡片資訊密度。

## Immersive 分支版本

`branches/immersive/` 提供一個全畫面實驗室介面（參考 `1.html` 的設計），可 360° 旋轉並顯示更完整的公式推導。若要在 Git 中建立對應分支：

```bash
git checkout -b immersive-lab
git add branches/immersive
git commit -m "Add immersive Gargantua lab variant"
```

部署時只需將 `branches/immersive/index.html`、`styles.css`、`main.js` 連同主卡一起上傳，即可在 `/branches/immersive/index.html` 看到新版 UI。該版本與主卡共用同一套 Kerr 參數計算，並另外顯示 Rg、ISCO、Photon sphere、紅移與 β 的公式推導，方便對照實際數值。
