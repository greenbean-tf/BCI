# 專題執行與環境設定流程 (Execution Flow)

這份文件提供完整的專案執行流程，包含如何建立虛擬環境、安裝所需的套件，以及如何啟動主程式。

## 一、 環境準備與依賴安裝

專案的環境相關設定檔統一放置在 `combine_all/env_setup/` 目錄下。請確保您的電腦已安裝 **Python 3.12**（參考 `.python-version`）。

### 方法一：使用 Python 內建 `venv` 與 `pip` (推薦一般使用者)

1. **建立虛擬環境**
   在專案根目錄 (`2026_BCI_-`) 下，開啟終端機並執行以下指令建立虛擬環境（名稱為 `venv`）：
   ```bash
   python -m venv venv
   ```

2. **啟動虛擬環境**
   - **Windows (Command Prompt / PowerShell)**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```
   啟動成功後，您的命令提示字元前面應該會多出 `(venv)`。

3. **安裝依賴套件**
   利用專案準備好的 `requirements.txt` 進行安裝，這將會自動下載所需的機器學習與遊戲相關套件：
   ```bash
   pip install -r combine_all/env_setup/requirements.txt
   ```

### 方法二：使用 `uv` (更快速的套件管理工具)

由於專案內有提供 `uv.lock` 與 `pyproject.toml`，如果您有安裝 `uv`，可以使用它來更快速地建立環境與同步依賴：

1. **進入環境設定資料夾**：
   ```bash
   cd combine_all/env_setup
   ```
2. **自動建立虛擬環境並同步套件**：
   ```bash
   uv sync
   ```
   *這將會根據 `uv.lock` 精確還原所需的套件版本。*

3. **退回並啟動主程式**：
   可以透過 `uv run` 直接執行外層的主程式：
   ```bash
   cd ../..
   uv run python combine_all/main.py
   ```

---

## 二、 執行主程式

環境設定與依賴安裝完成後，即可啟動專案的主要系統。

1. **進入程式主要目錄**
   請確保您位於 `combine_all` 資料夾內：
   ```bash
   cd combine_all
   ```

2. **啟動 main.py**
   執行主程式：
   ```bash
   python main.py
   ```

3. **系統啟動與遊玩流程**
   - **初始化**：程式將會初始化 LSL (腦波數據傳輸)、TCP Server/Client (跨程序通訊) 以及 osu 遊戲介面。
   - **帳號登入**：首次進入會進入帳號選擇畫面 (`run_account_login`)，建立或選擇帳號後，將為您載入專屬的強化學習(RL)設定。
   - **主選單 (Main Menu)**：
     - `開始遊戲 (song_select)`: 選擇譜面並進入遊戲。遊戲中會在背景同步處理腦波預測與資料收集。
     - `模型選擇 (model_select)`: 切換或訓練腦波辨識模型。
     - `測試模型 (test_model)`: 即時預測腦波並顯示結果。
     - `設定 (settings)`: 調整音量、下落速度與延遲 (Offset) 等。

---

## 三、 常見問題 (Troubleshooting)

- **找不到套件錯誤 (`ModuleNotFoundError`)**
  請確認您有正確啟動虛擬環境（終端機提示符號前有 `(venv)`），並且已完成 `requirements.txt` 的安裝。
- **Python 版本問題**
  本專案設定依賴 `>=3.12` 的環境。如果安裝套件時發生大量編譯錯誤，請確認您的 Python 版本是否正確。
- **PyTorch / 深度學習環境支援**
  安裝清單內包含了 `braindecode` 等依賴 PyTorch 的套件。若是您有 NVIDIA 獨立顯示卡且希望使用 GPU 加速，建議另外至 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查詢對應 CUDA 版本的安裝指令（例如：`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`）。
