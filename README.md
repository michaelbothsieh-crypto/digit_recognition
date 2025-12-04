# 手寫數字辨識系統

這是一個基於網頁的手寫數字辨識系統，使用 Python (FastAPI, Scikit-learn) 作為後端，並使用原生 HTML/JS 作為前端。

## 功能特色

*   **手寫板介面**：支援滑鼠與觸控操作的畫布。
*   **即時辨識**：畫完數字後點擊「辨識」即可獲得結果。
*   **高準確率**：使用經過資料增強 (Data Augmentation) 訓練的 MLP (Multi-Layer Perceptron) 模型，準確率達 98.7%。
*   **強大的前處理**：
    *   自動裁切數字邊界 (Bounding Box)。
    *   縮放至標準 20x20 大小。
    *   使用質心 (Center of Mass) 自動置中。
    *   空白輸入檢測。

## 系統需求

*   Python 3.10+ (測試環境為 Python 3.14)

## 安裝與執行

1.  **建立並啟動虛擬環境**：
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate  # Windows
    ```

2.  **安裝依賴套件**：
    ```bash
    pip install -r requirements.txt
    ```

3.  **訓練模型 (首次執行需要)**：
    ```bash
    python3 model/train.py
    ```
    *這會自動下載 MNIST 資料集並訓練模型，訓練完成後會產生 `model/mnist_model.pkl`。*

4.  **啟動後端伺服器**：
    ```bash
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

5.  **使用系統**：
    打開瀏覽器前往 [http://localhost:8000](http://localhost:8000)。

## 專案結構

*   `app/main.py`: FastAPI 後端程式碼，包含影像前處理與預測 API。
*   `model/train.py`: 模型訓練腳本。
*   `static/`: 前端檔案 (HTML, CSS, JS)。
*   `requirements.txt`: 專案依賴列表。

## 開發筆記

*   **模型選擇**：由於 Python 3.14 兼容性問題，本專案使用 Scikit-learn 的 MLPClassifier 取代 TensorFlow/PyTorch。
*   **除錯**：預處理後的影像會儲存在 `debug_images/` 資料夾中（需手動開啟功能或查看程式碼）。
