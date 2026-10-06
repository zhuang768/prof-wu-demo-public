---
title: MobileNetV2 Static Quantization Demo
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# MobileNetV2 靜態量化展示

這是教授展示用的 Gradio 原型：上傳自己可使用的照片，觀察同一圖片在 FP32 與 INT8 MobileNetV2 的 top-1 分類、機率、推論時間（毫秒）與模型檔案大小（MB）。模型推論是真實執行，校準資料則為 10 個隨機 tensor。

## 本機執行

需要 Python 與可用的 PyTorch CPU 環境。安裝依賴後啟動：

```sh
python -m pip install -r requirements.txt
python app.py
```

首次啟動會下載 torchvision 預訓練 MobileNetV2 與 ImageNet 類別標籤，並執行 qnnpack 的 fuse、prepare、校準、convert 流程。程式會在執行目錄自動產生 `model_fp32.pth`、`model_int8.pth`；它們是生成檔，沒有納入乾淨公開版本。若標籤下載失敗，會使用通用類別名稱。

展示照片、私人開發文件和網站捷徑也未納入；請自行提供有權使用的照片。本次準備未執行模型下載、推論或雲端部署，依賴版本與端到端相容性仍需在目標環境驗證。

## 比較範圍

- 校準使用隨機資料，沒有代表性影像校準集或 accuracy 評估。
- 每次操作在暖身後對兩個模型各量測一次 `time.time()`；數值不是穩定的 benchmark 或硬體加速驗證。
- FP32 物件先經過 fuse／prepare，再轉換得到 INT8；介面的「原始模型」指此展示中的 FP32 比較對象。
- 模型大小來自程式生成的 state dictionary 檔案，不能直接代表所有部署環境的記憶體用量。

## 選擇性部署至 Hugging Face

部署是執行者主動進行的獨立步驟。先在自己帳號下準備既有的 Gradio Space，設定 `HF_SPACE_ID` 為 `使用者名稱/Space名稱`，再透過執行環境或秘密管理工具提供 `HF_TOKEN`。不要把 token 寫進程式、命令紀錄或 Git。

`.env.example` 只有 placeholder；腳本不會自動載入 `.env`，也不使用帳號的隱藏登入快取。HF_TOKEN 會先驗證，再匯入 SDK 並作為明確參數傳入。缺少環境變數、仍用 placeholder、命名空間不是 token 所屬使用者，或缺少必要公開檔案時，都會停止。腳本只上傳到既有 Space，不會建立 Space 或修改可見性。環境變數行為可參考 [Hugging Face 官方文件](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf_token)。

```sh
python deploy_to_hf.py
```

固定上傳清單只有 `app.py`、`requirements.txt`、`README.md`。`.env`、Git 資料、生成模型、圖片、私人文件與本機絕對路徑不會被上傳。SDK 錯誤只顯示一般診斷，不顯示 token 或完整回應。

## 離線部署流程測試

```sh
python -m unittest discover -s tests -v
```

測試使用 fake `HfApi` 和明確的測試字串，不使用真實憑證、不存取 Hugging Face、不建立或上傳任何遠端資源，也不匯入會下載與生成模型的 `app.py`。
