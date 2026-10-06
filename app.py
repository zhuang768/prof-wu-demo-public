import torch
import torchvision.models.quantization as models
from torchvision import transforms
import torch.quantization
from PIL import Image
import time
import urllib.request
import gradio as gr
import os

# 模擬單核心邊緣運算設備 (Edge Device)
torch.set_num_threads(1)

# 1. 取得 ImageNet 分類標籤
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = urllib.request.urlopen(url)
        categories = [s.strip().decode('utf-8') for s in response.readlines()]
        return categories
    except Exception as e:
        return [f"Class {i}" for i in range(1000)]

categories = load_labels()

print("準備環境與模型...")
# 2. 載入原始 MobileNetV2 模型 (FP32)，注意這邊要使用 quantization 子模組下的模型
model_fp32 = models.mobilenet_v2(pretrained=True, quantize=False)
model_fp32.eval()

# 設定量化引擎 (解決 Mac ARM M1/M2/M3 晶片的 NoQEngine 錯誤)
torch.backends.quantized.engine = 'qnnpack'

# 3. 實作真實邊緣晶片需要的「靜態量化 (Static Quantization / PTQ)」
print("進行 Conv2d 融合與靜態量化校正...")
# (a) 融合 Conv+BN+ReLU 層以加速推論
model_fp32.fuse_model()
# (b) 設定量化配置
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
# (c) 插入 Quant/DeQuant 觀察點
torch.quantization.prepare(model_fp32, inplace=True)

# (d) 校正迴圈 (Calibration Loop)：送入模擬資料讓模型抓取神經元活化的範圍 (Min/Max)
def calibrate(model, data_batches):
    model.eval()
    with torch.no_grad():
        for batch in data_batches:
            model(batch)

# 產生 10 張模擬的隨機圖片來做校正
dummy_data = [torch.randn(1, 3, 224, 224) for _ in range(10)]
calibrate(model_fp32, dummy_data)

# (e) 轉換為真正的 INT8 模型
model_int8 = torch.quantization.convert(model_fp32, inplace=False)
model_int8.eval()

# 將模型存檔以比較大小
torch.save(model_fp32.state_dict(), "model_fp32.pth")
torch.save(model_int8.state_dict(), "model_int8.pth")
size_fp32 = os.path.getsize("model_fp32.pth") / (1024 * 1024)
size_int8 = os.path.getsize("model_int8.pth") / (1024 * 1024)

# 4. 影像預處理設定
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5. 定義辨識與時間測量函式
def predict(image):
    if image is None:
        return "請上傳圖片", "請上傳圖片"

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # 建立 mini-batch
    
    # 暖機 (Warmup) 讓 CPU/記憶體進入狀態
    with torch.no_grad():
        model_fp32(input_batch)
        model_int8(input_batch)

    # --- FP32 原始模型推論 ---
    start_time = time.time()
    with torch.no_grad():
        output_fp32 = model_fp32(input_batch)
    fp32_time = (time.time() - start_time) * 1000 # 轉換為毫秒
    
    probabilities_fp32 = torch.nn.functional.softmax(output_fp32[0], dim=0)
    top1_prob_fp32, top1_catid_fp32 = torch.topk(probabilities_fp32, 1)
    label_fp32 = categories[top1_catid_fp32[0]]
    
    # --- INT8 壓縮模型推論 ---
    start_time = time.time()
    with torch.no_grad():
        output_int8 = model_int8(input_batch)
    int8_time = (time.time() - start_time) * 1000 # 轉換為毫秒
    
    probabilities_int8 = torch.nn.functional.softmax(output_int8[0], dim=0)
    top1_prob_int8, top1_catid_int8 = torch.topk(probabilities_int8, 1)
    label_int8 = categories[top1_catid_int8[0]]

    # 輸出文字格式化
    res_fp32 = (
        f"辨識結果: {label_fp32} ({top1_prob_fp32[0].item():.2%})\n"
        f"花費時間: {fp32_time:.2f} 毫秒\n"
        f"模型大小: {size_fp32:.2f} MB"
    )
    
    res_int8 = (
        f"辨識結果: {label_int8} ({top1_prob_int8[0].item():.2%})\n"
        f"花費時間: {int8_time:.2f} 毫秒\n"
        f"模型大小: {size_int8:.2f} MB"
    )
    
    return res_fp32, res_int8

# 6. 使用 Gradio 建立網頁介面
with gr.Blocks(title="AI 邊緣運算：靜態量化展示", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# AI 邊緣運算展示：MobileNetV2 靜態量化 (Static Quantization / PTQ)")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="請上傳一張照片")
            btn = gr.Button("開始辨識", variant="primary")
        
    with gr.Row():
        out_fp32 = gr.Textbox(label="原始模型 (FP32)", lines=4)
        out_int8 = gr.Textbox(label="靜態壓縮模型 (INT8 / PTQ)", lines=4)
        
    btn.click(fn=predict, inputs=img_input, outputs=[out_fp32, out_int8])

if __name__ == "__main__":
    print("啟動 Gradio 網頁介面...")
    demo.launch()
