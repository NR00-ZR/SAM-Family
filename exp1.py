import time
import torch
from ultralytics import SAM
import matplotlib.pyplot as plt
import os
import cv2

# 1. 实验设置：选择要对比的模型
model_names = {
    "MobileSAM": "mobile_sam.pt",
    "SAM (Base)": "sam_b.pt",
    "SAM 2 (Tiny)": "sam2_b.pt" 
}

# 准备一张测试图片 
img_url = "https://ultralytics.com/images/bus.jpg"
img_path = "bus.jpg"
if not os.path.exists(img_path):
    torch.hub.download_url_to_file(img_url, img_path)

# 2. 循环运行实验：加载模型 -> 推理 -> 记录数据
performance_data = []
for model_pretty_name, model_file in model_names.items():
    print(f"\n正在加载模型: {model_pretty_name} ({model_file})...")
    
    model = SAM(model_file)
    _ = model(img_path, verbose=False) 
    
    # 正式推理
    start_time = time.time()
   
    results = model(img_path, verbose=True)  # 使用 'segment' 模式进行全图分割
    end_time = time.time()
    
    # 收集数据 
    inference_time = (end_time - start_time) * 1000   # 推理耗时 (毫秒)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # 参数量 (百万)
    
    performance_data.append({
        "Model": model_pretty_name,
        "Time (ms)": f"{inference_time:.2f}",
        "Params (M)": f"{params:.2f}"
    })
    
    # 保存结果图 
    res_img = results[0].plot()      # Ultralytics 的 plot() 方法会把掩码画在图上
    save_name = f"result_{model_file.replace('.pt', '')}.jpg"
    cv2.imwrite(save_name, res_img)
    print(f"结果图已保存: {save_name}")

print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)
print(f"{'模型名称':<20} | {'参数量 (M)':<15} | {'推理耗时 (ms)':<15}")
print("-" * 60)
for data in performance_data:
    print(f"{data['Model']:<20} | {data['Params (M)']:<15} | {data['Time (ms)']:<15}")
print("=" * 60)