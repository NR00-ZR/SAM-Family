import torch
from ultralytics import SAM
import cv2
import os
import time

video_path = "/data2/zhuangyn/04.mp4" 
SNAPSHOT_INTERVAL = 5   # 设置截图间隔
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🚀开始验证 SAM 2 视频处理能力 (每 {SNAPSHOT_INTERVAL} 帧保存一次截图)...")

# 对比模型
video_models = [
    {"name": "SAM 1 (Base)", "file": "sam_b.pt"},
    {"name": "SAM 2 (Base)", "file": "sam2_b.pt"} 
]

video_stats = []

for config in video_models:
    model_name = config['name']
    model_file = config['file']
    
    # 生成安全的文件名
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n--> 正在运行 {model_name} ...")
    
    try:
        model = SAM(model_file)
        t0 = time.time()
    
        # 使用 track 模式以启用视频流式处理和记忆功能
        results = model.track(
            source=video_path, 
            persist=True,        # 开启持久化追踪 
            stream=True,         # 流式生成器，节省内存
            device=device, 
            verbose=False,
            save=True,           # 保存视频
            project="runs/comparison",
            name=safe_name,      
            exist_ok=True        
        )
        
        frame_idx = 0
        processed_frames = 0

        for r in results:
            frame_idx += 1
            processed_frames += 1
     
            if frame_idx % SNAPSHOT_INTERVAL == 0:
                # 文件名例如: video_frame_005_SAM_1_Base.jpg
                save_img_name = f"video_frame_{frame_idx:03d}_{safe_name}.jpg"
                
                # 绘制并保存图片
                res_img = r.plot()
                cv2.imwrite(save_img_name, res_img)
                
                # 只打印关键节点的提示，避免刷屏
                if frame_idx % 20 == 0:
                    print(f"    [进度] 已处理 {frame_idx} 帧，最新截图: {save_img_name}")

        t1 = time.time()
        
        # 计算 FPS
        if processed_frames > 0:
            total_time = t1 - t0
            fps = processed_frames / total_time
            video_stats.append({"Model": model_name, "FPS": f"{fps:.2f}"})
            print(f"完成。保存路径: runs/comparison/{safe_name}/{os.path.basename(video_path)}")
            print(f"耗时: {total_time:.2f}s, 平均 FPS: {fps:.2f}")

    except Exception as e:
        print(f"运行出错: {e}")

print("\n" + "="*60)
print("[实验数据汇总]")
print("="*60)

if not video_stats:
    print("没有收集到实验数据。")
else:
    print(f"{'模型名称':<20} | {'FPS (处理速度)':<15}")
    print("-" * 40)
    for stat in video_stats:
        print(f"{stat['Model']:<20} | {stat['FPS']:<15}")

print("\n 截图已保存在当前目录下，请查看 video_frame_xxx.jpg")
print("="*60)