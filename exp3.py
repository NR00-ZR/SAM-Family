import webbrowser
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==========================================
# 实验三配置：SAM 3 语义概念分割验证
# ==========================================
DEMO_URL = "https://aidemos.meta.com/segment-anything/editor/segment-image"

# 定义报告中提到的测试案例
test_cases = [
    {
        "image": "dataset/exp3/street_scene.jpg",
        "prompts": ["women wearing sunglasses", "person wearing red scarf"],
        "result_file": "results/exp3/street_scene_result.png",
        "description": "验证属性对齐能力 (近景/显著目标)"
    },
    {
        "image": "dataset/exp3/market_scene.jpg",
        "prompts": ["woman handing a yellow coat", "woman wearing red coat"],
        "result_file": "results/exp3/market_scene_result.png",
        "description": "验证长尾语义与感知边界 (远景/复杂交互)"
    }
]

def run_experiment_guide():
    print(f"\n正在启动 SAM 3 在线实验验证...")
    print(f"目标网址: {DEMO_URL}")
    
    # 1. 打开浏览器
    webbrowser.open(DEMO_URL)
    
    print("\n" + "="*60)
    print(" 实验操作指引 (Human-in-the-loop Experiment)")
    print("="*60)
    
    # 2. 打印操作步骤
    for idx, case in enumerate(test_cases):
        print(f"\n[测试案例 {idx+1}]: {case['description']}")
        print(f"  1. 请上传图片: {os.path.abspath(case['image'])}")
        print(f"  2. 输入提示词 (Text Prompt):")
        for p in case['prompts']:
            print(f"     - \"{p}\"")
        print(f"  3. 请将网页生成的结果截图或保存，并命名为: {os.path.basename(case['result_file'])}")
        print(f"  4. 将结果放入目录: {os.path.dirname(case['result_file'])}/")

    # 3. 尝试展示结果 
    print("\n" + "="*60)
    print(" 结果可视化 (Result Visualization)")
    print("="*60)
    
    for case in test_cases:
        if os.path.exists(case['image']) and os.path.exists(case['result_file']):
            plt.figure(figsize=(12, 6))
            
            # 原图
            img_src = mpimg.imread(case['image'])
            plt.subplot(1, 2, 1)
            plt.imshow(img_src)
            plt.title(f"Input: {os.path.basename(case['image'])}")
            plt.axis('off')
            
            # 结果图
            img_res = mpimg.imread(case['result_file'])
            plt.subplot(1, 2, 2)
            plt.imshow(img_res)
            plt.title(f"SAM 3 Output\nPrompts: {case['prompts']}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            print(f"成功展示案例: {os.path.basename(case['image'])}")
        else:
            print(f"未找到结果文件: {case['result_file']} (请手动完成实验并保存图片)")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("results/exp3", exist_ok=True)
    run_experiment_guide()