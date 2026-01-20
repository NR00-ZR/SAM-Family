
# SAM Family: Evolution & Core Technology Research

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO%2FSAM-blue)](https://github.com/ultralytics/ultralytics)
[![Meta AI](https://img.shields.io/badge/Meta%20AI-SAM%203%20Demo-blueviolet)](https://aidemos.meta.com/segment-anything)

本仓库包含 **《SAM 家族模型演进与核心技术研究报告》** 的实验验证代码与数据。项目旨在通过多层次对比实验，量化评估 SAM 1、MobileSAM、SAM 2 及 SAM 3 在基础架构效率、视频时空一致性以及语义概念理解能力上的演进差异。

## 📂 项目结构 (Directory Structure)

```text
SAM-Family
├── dataset/                 # 实验输入数据
│   ├── exp1/
│   │   └── bus.jpg          # 实验一：静态图像基准测试图
│   ├── exp2/
│   │   └── football.mp4     # 实验二：高动态视频测试序列
│   └── exp3/
│       ├── market_scene.jpg # 实验三：复杂语义场景 (集市)
│       └── street_scene.jpg # 实验三：属性对齐场景 (街景)
├── results/                 # 实验输出结果
│   ├── exp1/                # 基础模型效率对比结果
│   ├── exp2/                # 视频时空一致性结果
│   └── exp3/                # SAM 3 语义提示验证结果 (来自官方Demo)
├── exp1.py                  # 脚本：基础架构参数量与推理效率对比
├── exp2.py                  # 脚本：视频流式处理与记忆机制验证
├── exp3.py                  # 脚本：SAM 3 在线实验引导与结果可视化
└── README.md                # 项目说明文档
```

## 🛠️ 环境准备 (Prerequisites)

请确保您的环境中已安装 Python 3.8+ 及以下依赖库：

```bash
pip install torch torchvision
pip install ultralytics opencv-python matplotlib
```

## 🚀 实验运行指南 (Usage)

### 1. 基础模型架构与效率对比 (Experiment 1)
对比 SAM 1 (Base)、MobileSAM 与 SAM 2 (Tiny/Base) 在单张图像上的零样本分割效果、参数量及推理耗时。

**运行脚本：**
```bash
python exp1.py
```

### 2. 视频时空一致性验证 (Experiment 2)
验证 SAM 2 的流式记忆机制 (Streaming Memory) 相比 SAM 1 在处理视频长程依赖和遮挡时的表现。

**配置与运行：**
请在 `exp2.py` 中修改 `video_path` 为本地视频路径。
```bash
python exp2.py
```

### 3. SAM 3 语义概念理解验证 (Experiment 3)
验证 SAM 3 从“几何分割”向“概念分割”的跨越。由于 SAM 3 权重目前受限，本实验基于 **Meta AI Official Demo** 进行交互式验证。

**实验步骤：**
1. 运行引导脚本：
   ```bash
   python exp3.py
   ```
   *脚本将自动打开浏览器并跳转至 SAM 3 Demo 页面。*
2. **上传图片**：使用 `dataset/exp3/` 目录下的测试图片。
3. **输入提示词**：根据控制台打印的 Prompt（如 "woman wearing red coat"）进行输入。
4. **保存结果**：将生成的分割结果截图保存至 `results/exp3/` 目录。
5. **结果分析**：再次运行脚本可自动并列展示原图与分割结果进行分析。

## 📊 实验结果摘要 (Results Summary)

### 实验一：参数量与效率 (Efficiency)
| 模型名称 | 参数量 (M) | 推理耗时 (ms) | 结论 |
| :--- | :--- | :--- | :--- |
| **MobileSAM** | ~10.1 | Low | 解耦蒸馏策略有效，但牺牲了极细微纹理的捕捉能力。 |
| **SAM 1 (Base)** | ~93.7 | High | 计算负载较重，适合离线处理。 |
| **SAM 2 (Base)** | ~80.8 | Medium | Hiera 架构优化了计算，但在引入记忆机制后耗时有所回升。 |

### 实验二：视频跟踪能力 (Video Tracking)
- **SAM 1**: 表现为逐帧独立预测，存在 ID 频繁切换（闪烁）现象，缺乏时序记忆。
- **SAM 2**: 利用 Memory Bank 机制，在目标发生遮挡并重现后，能保持 ID 一致性（Object Permanence），验证了流式记忆机制的有效性。

### 实验三：语义概念理解 (Semantic Understanding)
- **语义对齐有效性**: SAM 3 能够准确响应 "women wearing sunglasses" 等包含属性（配饰）的指令，证明了多模态特征在潜在空间的有效对齐。
- **感知边界局限**: 在处理 "woman handing a yellow coat" (复杂交互) 或远景小目标时，模型表现出召回率衰减。这佐证了报告中关于“识别-定位”解耦机制在极端场景下仍存在分辨率瓶颈的分析。

## 📝 引用 (Citation)

如果您在研究中使用了本仓库的代码或思路，请参考以下相关文献：

1. Kirillov, A., et al. "Segment Anything." ICCV 2023.
2. Ravi, N., et al. "SAM 2: Segment Anything in Video and Images." arXiv 2024.
3. Carion, N., et al. "SAM 3: Segment Anything with Concepts." arXiv 2025.

---
*Created by Yina Zhuang for the Research Report on SAM Family Evolution.*