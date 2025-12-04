# Statistical Fairness of Gaokao Tiered-Scaling: A Multi-Model Monte Carlo Simulation
# 高考等级赋分制度统计公平性：多模型蒙特卡洛模拟研究

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**English** | **[中文](#中文)**

---

## English

### 📖 Project Overview
This repository contains the complete replication package for the academic research on the statistical fairness of **China's National College Entrance Examination (Gaokao) Tiered-Scaling System**.

> **Note on Terminology:** The term **"Scaling"** (or **"Tiered-Scaling"**) in this context is a direct translation of the Chinese term **"赋分" (fùfēn)**. It refers to the specific policy of converting raw subject scores into standardized "tiered scores" based on a test-taker's percentile rank within the subject cohort, to ensure comparability across different elective subjects.

### 🔬 Research Abstract
The study employs **Monte Carlo simulation** to quantitatively evaluate the design of the current scoring system. It compares four models to assess trade-offs between **inter-subject fairness, score discrimination, and system robustness**.

### 🗂️ Repository Structure# statistical-fairness-gaokao-scaling-simulation
.
├── score.py # Basic simulation & validation
├── score_3models.py # Four-model comparison (Produces key Table 1)
├── score_final_analysis.py # Final analysis with stress tests (Produces Table 2)
├── *.png # All generated analysis figures
└── README.md # This file


### 🚀 Getting Started
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/huangmei-minming/statistical-fairness-gaokao-scaling-simulation.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib scipy
    ```
3.  **Run the analysis (in order):**
    ```bash
    python score.py
    python score_3models.py
    python score_final_analysis.py
    ```

### 📊 Key Findings (Simulated)
*   **Fairness Achieved:** The tiered-scaling system eliminated the initial 5.04-point mean difference between simulated Chemistry and Geography subjects.
*   **High Discrimination Preserved:** The standard deviation of scaled scores (14.92) was 99.5% of the theoretical optimum (Z-score model).
*   **System Robustness:** Stress tests ("high-density" and "bimodal" distributions) demonstrated the system's stable output variance.

### 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments
*   Real-world score distribution parameters were informed by data from provincial education examination authorities.
*   The research design and interpretation were conducted independently. AI programming assistants were utilized for code implementation.

---

## 中文

### 📖 项目概述
本仓库包含一项关于 **中国高考等级赋分制度** 统计公平性学术研究的完整复现资源包。

> **术语说明：** 本研究中的英文术语 **"Scaling"** 或 **"Tiered-Scaling"**，是中文 **"赋分"** 的直译。它特指一项将选考科目的原始卷面分，依据考生在该科目内的排名百分比，转换为标准化的"等级分"的政策，旨在实现不同选考科目成绩之间的可比性。

### 🔬 研究摘要
本研究采用 **蒙特卡洛模拟** 方法，对现行赋分制度的设计进行量化评估。通过对比四个计分模型，分析了其在 **跨科目公平性、分数区分度和系统稳健性** 之间的权衡。

### 🗂️ 仓库文件结构
仓库根目录包含核心分析文件：
*   `score.py`: 基础赋分模拟与验证
*   `score_3models.py`: 四模型对比分析（生成论文核心表1）
*   `score_final_analysis.py`: 包含压力测试的最终分析（生成压力测试表2）
*   `*.png`: 所有生成的分析图表
*   `README.md`: 本说明文件

### 🚀 快速开始
1.  **克隆仓库：**
    ```bash
    git clone https://github.com/huangmei-minming/statistical-fairness-gaokao-scaling-simulation.git
    ```
2.  **安装依赖：**
    ```bash
    pip install numpy pandas matplotlib scipy
    ```
3.  **按顺序运行分析脚本：**
    ```bash
    python score.py
    python score_3models.py
    python score_final_analysis.py
    ```

### 📊 核心发现（模拟结果）
*   **实现公平性：** 等级赋分制度完全消除了模拟中化学与地理科目之间5.04分的原始均分差。
*   **保持高区分度：** 赋分后的成绩标准差（14.92）达到了理论最优模型（Z分数模型）的99.5%。
*   **系统稳健性：** 通过"高分密集"与"两极分化"压力测试，证明了系统输出方差的稳定性。

### 📜 许可证
本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

### 🙏 致谢与说明
*   模拟所参考的现实分数分布参数来源于各省教育考试院公开数据。
*   本研究的设计、分析与结论由研究者独立完成。在代码实现阶段使用了AI编程助手进行辅助。

---
