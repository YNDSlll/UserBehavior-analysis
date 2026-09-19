# 电商用户行为分析项目

## 项目概述
基于阿里天池 UserBehavior 数据集（采样 100 万行）的用户行为分析：
清洗 → 漏斗与 EDA → RFM + K-Means 分层 → 流失预测（逻辑回归 / 随机森林）→ Streamlit 展示。

## 快速开始

### 0. 在线演示（无需安装任何东西）

浏览器直接打开：https://userbehavior-analysis.streamlit.app/   （Streamlit Cloud 托管）

### 1. 创建环境并安装依赖

建议使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 的激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 直接运行看板

仓库已提供 `data/processed/` 和 `output/app_data/` 中的演示数据，安装依赖后即可运行：

```bash
streamlit run app.py
```

浏览器打开 [`http://localhost:8501`](http://localhost:8501/)。

项目总结报告.md在根目录下。

### 3. 从原始数据复刻全流程

原始文件约 3.4 GB，没有提交到 GitHub。请从阿里天池下载[数据集](https://tianchi.aliyun.com/dataset/649) `UserBehavior.csv`，放到 `data/raw/`，然后按以下顺序运行：

1. `notebooks/M0_数据读取.ipynb`：读取原始数据并生成样例数据。
2. `notebooks/M1_数据清洗.ipynb`：清洗数据并生成 `data/processed/UserBehavior_clean.csv`。
3. `notebooks/M2_EDA与转化漏斗.ipynb`：生成 EDA 和转化漏斗结果。
4. `notebooks/M3_RFM用户分层.ipynb`：生成 `data/processed/UserBehavior_rfm.csv`。
5. `notebooks/M4_流失预测建模.py`：生成特征表和模型结果。
6. `notebooks/M5_准备演示数据.py`：生成 `output/app_data/` 中的看板数据。

运行 Python 脚本时请保持当前目录为项目根目录：

```bash
python notebooks/M4_流失预测建模.py
python notebooks/M5_准备演示数据.py
```

原始数据目录已通过 `.gitignore` 忽略，避免误提交大文件；样例数据、处理后数据和看板预计算数据已保留，便于直接查看项目效果。

## 目录结构
```
UserBehavior/
├── README.md     项目说明与复刻指南
├── requirements.txt  Python 环境依赖
├── app.py        Streamlit 可视化看板入口
├── 项目总结报告.md  项目总结报告
├── guides/       分析思路整理（本地使用，不上传 GitHub）
├── data/
│   ├── raw/        原始数据 UserBehavior.csv、UserBehavior.csv.zip（本地保存，不上传）
│   ├── sample/     采样数据 UserBehavior_1m.csv（前 100 万行）
│   └── processed/  清洗后 UserBehavior_clean.csv、分层 UserBehavior_rfm.csv、特征表 UserBehavior_features.csv
├── notebooks/      分析代码
│   ├── M0_数据读取.ipynb
│   ├── M1_数据清洗.ipynb
│   ├── M2_EDA与转化漏斗.ipynb
│   ├── M3_RFM用户分层.ipynb
│   ├── M4_流失预测建模.py
│   └── M5_准备演示数据.py
└── output/
    ├── app_data/   看板预计算数据
    ├── charts/     图表
    └── reports/    分析报告
```

### 分析代码命名
| 阶段 | 文件 | 内容 |
|---|---|---|
| M0 | `notebooks/M0_数据读取.ipynb` | 数据读取与抽样 |
| M1 | `notebooks/M1_数据清洗.ipynb` | 数据清洗与预处理 |
| M2 | `notebooks/M2_EDA与转化漏斗.ipynb` | 探索性分析与转化漏斗 |
| M3 | `notebooks/M3_RFM用户分层.ipynb` | RFM 指标与 K-Means 聚类 |
| M4 | `notebooks/M4_流失预测建模.py` | 流失定义与预测建模 |
| M5 | `notebooks/M5_准备演示数据.py` | 生成看板预计算数据 |
| 应用 | `app.py` | Streamlit 可视化看板入口 |

## 里程碑进度
- [x] M0 环境搭建 + 数据到手（采样 100 万行，ID 转字符串）
- [x] M1 数据清洗与预处理（剔除时间范围外脏数据 469 行；清洗后 999531 行；完成重复值检查和字段校验）
- [x] M2 探索性分析 + 转化漏斗（人均 102.6 条行为；活跃高峰 19-22 点；完成用户口径转化漏斗，结论落盘 `output/reports/M2_EDA结论.md`）
- [x] M3 RFM + K-Means 用户分层（M 用购买商品种类数替代金额；通过肘部图和轮廓系数选择 K=4；完成四类用户画像）
- [x] M4 流失定义 + 预测模型（前 6 天构造特征，后 3 天定义标签；随机森林 AUC 0.736，结果见 `output/reports/M4_建模结果.md`）
- [x] M5 报告 + 可视化（完成 Streamlit 看板、项目总结报告、和预计算数据）


## 学习记录
- 项目启动，完成路线图、数据准备和项目架构搭建。
- M0：使用 `read_csv`（`header=None`、`names`、`nrows=1000000`）完成采样；ID 列转为字符串；行为分布为 pv 89.6%、cart 5.5%、fav 2.8%、buy 2.0%。
- M1：完成 Unix 时间戳转北京时间、时间范围检查和重复值检查；剔除官方时间窗口外的脏数据；结果存入 `data/processed/UserBehavior_clean.csv`。
- 关键结论：数据窗口仅 9 天，M4 不适合直接使用“30 天未活跃”定义，需要根据观察窗口重新设计流失口径。
- M3：R 使用购买口径，M 使用购买商品种类数替代金额；肘部图显示 K=4~5 较合适，轮廓系数选择 K=4；完成高价值忠诚、早期沉睡、活跃潜力和超级用户四类画像；结果存入 `data/processed/UserBehavior_rfm.csv`。
- M3 方法局限：超级用户离群簇容易被 K-Means 吸引；F/M 相关系数较高，价值维度主要由综合购买强度驱动。
- M4：将观察期最后 3 天无购买定义为流失，特征只使用前 6 天构造，标签使用后 3 天生成，避免数据穿越。
- M4：完成 pv/cart/fav/buy 计数、活跃天数、类目宽度和最近行为等特征；随机森林优于逻辑回归；产出 `data/processed/UserBehavior_features.csv`、特征重要性图和建模报告。
- M5：完成预计算看板数据、Streamlit 可视化应用、项目总结报告；看板入口为 `app.py`。
