# Spaceship Titanic · Python 期末大作业

Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) 二分类竞赛的课程小组工程：
根据乘客记录预测其是否被传送至异次元（`Transported` = True / False），评价指标 **Accuracy**。

> 工程约定：模块化、接口统一、配置与随机种子集中管理、可复现。
> 架构与协作契约见仓库内三份文档：**《架构设计.md》《统一接口定义.md》《任务分工表.md》**——接口与契约的修改必须先改文档、双方确认、再改代码。

## 项目状态

- ✅ M0 骨架完成：包结构、`config.py` / `utils.py`、CI、README；
- ✅ M1 数据轴（loader / preprocessing / features，PR 见提交历史）与模型轴（train / predict / pipeline / main，PR #1）均已合入；
- ✅ 端到端联调通过：四个注册模型均可 `python main.py --model <名>` 一键出提交文件，落盘校验全绿；
- 🚧 待办：tests/（测试负责人 T1–T4）、Kaggle 提交与分数记录（A4）、模型调参/集成（M2 可选）。

## 实验记录（5 折 CV Accuracy，seed=42）

| 日期 | 模型 | CV Accuracy | 提交文件 | 榜上分数 | 备注 |
|---|---|---|---|---|---|
| 2026-10-21 | logreg | 0.7942 | ✅ | — | 基线；lbfgs 有收敛警告（max_iter=1000 未加标准化） |
| 2026-10-21 | xgboost | 0.8119 | ✅ | — | 默认参数（300 树，lr=0.05） |
| 2026-10-21 | lightgbm | 0.8102 | ✅ | — | 默认参数（300 树，lr=0.05） |
| 2026-10-21 | hist_gb | **0.8123** | ✅ | **889 / 1726**（2026-10-24 首投） | 全默认参数；确定性通过 |

## 迭代记录（2026-10-24，本地 CV 协议：5 折分层、seed=42、n_jobs=1）

**第一轮·参数扫描（16 配置）**：hist_gb / xgboost / lightgbm 参数网格 + 三模型软投票，结果 0.8032~0.8121，**无一超过基线 hist_gb 0.8123**——参数已饱和，波动均在 ±0.005 噪声带内。

**第二轮·特征消融（单因素归因）**：在 32 特征基线上逐项添加——pp_num(组内序号) 0.8111、组聚合 spend/age 0.8097、age_bin+deck_side 0.8102、surname_freq 0.8115、客舱组填补 0.8110、全部附加 0.8097、全部+客舱组填补 0.8098。**全部中性偏负**。

**结论**：当前特征集 + 单一 GBDT 已在 ~0.812 饱和；CV 单种子噪声约 ±0.005，任何小于此幅度的"提升"不可信。后续迭代应转向：特征**选择**（删弱而非加新）、更聪明的填补分布、集成/堆叠、以及用 10 折/重复 CV 降噪声后再比较。多分支并行迭代已启动（见各 `iter/*` 分支）。

## 快速开始

环境要求：Python 3.14（Windows x64；语法基线 ≥3.10）、git。

```bash
# 1. 克隆（私有仓库，需先被邀请为 Collaborator，见《组员接入指南.md》）
git clone https://github.com/WYQ010/python_final.git
cd python_final

# 2. 安装依赖（国内镜像加速；跑测试另需 pytest）
python -m pip install -r requirements.txt pytest -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 获取比赛数据（数据不入库；两种方式任选，三个文件放入 data/，名称不变）
#    a) Kaggle CLI（需先配置 ~/.kaggle/kaggle.json 并在比赛页接受规则）
kaggle competitions download -c spaceship-titanic
#       将下载的 zip 解压到 data/
#    b) 浏览器：比赛页 https://www.kaggle.com/competitions/spaceship-titanic/data
#       点 Download All，解压 train.csv / test.csv / sample_submission.csv 到 data/

# 4. 端到端运行（训练 + 5 折 CV + 生成提交文件）——待 M1 合并后可用
python main.py
```

输出位置：提交文件 → `outputs/submissions/`，模型 → `outputs/models/`，CV 报告 → `outputs/reports/`。

## 运行测试

```bash
pytest    # 契约测试用合成数据，无需 data/（CI 同款）；E2E 测试需要 data/，缺失时自动跳过
```

## 目录结构（摘要）

```
├── spaceship/    源码包：config / utils / data_loader / preprocessing / features /
│                 train / predict / pipeline（后六个模块 M1 交付）
├── tests/        契约测试 + 端到端测试（M1 交付）
├── main.py       CLI 入口（M1 交付）
├── data/         比赛数据（不入库，自取）
└── outputs/      全部生成物（不入库，运行时自动创建）
```

## 协作约定（摘要）

- **Git 流程**：`main` 受保护；feature 分支开发 → 发 PR → **至少一人 review** → 合并。逐步操作见《组员接入指南.md》；
- **commit message**："模块: 一句话" 简式，如 `features: 拆分 deck/side`；
- **禁止入库**：`data/`、`outputs/`、zip 与过程文档均已被 `.gitignore` 排除（Kaggle 数据条款 + 隐私），请勿 `git add -f` 强行加入；
- **环境**：Python 3.14.0，依赖一律钉版本（`requirements.txt`）；`kaggle` CLI 仅取数用，不属于工程依赖。

## 指标参考（非承诺）

全猜多数类 ≈ 0.50；逻辑回归基线 ≈ 0.75~0.78；特征工程 + GBDT + 集成 ≈ 0.79~0.81。
