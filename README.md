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
| 2026-10-24 | logreg | 0.7939 | ✅ | — | solver=newton-cholesky，收敛警告消除（-0.0003 在噪声内；分支 iter/tuning） |
| 2026-10-21 | xgboost | 0.8119 | ✅ | — | 默认参数（300 树，lr=0.05） |
| 2026-10-21 | lightgbm | 0.8102 | ✅ | — | 默认参数（300 树，lr=0.05） |
| 2026-10-21 | hist_gb | **0.8123** | ✅ | **889 / 1726**（2026-10-24 首投） | 全默认参数；确定性通过 |

## 迭代记录（2026-10-24，本地 CV 协议：5 折分层、seed=42、n_jobs=1）

**第一轮·参数扫描（16 配置）**：hist_gb / xgboost / lightgbm 参数网格 + 三模型软投票，结果 0.8032~0.8121，**无一超过基线 hist_gb 0.8123**——参数已饱和，波动均在 ±0.005 噪声带内。

**第二轮·特征消融（单因素归因）**：在 32 特征基线上逐项添加——pp_num(组内序号) 0.8111、组聚合 spend/age 0.8097、age_bin+deck_side 0.8102、surname_freq 0.8115、客舱组填补 0.8110、全部附加 0.8097、全部+客舱组填补 0.8098。**全部中性偏负**。

**结论**：当前特征集 + 单一 GBDT 已在 ~0.812 饱和；CV 单种子噪声约 ±0.005，任何小于此幅度的"提升"不可信。后续迭代应转向：特征**选择**（删弱而非加新）、更聪明的填补分布、集成/堆叠、以及用 10 折/重复 CV 降噪声后再比较。多分支并行迭代已启动（见各 `iter/*` 分支）。

**第三轮·降噪声与真伪甄别（分支 iter/tuning，2026-10-24）**：对第一轮排名做 5×5 重复 CV（共 25 折配对，seed=42、n_jobs=1）。

- **头部排名全是噪声**：hist_gb/xgboost 各变体 vs hist_gb 默认，配对差 |Δ|≤0.0013、|t|≤1.2、胜场 10~14/25——第一轮 0.8113~0.8123 的名次不可分辨；
- **真差异**：logreg 真差（-0.017，t=-11.8）；lgbm 500 树×63 叶是真降分（-0.007，t=-6.1，大树数过拟合）；lgbm 1000×0.03 边缘真差（-0.003，t=-2.7）；
- **噪声刻度**：折级 σ≈0.009，同折配对 σ≈0.005，与 ±0.005 噪声带设定吻合；标准单切 5 折比重复 CV 均值系统性偏高 ~0.001-0.002（该切分偏乐观）；
- **early stopping**（eval_set 在 CV 折内部切，防泄漏）：xgb d5/d4 大树数固定配置确实过拟合（500/800 树掉到 0.808），ES 自动停在最优树数区（d6≈220、d5≈315、lgbm lr0.05≈145、lr0.03≈210）可恢复到第一梯队，但不超 hist_gb 默认；
- **top 邻域细网格**（42 配置，10 折筛选→头部 5×5 复核）：筛选第一名 hgb_80_0.08 复核后配对差 -0.0013（回归均值，选择效应）；lgbm 250 树×0.04 对 lgbm 默认 +0.0018（t=2.05），低于 +0.003 采用线，**不落 config**；
- **logreg 修复**：lbfgs 因特征未标准化在 max_iter 3000/5000 仍不收敛；改 solver=newton-cholesky（仅改 config 值，二阶法对量纲不敏感）→ 零警告，CV 0.7942→0.7939（噪声内）。

**第三轮结论**：GBDT 默认参数已在重复 CV 意义下最优（~0.811），维持 hist_gb 全默认为最优单模型；调参方向正式关闭，收益只能来自特征/填补/集成轴。

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
