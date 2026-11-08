# djr_tinygpt_perf —— TinyGPT 性能工程实战(Task 0-3)工作区

> 作者:djr(group 22) · 2026-11-09
> 本目录是 **完全独立的新增目录**,不修改 tinytorch 课程源码的任何文件;
> 仅通过 `import tinytorch.*` 调用课程导出的包(模块 01-16 已按课程流程
> `tito dev export --all` 导出,导出产物属于课程标准工作流)。

## 目录布局

```
djr_tinygpt_perf/
├── common.py                  # 四个 Task 共用:模型构建/训练管线/权重缓存/评估
├── task0_baseline/            # Task 0 基准(issue #29)
│   ├── pytest_tensor.log          # 模块01测试通过记录(32 passed)
│   ├── pytest_transformers.log    # 模块13测试通过记录(42 passed)
│   ├── milestone05_train.log      # milestone 05 官方脚本完整输出(🏆 ACHIEVED)
│   ├── train_and_curve.py         # 自采逐epoch曲线 + bigram判据复算
│   ├── loss_curve.png / .csv      # 训练损失曲线
│   ├── train_summary.json         # 判据结果汇总(双PASS)
│   └── tinygpt_weights.npz        # 基线FP32权重(Task1/2/3复用同一基线)
├── task1_profiling/           # Task 1 Profiling(issue #30)
│   ├── run_profiling.py           # 算子计时/内存热力图/FLOPs 计数
│   ├── op_time_share.png          # 算子耗时占比(Prefill vs Decode)
│   ├── memory_heatmap.png         # 逐层内存热力图(权重/梯度/激活)
│   ├── flops_table.csv            # 逐层 FLOPs + 算术强度 + 瓶颈标签
│   └── profiling_summary.json
├── task2_quantization/        # Task 2 Quantization(issue #31)
│   ├── run_quantization.py        # 对称/非对称PTQ + 大小/精度/速度对比
│   ├── quant_error_table.csv      # 逐层量化误差(对称 vs 非对称)
│   ├── quant_error.png / size_compare.png / speed_compare.png
│   ├── speed_table.csv            # 逐层 Prefill/Decode 实测
│   └── quantization_summary.json
├── task3_compression/         # Task 3 Compression(issue #32)
│   ├── run_pruning.py             # 幅度/结构化剪枝 + 掩码微调 + 物理裁剪
│   ├── pruning_table.csv          # 30%/50%/70% 精度损失与微调恢复
│   ├── pruning_accuracy.png       # 精度 vs 剪枝率曲线
│   ├── pruning_summary.json
│   └── run_output.log             # 运行完整输出
├── task4_acceleration/        # Task 4 Acceleration(issue #33)
│   ├── run_acceleration.py        # 融合算子(fused GELU/add+LN)+ 微基准 + 模型级 ITL/Prefill
│   ├── fusion_micro.png           # 逐算子融合微基准(同公式公平对照)
│   ├── fusion_model.png           # 模型级 Decode vs Prefill 收益
│   └── fusion_summary.json
├── reports/                   # (预留)
└── REPORT.md                  # 总报告:原理 + 结论(打卡文字直接引用)
```

## 环境(与共享环境完全隔离)

tinytorch 要求 `numpy>=2.2.6`,与共享 hw1 venv(numpy 1.26.4,为兼容 NGC
torch 钉版)冲突,因此建了独立 venv:

```bash
python3 -m venv /user-data/djr/env-tito
/user-data/djr/env-tito/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    'numpy>=2.2.6' rich PyYAML certifi pytest matplotlib 'nbdev>=3.0.15,<3.0.16'
cd /user-data/djr/cs249r_book/tinytorch
/user-data/djr/env-tito/bin/pip install -e .
/user-data/djr/env-tito/bin/tito dev export --all   # 模块 -> tinytorch/core + perf
```

**计算资源:全程纯 NumPy CPU,未占用任何 GPU**(TinyTorch 本身不依赖 CUDA),
对组内 2/3 号卡与其他组零影响。

## 复现命令

```bash
cd /user-data/djr/cs249r_book/tinytorch
T=/user-data/djr/env-tito/bin/python
$T -m pytest tests/01_tensor/test_tensor_core.py -q          # Task0 测试
$T djr_tinygpt_perf/task0_baseline/train_and_curve.py        # Task0 训练+判据
$T milestones/05_2017_transformer/01_tinygpt_shakespeare.py  # 官方 milestone
$T djr_tinygpt_perf/task1_profiling/run_profiling.py         # Task1
$T djr_tinygpt_perf/task2_quantization/run_quantization.py   # Task2
$T djr_tinygpt_perf/task3_compression/run_pruning.py         # Task3
```
