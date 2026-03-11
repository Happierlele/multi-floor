# 🚀 HPC 运行指南 (Run Guide)

本指南介绍如何在 HPC (High Performance Computing) 集群上运行本项目。

## 1. 交互式调试 (Interactive Mode)
如果你需要**实时调试代码**或**短暂运行**（例如查看是否报错），请申请交互式节点。

**命令 (复制运行):**
```bash
qsub -I -q interactive_gpu -l select=1:ncpus=2:ngpus=1:mem=10gb -l walltime=00:30:00 -P Personal
```
*参数说明:*
*   `-I`: Interactive (交互式)
*   `-q interactive_gpu`: 队列名称 (根据你的集群调整，如 `gpu` 或 `debug`)
*   `-l select=...`: 资源申请 (1个节点, 2核CPU, 1张GPU, 10GB内存)
*   `-l walltime=...`: 申请时长 (30分钟)

**进入节点后:**
```bash
# 1. 进入项目目录
cd $PBS_O_WORKDIR  # 或者手动 cd 到你的项目路径

# 2. 激活环境
source activate habitat  # 或者 conda activate habitat

# 3. 运行代码
python driver.py
```

---

## 2. 批量作业提交 (Batch Job)
如果你需要**长时间训练**（例如跑一整天），请使用 `qsub` 提交脚本。

**核心脚本:** `train_job.pbs` (推荐) 或 `run.pbs`

**提交命令:**
```bash
qsub train_job.pbs
```

**常用命令:**
*   **查看作业状态**: `qstat -u $(whoami)`
*   **查看实时日志**: `tail -f training_log_*.txt` (脚本会自动生成这个日志文件)
*   **删除作业**: `qdel <JOB_ID>` (例如 `qdel 12345.pbs`)

---

## 3. 关键文件说明

| 文件名 | 用途 | 备注 |
| :--- | :--- | :--- |
| **`train_job.pbs`** | **主要训练脚本** | 包含环境配置、GPU检测、日志重定向。推荐使用。 |
| `run.pbs` | CPU/软件渲染测试 | 强制使用软件渲染，适合没有 GPU 的节点测试。 |
| `run_job.sh` | 备用脚本 | 简单的 Shell 脚本模板。 |
| `driver.py` | **程序入口** | 主程序，启动 RL 训练或评估。 |

## 4. 常见问题 (Troubleshooting)

**Q: 报错 `No module named 'habitat'`**
*   **解法**: 检查环境是否激活。在脚本中添加 `source activate habitat`。

**Q: 报错 `EGL_NOT_INITIALIZED` 或黑屏**
*   **解法**: 确保使用了 GPU 节点 (`ngpus=1`)，且脚本中设置了 `export MAGNUM_TARGET_HEADLESS=1`。`train_job.pbs` 已经配置好了这些。

**Q: 作业跑一会就断了 (Walltime Exceeded)**
*   **解法**: 修改 `.pbs` 文件中的 `#PBS -l walltime=24:00:00`，增加时间（例如 `48:00:00`）。
