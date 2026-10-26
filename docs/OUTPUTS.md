# 输出格式

训练输出路径为`output_dir/seed_<seed>/<experiment>/`。每个实验保存以下文件：

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 逐样本预测及类别概率 |
| `metrics.json` | 各评估集的分类指标 |
| `history.json` | 每轮损失；主训练及LOCO同时记录B1验证指标 |
| `run_manifest.json` | 实验名称、数据卡编号、种子、完整配置、数据和源码SHA-256、父模型记录 |
| `COMPLETED.json` | 任务完成记录 |
| `data_protocol.json` | 主训练的数据划分、VOC标准化参数及波长 |

## 样本预测

`predictions.csv`字段如下。类别编码为H=0、M=1、L=2。

| 字段 | 内容 |
| --- | --- |
| `sample_id` | 输入表中的样本编号 |
| `data_card_ids` | 数据卡结果编号，多个编号以分号连接 |
| `true`、`pred` | 实测类别编码、预测类别编码 |
| `p_H`、`p_M`、`p_L` | 三类softmax概率 |
| `cultivar`、`batch`、`site` | 样本信息 |
| `evaluation` | 评估集名称，如`B1-Test`、`B2-All`、`B3-All`、`B2-Holdout`、`B3-Holdout` |
| `fold` | LOCO输出中的留出品种 |

LOCO根目录保存五折合并预测及`metrics.json`；各品种子目录保存本折`best.pt`、预测和训练记录。

## 指标

`metrics.json`包含`task`、`seed`、`synthetic`和`metrics`。`metrics`以评估集名称为键，每个评估集含`data_card_ids`列表，记录样本数、accuracy、macro precision、macro recall、macro F1、Cohen's kappa、预测为M的数量及H/M/L顺序的3×3混淆矩阵。

## 绘图文件

| 操作 | 输出 |
| --- | --- |
| `figure_visualization.export` | `features.npy`：N×D融合特征；`samples.csv`：同序的sample_id、label、domain；`provenance.json`：模型、样本及特征信息 |
| 指定`--sample-id`的特征导出 | 另存`spectral_attention.npz`，含`wavelengths`、`nir_values`、`attn_weights`、`label`、`sample_id` |
| 混淆矩阵与光谱注意力 | `--out`指定的图片 |
| 辅助空间注意力 | `image_attention.py`生成图片及同名`.provenance.json`，记录Student-Image、seed42、样本和bicubic插值 |
| Fig. 6D（Grad-CAM） | `gradcam.py`生成图片及同名`.provenance.json`，记录样本、模型、seed和目标类别 |
| t-SNE | 包含边缘KDE的图片、同名`.coordinates.csv`和`.settings.json`；椭圆为2σ协方差椭圆 |


## 汇总与资源测量

`analysis.aggregate`要求每个实验包含种子6、7、42，输出：

| 文件 | 内容 |
| --- | --- |
| `seed_metrics.csv` | 每个实验、种子及评估集的指标 |
| `mean_sd.csv` | 三种子指标均值及样本标准差，`ddof=1` |
| `welch_tests.csv` | B2上BCA与DeepCORAL、DANN的Welch检验 |

`profile.json`记录模型参数量、state_dict文件大小，以及batch=1和16时的前向延迟、重复测量标准差、吞吐量、CUDA峰值显存和FLOPs。测量范围为模型前向传播，分别报告每批次和每样本FLOPs。

光谱绘图可读取已有NPZ。必需字段为`wavelengths`、`nir_values`、`attn_weights`、`label`，`sample_id`为可选字段；曲线使用文件内的权重值。
