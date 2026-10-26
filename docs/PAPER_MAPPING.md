# 论文图表与代码对应

## 训练实验

| 论文位置 | 代码入口或实验名称 |
| --- | --- |
| Fig. 1 教师与学生结构 | `bca_net/model.py` |
| Table 2 对比基线 | `table2_baselines/`；`python run.py --config "**" --experiment table2` |
| Table 3 模块消融 | `table3_ablations/`；`python run.py --config "**" --experiment table3` |
| Table 4 教师、学生与泛化实验 | `table4_protocols/`；`python run.py --config "**" --experiment table4` |

## 绘图与资源测量

| 论文位置 | 脚本 | 模型或输入 |
| --- | --- | --- |
| Fig. 6A | `figure_visualization/confusion.py` | A09/B03，seed42，B1-Test逐样本预测 |
| Fig. 6B | `figure_visualization/export.py`、`spectral_attention.py` | A09/B03，对应样本的光谱及注意力 |
| Fig. 6C | `figure_visualization/confusion.py` | A09/B03，seed42，B2-All逐样本预测 |
| Fig. 6D | `figure_visualization/gradcam.py` | Student-Image，seed42，Grad-CAM |
| Fig. 6E | `figure_visualization/export.py`、`tsne.py` | B02，B1 Test与B2 All融合特征 |
| Fig. 6F | `figure_visualization/export.py`、`tsne.py` | A09/B03，B1 Test与B2 All融合特征 |
| Table S4 | `supplementary/profile.py` | 学生模型参数量、FLOPs及前向耗时 |
