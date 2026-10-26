# BCA-Net

菊花害虫易感性评估的训练、评价和绘图代码，包括BCA-Net、对比基线、消融实验及泛化实验。

## 目录

| 目录 | 内容 |
| --- | --- |
| `bca_net` | 教师和学生网络，对应Fig. 1 |
| `table2_baselines` | Table 2对比基线 |
| `table3_ablations` | Table 3消融实验 |
| `table4_protocols` | Table 4泛化实验 |
| `figure_visualization` | Fig. 6绘图与特征导出 |
| `supplementary` | Table S4资源测量 |
| `common` | 数据处理、训练、损失函数和指标 |
| `analysis` | 三种子结果汇总和Welch检验 |
| `tools` | 叶片图像预处理、环境检查、Excel转换和测试数据生成 |

表图与实验名称见[实验对应表](docs/PAPER_MAPPING.md)，参数见[训练参数](docs/PARAMETERS.md)。

## 安装

使用Python 3.11或3.12。GPU环境按[PyTorch安装说明](https://pytorch.org/get-started/locally/)选择配套的PyTorch和torchvision版本。

```bash
python -m pip install -r requirements.txt
python tools/check_environment.py
```

图像骨干默认使用ConvNeXt-Base及ImageNet预训练权重，首次运行会通过timm下载权重。

## 数据与配置

请按[输入格式](docs/DATA_SCHEMA.md)准备本地数据。随机数据仅用于软件流程测试。

所有需填写的路径以字面值`**`表示。将每个`"**"`分别替换为所需文件或目录的本地路径。

按[输入格式](docs/DATA_SCHEMA.md)准备四个CSV文件和叶片RGB图像。将`configs/paper.json`复制为`configs/local.json`，设置数据目录和输出目录：

```json
{
  "data_dir": "**",
  "output_dir": "**",
  "loco_epochs": 60,
  "loco_lr": 0.0004,
  "loco_weight_decay": 0.05
}
```

相对路径从项目根目录解析。检查输入：

```bash
python run.py --config "**" --check-data
```

### 叶片图像预处理

图像预处理使用`tools/U2-Net-processImage.py`，显式调用U²-Net模型，输出黑底RGB图像；失败图像另存到隔离目录。训练清单仅收录成功输出。

```bash
python -m pip install -r requirements-preprocess.txt
python tools/U2-Net-processImage.py
```

运行前将脚本末尾的`INPUT_ROOT`、`OUTPUT_ROOT`中的`**`替换为本地目录。输入目录下按类别设置子目录；脚本处理其中的`.jpg`和`.png`图像。输出按非透明区域裁剪，训练数据加载器随后将图像调整为224×224。

## 训练

在项目根目录执行：

```bash
python run.py --config "**" --experiment B03
python run.py --config "**" --experiment table2
python run.py --config "**" --experiment table3
python run.py --config "**" --experiment table4
```

默认运行种子6、7、42；使用`--seed 42`指定单个种子。`--experiment all`运行全部实验。蒸馏任务自动训练或加载同种子的教师。

各实验文件夹的`run.py`可直接运行。在PyCharm运行参数中设置`--config`，例如`--config "**"`。共同参数由`common/config.py`和所选JSON配置读取。

`--reuse`复用配置、源码、数据和父模型记录一致的已完成任务。更改配置或源码后，为新实验设置新的`output_dir`。

## 绘图

以下绘图命令使用种子42的输出。混淆矩阵与UDA特征使用A09/B03模型；Source-only特征使用B02模型；图6D使用C14/C15模型。`SAMPLE_ID`填写输入表中的真实样本编号。

```bash
# Fig. 6A：A09/B03，B1 Test
python -m figure_visualization.confusion --predictions "**" --evaluation B1-Test --out "**"
# Fig. 6C：A09/B03，B2 All
python -m figure_visualization.confusion --predictions "**" --evaluation B2-All --out "**"
# Fig. 6B：A09/B03，导出指定样本
python -m figure_visualization.export --config "**" --checkpoint "**" --sample-id SAMPLE_ID --out "**"
python -m figure_visualization.spectral_attention --input "**" --out "**"
# Fig. 6D：Student-Image，seed 42，Grad-CAM
python -m figure_visualization.gradcam --config "**" --checkpoint "**" --sample-id SAMPLE_ID --out "**"
# Fig. 6E：B02，导出Source-only特征
python -m figure_visualization.export --config "**" --checkpoint "**" --out "**"
# Fig. 6E：输入B02导出的特征目录
python -m figure_visualization.tsne --input "**" --out "**"
# Fig. 6F：输入A09/B03导出的特征目录
python -m figure_visualization.tsne --input "**" --out "**"
```

特征默认导出B1 Test与B2 All；`--b1-scope`可选择`train`、`val`或`test`。Fig. 6D使用Student-Image（seed 42）的Grad-CAM。`image_attention.py`保留为辅助空间注意力可视化工具。

## 汇总与资源测量

```bash
python -m analysis.aggregate --runs "**" --out "**"
python -m supplementary.profile --config "**" --out "**"
```

输出文件和字段见[输出格式](docs/OUTPUTS.md)。

## 测试

```bash
python -m unittest discover -s tests -v
python -m tools.make_synthetic_inputs --out "**"
python run.py --config "**" --experiment all --seed 42
```

测试数据由固定随机种子生成，配置采用ConvNeXt-Atto、32维嵌入和1轮训练。

## 使用许可

非商业科研可免费使用、修改和分享本代码，无需事先申请；分享时保留许可和版权声明，发表相关研究成果时引用对应论文。商业使用须另行取得书面授权，请联系论文通讯作者或仓库维护者。完整条款见[LICENSE](LICENSE)。第三方组件和模型权重适用各自许可。
