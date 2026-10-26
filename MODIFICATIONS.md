# BCANet10.16 修改说明

本次按精简版修改Guide整理代码包，并采用“非商业科研自动许可、商用另行授权”的政策。

| 文件 | 修改内容 |
| --- | --- |
| `README.md` | Fig. 6D改用实际存在的`figure_visualization.gradcam`入口；补充seed42、预测类别和指定类别说明；增加预处理用法、数据范围及许可摘要 |
| `docs/PAPER_MAPPING.md` | 补回README引用但原包缺失的图表映射文件，使用本包现有目录名称 |
| `docs/OUTPUTS.md` | 将Fig. 6D的Grad-CAM与辅助空间注意力分别说明 |
| `figure_visualization/image_attention.py` | 修改文件说明和错误提示，将其定位为辅助空间注意力工具 |
| `tools/U2-Net-processImage.py` | 加入清理后的预处理脚本；路径统一为`**`；删除个人目录、旧项目名称和错误批次说明；显式指定`u2net`并复用会话 |
| `requirements-preprocess.txt` | 新增可选的rembg CPU后端、OpenCV和tqdm依赖 |
| `LICENSE` | 新增简短非商业科研许可；科研免费且无需申请，商业使用需书面授权，保留第三方许可和法定权利 |
| `MODIFICATIONS.md` | 本修改清单 |

## 预处理流程

保留非透明区域大于1000像素的判定、外接矩形裁剪、黑底RGB输出及失败隔离流程；处理`.jpg`、`.png`图像。脚本末尾的两个目录需由使用者填写。

## 保留的训练实现

网络结构、CORAL与其他损失、Mixup、训练参数、数据划分、LOCO、评估指标以及Grad-CAM计算逻辑均沿用上传包。

## 许可使用方式

非商业科研无需联系作者；分享代码时保留许可与版权声明，发表相关研究成果时引用对应论文。商业用途请联系论文通讯作者或仓库维护者。原始数据不随代码包提供。

## 检查结果

78个Python文件的语法检查通过；README命令入口与文档链接检查通过。此次未重新运行模型训练或真实图像分割。
