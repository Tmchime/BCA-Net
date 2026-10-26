# 输入格式

`data_dir`目录包含`manifest.csv`、`nir.csv`、`voc.csv`、`wavelengths.csv`及叶片RGB图像。CSV采用UTF-8编码。

## manifest.csv

每行对应一个样本，列名如下：

| 字段 | 类型与取值 |
| --- | --- |
| `sample_id` | 全局编号，字符串 |
| `plant_id` | 全局编号，字符串|
| `batch` | `B1`、`B2`或`B3` |
| `cultivar` | 品种缩写 |
| `site` | 采样地点 |
| `nitrogen` | 氮处理名称|
| `label` | `H`、`M`、`L`，表示高、中、低易感性 |
| `image_path` | RGB图像路径 |

图像、NIR和VOC通过`sample_id`连接。

## nir.csv

列顺序为`sample_id,band_000,...,band_498`，包含B1、B2和B3的全部样本。波段列共499列，按`wavelengths.csv`的波长顺序排列。波段列也可使用其他名称，读取顺序为CSV列顺序。

读取后执行Savitzky-Golay平滑，窗口9、阶数5，再对每条光谱执行SNV，标准差使用`ddof=1`。

## voc.csv

B1样本：

| 列名 | 内容 |
| --- | --- |
| `sample_id` | B1样本编号 |
| `beta_caryophyllene` | beta-caryophyllene特征值 |
| `camphene` | camphene特征值 |
| `gamma_elemene` | gamma-elemene特征值 |
| `propyl_2_butenoate` | propyl 2-butenoate特征值 |

VOC标准化的均值和标准差由B1训练集计算，再应用于B1全部样本。

## wavelengths.csv

包含一列`wavelength_nm`，共499行光谱波长，范围为1550-1950 nm。

## 数据划分

默认设置按以下样本数校验输入。

| 批次 | H / M / L | 划分 |
| --- | --- | --- |
| B1 | 47 / 69 / 23 | 训练83，验证28，测试28 |
| B2 | 47 / 71 / 24 | 适应70，留出72 |
| B3 | 300 / 100 / 200 | 适应300，留出300 |

| 品种 | 类别 | `site`取值 |
| --- | --- | --- |
| DX、PLBZ | DX为H，PLBZ为L | `Nanjing` |
| LM、MMK | H | `Kaiyuan` |
| XCX、XHL | XCX为M，XHL为L | `Quanjiao` |

## Excel转换

`tools/import_legacy_excel.py`根据行映射生成上述CSV。输入参数为：

| 参数 | 内容 |
| --- | --- |
| `--mapping` | CSV，包含manifest字段，以及`nir_file`、`nir_sheet`、`nir_row`、`voc_file`、`voc_sheet`、`voc_row` |
| `--schema` | JSON，`nir_columns`按B1/B2/B3列出499个原始列名；`voc_columns`将四个VOC字段映射到Excel列名 |
| `--wavelengths` | 已准备的`wavelengths.csv` |
| `--out` | 新建的输出目录 |

`nir_row`和`voc_row`使用pandas读取后的0基数据行号。B1样本填写VOC映射；B2和B3的VOC映射字段可留空。相对文件路径从mapping文件所在目录解析。

```bash
python -m tools.import_legacy_excel --mapping "**" --schema "**" --wavelengths "**" --out "**"
```
