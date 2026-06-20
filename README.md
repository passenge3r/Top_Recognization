# Top Recognition

顶部绿色扇形圆心角识别项目，用于从阀门顶部图片中估计绿色扇形角度。项目提供统一入口 `top_angle_estimator.py`，支持传统 OpenCV 几何法和 ResNet50 角度回归模型两种方式。

## 项目内容

- `top_angle_estimator.py`: 统一预测入口。
- `preprocess_interface.py`: YOLO ROI 预处理接口。
- `train_top_angle_model.py`: ResNet50 角度回归模型训练入口。
- `top_best.pt`: 顶部阀门检测模型。
- `side_best.pt`: 侧面阀门检测模型。
- `outputs/top_angle_resnet50_retrained_300_best.pth`: 当前默认 ResNet50 角度回归模型，已更新为全模型 300 轮训练中的验证集最佳权重。
- `data_samples/`: 样本图片与测试数据。
- `TOP_ANGLE_USAGE.md`: 更详细的参数和使用说明。

## 方法

### OpenCV

OpenCV 方法会先检测顶部阀门 ROI，再通过 HSV 阈值提取绿色扇形和红色参考区域，结合面积比例和外圈角跨度估计绿色扇形角度。当前默认使用 `hybrid` 模式。

### AI

AI 方法会先提取顶部阀门 ROI，将 ROI 缩放到 `224x224`，再用 ResNet50 回归模型预测绿色扇形角度。

当前默认 AI 模型来自 `data_samples/top` 的 847 张带标签图片，使用 `--train-mode full` 全模型训练 300 轮后取验证集 MAE 最低的第 60 轮权重。训练后在 `data_samples/top_test` 上的 AI 结果为 `n=70, MAE=0.514, MaxAE=2.863`。

## 环境依赖

推荐使用已有 conda 环境：

```powershell
conda activate k
```

也可以手动安装依赖：

```powershell
pip install opencv-python ultralytics torch torchvision pillow numpy
```

在受限环境中，建议设置 Ultralytics 配置目录到项目内部：

```powershell
$env:YOLO_CONFIG_DIR='C:\Users\HP\Desktop\Top_Recognization\outputs\ultralytics_config'
```

## 常用命令

只使用 OpenCV 预测全部测试图片：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method opencv --output outputs/opencv_refined_all.csv
```

只使用 OpenCV 预测某一组测试图片：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test/t4 --method opencv --output outputs/opencv_t4.csv
```

同时运行 OpenCV 和 AI：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method both --output outputs/top_angle_predictions.csv
```

只运行 AI：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method ai --output outputs/top_angle_predictions.csv
```

重新训练 AI 模型 300 轮，并保存验证集最佳权重：

```powershell
$env:YOLO_CONFIG_DIR='C:\Users\HP\Desktop\Top_Recognization\outputs\ultralytics_config'
C:\Users\HP\anaconda3\envs\k\python.exe .\train_top_angle_model.py --train-mode full --epochs 300 --batch-size 8 --eval-batch-size 32 --lr 1e-5 --output-model outputs/top_angle_resnet50_retrained_300_best.pth --log-csv outputs/top_angle_full_300_log.csv --split-csv outputs/top_angle_full_300_split.csv
```

不使用 YOLO 预处理，直接按图片中心估计：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test/t1 --method opencv --preprocess none
```

## 输出

CSV 输出字段包括：

- `image`: 图片相对路径。
- `method`: 使用的方法，值为 `opencv` 或 `ai`。
- `angle`: 预测角度。
- `true_angle`: 从文件名解析出的真实角度。
- `error`: 绝对误差。
- `status`: 处理状态，例如 `ok`、`no_roi`、`no_green`。

## 数据命名

测试图片文件名需要包含真实角度，例如：

```text
10_18.2.jpg
```

脚本会取最后一个下划线后的数字作为真实角度。没有角度标签的图片仍会预测，但不会参与 MAE 统计。

## 说明

当前 OpenCV 默认参数已经针对 `data_samples/top_test/t1` 到 `t4` 四组测试集做过平衡。更完整的参数说明和验证结果请查看 `TOP_ANGLE_USAGE.md`。
