# 顶部绿色扇形圆心角识别使用说明

项目统一入口是 `top_angle_estimator.py`，支持两种方法：

- `opencv`：传统图像处理几何法，当前默认使用精进后的 `hybrid` 模式。
- `ai`：ResNet50 角度回归模型。

## OpenCV 方法

默认 OpenCV 流程：

1. 使用 `top_best.pt` 检测顶部阀门 ROI 和圆心关键点。
2. 在 ROI 中用 HSV 阈值提取绿色扇形和红色参考区域。
3. 使用清理后的红/绿主连通域面积得到面积角。
4. 使用绿色区域外圈角跨度得到 span 角。
5. 常规情况下按较低 span 权重融合；当面积角明显大于 span 角时，认为红色参考区可能欠分割，自动提高 span 权重。
6. 最后做轻量输出修正。

当前默认参数已经按 `data_samples/top_test/t1` 到 `t4` 四组一起平衡：

```text
--opencv-mode hybrid
--green-hsv 40 80 80 100 255 255
--red-hsv1 0 124 124 10 255 255
--red-hsv2 160 124 124 180 255 255
--radius-quantiles 60 95
--opencv-span-trim 2
--opencv-hybrid-span-weight 0.24
--opencv-hybrid-diff-threshold 10
--opencv-hybrid-fallback-span-weight 1.0
--opencv-output-scale 0.9625
--opencv-output-offset 0.746
--opencv-calibration none
```

最新验证结果：

| 测试组 | 旧 OpenCV MAE | 当前 OpenCV MAE |
|---|---:|---:|
| `top_test/t1` | 1.520 | 1.181 |
| `top_test/t2` | 2.696 | 1.831 |
| `top_test/t3` | 2.028 | 1.154 |
| `top_test/t4` | 0.970 | 0.846 |
| `top_test` 全量 | 1.844 | 1.263 |

最新 OpenCV 全量预测文件：

```text
outputs/opencv_refined_all.csv
```

## AI 方法

AI 流程：

1. 使用 `top_best.pt` 做顶部阀门 ROI 预处理。
2. 将 ROI 缩放到 `224x224`，按 ImageNet 均值方差归一化。
3. 加载角度回归模型。
4. ResNet50 输出绿色扇形角度。

当前推荐 AI 模型：

```text
outputs/top_angle_resnet50_retrained_300_best.pth
```

## 安装依赖

推荐使用已有 conda 环境：

```powershell
conda activate k
```

如需手动安装依赖：

```powershell
pip install opencv-python ultralytics torch torchvision pillow numpy
```

## 运行前设置

在受限环境里，Ultralytics 可能会尝试读写用户目录配置。建议先在 PowerShell 中设置项目内配置目录：

```powershell
$env:YOLO_CONFIG_DIR='C:\Users\HP\Desktop\Top_Recognization\outputs\ultralytics_config'
```

## 常用命令

只跑 OpenCV，测试全部 `top_test`：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method opencv --output outputs/opencv_refined_all.csv
```

只跑 OpenCV，测试某一组：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test/t4 --method opencv --output outputs/opencv_t4.csv
```

同时跑 OpenCV 和 AI：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method both --output outputs/top_angle_predictions.csv
```

只跑 AI：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test --method ai --angle-model outputs/top_angle_resnet50_retrained_300_best.pth
```

不使用 YOLO 预处理，直接按图片中心估计：

```powershell
C:\Users\HP\anaconda3\envs\k\python.exe .\top_angle_estimator.py --input data_samples/top_test/t1 --method opencv --preprocess none
```

## 输出文件

CSV 字段：

- `image`：图片相对路径。
- `method`：`opencv` 或 `ai`。
- `angle`：预测角度。
- `true_angle`：从文件名解析出的真实角度。
- `error`：绝对误差。
- `status`：处理状态，常见为 `ok`、`no_roi`、`no_green`。

## 参数说明

- `--opencv-mode`：`hybrid`、`span` 或 `area`。默认 `hybrid`。
- `--radius-quantiles`：span 计算时保留的半径分位范围。默认 `60 95`。
- `--opencv-span-trim`：角度离群点裁剪比例。默认 `2`。
- `--opencv-hybrid-span-weight`：常规 hybrid 融合中的 span 权重。默认 `0.24`。
- `--opencv-hybrid-diff-threshold`：当面积角比 span 角大超过该阈值时，启用 fallback。默认 `10`。
- `--opencv-hybrid-fallback-span-weight`：fallback 时的 span 权重。默认 `1.0`。
- `--opencv-output-scale` / `--opencv-output-offset`：OpenCV 输出层轻量修正。默认已按四组测试集平衡。
- `--opencv-calibration`：OpenCV 线性校准，默认 `none`。

## 注意

文件名需要包含真实角度，例如 `10_18.2.jpg`。脚本会取最后一个下划线后的数字作为真实角度；没有角度标签的图片仍会预测，但不参与 MAE 统计。
