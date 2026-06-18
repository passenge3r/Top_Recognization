import cv2
import torch
from ultralytics import YOLO
import numpy as np


class ValveAIPreprocessor:
    """
    阀门预处理专家系统接口
    功能：自动识别阀门、裁剪ROI特写、提取归一化坐标线索
    """

    def __init__(self, top_model='top_best.pt', side_model='side_best.pt'):
        # 1. 初始化加载两个专家模型
        try:
            self.model_top = YOLO(top_model)
            self.model_side = YOLO(side_model)
            print("✅ 阀门预处理专家模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败，请检查路径: {e}")

    def process(self, image_path, view_type='side'):
        """
        核心接口函数
        :param image_path: 原始大图路径
        :param view_type: 'top' 或 'side'，指定使用的专家模型
        :return: (roi_image, normalized_coords)
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        h_orig, w_orig = img.shape[:2]

        # 选择对应的模型
        model = self.model_top if view_type == 'top' else self.model_side

        # 执行推理
        results = model.predict(source=img, conf=0.5, verbose=False)[0]

        if len(results.boxes) == 0:
            return None, None

        # --- 1. 获取裁剪后的特写图 (ROI) ---
        box = results.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # 增加 10 像素缓冲，防止切得太死
        padding = 10
        roi = img[max(0, y1 - padding):min(h_orig, y2 + padding),
              max(0, x1 - padding):min(w_orig, x2 + padding)]

        # --- 2. 获取归一化坐标 (即你说的“直觉”) ---
        # 返回格式为: [Center_x, Center_y, Indicator_x, Indicator_y]
        # 所有值均在 0.0 ~ 1.0 之间
        kpts = results.keypoints.xyn[0].cpu().numpy()

        cx, cy = kpts[0]  # Center 归一化坐标
        ix, iy = kpts[1]  # Indicator 归一化坐标

        coords = [cx, cy, ix, iy]

        return roi, coords


# ==========================================
# 示例：给做算法的人演示怎么调用
# ==========================================
if __name__ == "__main__":
    # 1. 实例化接口
    preprocessor = ValveAIPreprocessor()

    # 2. 调用接口处理一张图
    # 假设他现在要处理一张侧视图
    test_img = 'D:/data_side_all/0001_3.4.jpg'
    roi, coords = preprocessor.process(test_img, view_type='side')

    if roi is not None:
        print(f"提取成功！")
        print(f"坐标线索 (Center & Indicator): {coords}")

        # 展示算法
        cv2.imshow("What ResNet sees", roi)
        cv2.waitKey(0)