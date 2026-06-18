import cv2
import numpy as np
import os
import math

# ===================== 只框圆盘内部扇形 · 彻底忽略外围 · 最终版 =====================
def detect_final_internal(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, None, None

    img_draw = img.copy()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --------------------- 原始颜色阈值（恢复最初版本） ---------------------
    # 绿色
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 红色（原始范围）
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # 合并红绿
    rg_mask = cv2.bitwise_or(green_mask, red_mask)

    # --------------------- 【核心：创建内部遮罩，只保留圆盘中间区域】 ---------------------
    center = (w // 2, h // 2)
    max_radius = min(w, h) // 3  # 只保留内部 1/3 区域，彻底屏蔽外围
    internal_mask = np.zeros_like(rg_mask)
    cv2.circle(internal_mask, center, max_radius, 255, -1)

    # 只保留内部的红绿区域 → 外围完全忽略
    rg_internal = cv2.bitwise_and(rg_mask, internal_mask)
    green_internal = cv2.bitwise_and(green_mask, internal_mask)

    # 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rg_internal = cv2.morphologyEx(rg_internal, cv2.MORPH_CLOSE, kernel, iterations=1)
    rg_internal = cv2.morphologyEx(rg_internal, cv2.MORPH_OPEN, kernel, iterations=1)

    # --------------------- 提取内部轮廓，只画合格扇形（圆心角 ≤90°） ---------------------
    contours_all, _ = cv2.findContours(rg_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours_all:
        # 最小外接圆，计算圆心角
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        circle_area = math.pi * radius * radius
        if circle_area < 1:
            continue
        sector_angle = (area / circle_area) * 360

        # 只保留圆心角 ≤90° 的扇形
        if 0 < sector_angle <= 90:
            cv2.drawContours(img_draw, [cnt], -1, (0, 255, 255), 2)

    # --------------------- 勾勒内部绿色部件 ---------------------
    contours_green, _ = cv2.findContours(green_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    angle = 0.0
    if contours_green:
        green_cnt = max(contours_green, key=cv2.contourArea)
        cv2.drawContours(img_draw, [green_cnt], -1, (0, 255, 0), 2)

        # 计算旋转角度
        rect = cv2.minAreaRect(green_cnt)
        angle = rect[2]
        if angle < -45:
            angle += 90
        angle = abs(angle)

    cv2.putText(img_draw, f"Angle: {angle:.1f}°", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return angle, rg_internal, img_draw

# ===================== 批量运行 =====================
folder = r"C:/Users/HP/Desktop/data_samples/top"

img_paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith(('jpg', 'png', 'jpeg'))]

for path in img_paths:
    angle, final_mask, img_draw = detect_final_internal(path)
    print(f"✅ {os.path.basename(path)} | 内部角度 = {angle:.2f}°")

    if img_draw is not None:
        m3 = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((cv2.imread(path), img_draw, m3))
        combined = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
        cv2.imshow("ONLY INTERNAL SECTOR", combined)
        cv2.waitKey(0)

cv2.destroyAllWindows()