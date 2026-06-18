import cv2
import numpy as np
import os
import math

# ===================== 最终版：只保留最标准的2个扇形 · 彻底剔除圆环 =====================
def draw_best_two_sectors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0, 0, None, None

    img_draw = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    img_area = h * w

    # 1. 提取绿色
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. 提取红色
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # 3. 合并红绿区域
    full_mask = cv2.bitwise_or(green_mask, red_mask)

    # 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ------------------- 第一步：提取所有符合面积的轮廓 -------------------
    contours_all, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []

    for c in contours_all:
        area = cv2.contourArea(c)
        ratio = area / img_area

        # 基础面积过滤
        if not (0.01 < ratio < 0.4):
            continue

        # 圆形度判断（扇形圆形度一定远低于圆形）
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter ** 2))

        # 圆环/整圆直接丢弃
        if circularity > 0.85:
            continue

        candidates.append((area, c))

    # ------------------- 核心：只保留面积最大的2个（最标准的两个扇形） -------------------
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_two = [c for (a, c) in candidates[:2]]

    # 计算面积
    total_sector_area = sum(cv2.contourArea(c) for c in best_two)

    # 只画这2个扇形
    for cnt in best_two:
        cv2.drawContours(img_draw, [cnt], -1, (0, 255, 255), 3)

    # ------------------- 绿色部分 -------------------
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_green_area = 0
    if contours_green:
        green_cnt = max(contours_green, key=cv2.contourArea)
        total_green_area = cv2.contourArea(green_cnt)
        cv2.drawContours(img_draw, [green_cnt], -1, (0, 255, 0), 3)

    # 显示面积
    cv2.putText(img_draw, f"Sector Area: {total_sector_area:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(img_draw, f"Green Area: {total_green_area:.1f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    return total_sector_area, total_green_area, full_mask, img_draw

# ===================== 批量运行 =====================
folder = r"C:/Users/HP/Desktop/data_samples/top"

img_paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith(('jpg', 'png', 'jpeg'))]

for path in img_paths:
    sector_area, green_area, full_mask, img_draw = draw_best_two_sectors(path)
    filename = os.path.basename(path)
    
    print(f"📌 {filename}")
    #print(f"   扇形窗口总面积：{sector_area:.2f} 像素")
    #print(f"   绿色部分面积：{green_area:.2f} 像素")
    print(f"   绿色部分角度：{green_area*2*80.0/sector_area:.2f}度")
    if img_draw is not None:
        mask_3ch = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((cv2.imread(path), img_draw, mask_3ch))
        combined = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
        cv2.imshow("BEST 2 SECTORS ONLY", combined)
        cv2.waitKey(0)

cv2.destroyAllWindows()