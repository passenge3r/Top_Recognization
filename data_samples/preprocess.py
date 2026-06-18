import cv2
import numpy as np
from scipy.signal import find_peaks
from config import *

class Preprocessor:
    def __init__(self, notch_angles=None):
        """
        notch_angles: 两个缺口的中心角度（度），列表或None。
                      如果为None，则推理时中心圆区域全部挖空（不排除缺口）。
        """
        self.notch_angles = notch_angles

    # ================== 几何校正 ==================
    def correct_ellipse(self, img):
        """
        检测椭圆并校正为标准圆。自动排除过小的椭圆（如中心卡槽）。
        若失败则回退到中心裁剪+缩放。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print("警告：未找到轮廓，使用中心裁剪+缩放")
            return self._center_crop_resize(img)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        h, w = img.shape[:2]
        min_side = min(h, w)
        min_axis_thresh = min_side * 0.6

        for cnt in contours:
            if cnt.shape[0] < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse
            if max(MA, ma) < min_axis_thresh:
                continue

            R = R_FIX
            scale = R / (MA / 2.0)
            cos_a = np.cos(np.deg2rad(angle))
            sin_a = np.sin(np.deg2rad(angle))
            Sy = MA / ma

            T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
            R_neg = np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
            S_y   = np.array([[1, 0, 0], [0, Sy, 0], [0, 0, 1]], dtype=np.float64)
            R_pos = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
            S_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
            T_pos = np.array([[1, 0, R], [0, 1, R], [0, 0, 1]], dtype=np.float64)

            M = T_pos @ S_scale @ R_pos @ S_y @ R_neg @ T_neg
            affine_mat = M[:2, :]

            corrected = cv2.warpAffine(img, affine_mat, (IMG_SIZE, IMG_SIZE),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(128,128,128))
            return corrected

        print("警告：未找到足够大的椭圆轮廓，使用中心裁剪+缩放")
        return self._center_crop_resize(img)

    def _center_crop_resize(self, img):
        """备用：中心裁剪并缩放至标准尺寸"""
        h, w = img.shape[:2]
        side = min(h, w)
        left, top = (w - side) // 2, (h - side) // 2
        return cv2.resize(img[top:top+side, left:left+side], (IMG_SIZE, IMG_SIZE))

    # ================== 光照归一化 ==================
    def light_normalize(self, img):
        """对 V 通道做 CLAHE 以平衡光照，保持色相不变。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ================== 窗口边界拟合 ==================
    def fit_window_boundaries(self, img):
        """
        利用红绿粗分割在极坐标下统计像素分布，找到两个80°扇形窗口的精确起止角度。
        返回列表：[(w1_start, w1_end), (w2_start, w2_end)]，单位为度。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 红色两段
        mask_red1 = cv2.inRange(hsv, (0, 30, 30), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (160, 30, 30), (180, 255, 255))
        mask_green = cv2.inRange(hsv, (40, 30, 30), (80, 255, 255))
        combined = cv2.bitwise_or(mask_red1, mask_red2)
        combined = cv2.bitwise_or(combined, mask_green)

        center = (R_FIX, R_FIX)
        angle_hist = np.zeros(360)
        for r in range(CENTER_R + 5, R_FIX - 5, 2):
            for theta_idx in range(360):
                rad = np.deg2rad(theta_idx)
                x = int(center[0] + r * np.cos(rad))
                y = int(center[1] + r * np.sin(rad))
                if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                    if combined[y, x] > 0:
                        angle_hist[theta_idx] += 1
        # 平滑
        angle_hist = np.convolve(angle_hist, np.ones(5)/5, mode='same')

        # 滑动窗口（宽度80°）求和，寻找两个峰（应间隔约180°）
        window_sum = np.convolve(angle_hist, np.ones(int(WINDOW_ANGLE)), mode='same')
        peaks = []
        temp = window_sum.copy()
        for _ in range(2):
            idx = np.argmax(temp)
            peaks.append(idx)
            temp[max(0, idx - 80):min(360, idx + 80)] = 0
        peaks.sort()
        w1c = peaks[0]
        w2c = peaks[1]
        w1c_corr = (w1c + (w2c - 180)) / 2 % 360
        w2c_corr = (w1c_corr + 180) % 360

        w1_start = (w1c_corr - WINDOW_ANGLE/2) % 360
        w1_end   = (w1c_corr + WINDOW_ANGLE/2) % 360
        w2_start = (w2c_corr - WINDOW_ANGLE/2) % 360
        w2_end   = (w2c_corr + WINDOW_ANGLE/2) % 360

        return [(w1_start, w1_end), (w2_start, w2_end)]

    # ================== 缺口标定（基于多张真实图） ==================
    def calibrate_notches(self, images):
        """
        利用多张真实圆盘图像，自动检测中心卡槽缺口的角度中心。
        原理：在中心圆区域内统计红绿像素随角度的分布，缺口处会露出底层红绿，形成峰值。
        返回两个缺口中心角度（度）。
        """
        angle_hist = np.zeros(360)
        for img in images:
            img_corr = self.correct_ellipse(img)
            img_eq = self.light_normalize(img_corr)
            red = self._get_red_mask(img_eq)
            green = self._get_green_mask(img_eq)
            combined = cv2.bitwise_or(red, green)
            center = (R_FIX, R_FIX)
            for r in range(1, CENTER_R):
                for theta_idx in range(360):
                    rad = np.deg2rad(theta_idx)
                    x = int(center[0] + r * np.cos(rad))
                    y = int(center[1] + r * np.sin(rad))
                    if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                        if combined[y, x] > 0:
                            angle_hist[theta_idx] += 1
        angle_hist = np.convolve(angle_hist, np.ones(5)/5, mode='same')
        peaks, properties = find_peaks(angle_hist, distance=30, height=np.max(angle_hist)*0.3)
        if len(peaks) >= 2:
            peak_vals = angle_hist[peaks]
            sorted_idx = np.argsort(peak_vals)[::-1][:2]
            notch_centers = peaks[sorted_idx]
            return sorted(notch_centers.tolist())
        else:
            print("警告：无法自动检测缺口，忽略缺口。")
            return None

    def _get_red_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 50, 30), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 50, 30), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)

    def _get_green_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (40, 50, 30), (80, 255, 255))

    # ================== 掩膜生成 ==================
    def _angle_in_range(self, angle, start, end):
        """判断角度是否在扇形区间内（0~360°环绕处理）"""
        angle = angle % 360
        start = start % 360
        end = end % 360
        if start < end:
            return start <= angle <= end
        else:
            return angle >= start or angle <= end

    def _create_individual_masks(self, window_angles):
        """
        根据两个窗口的角度范围，生成两个独立掩膜。
        每个掩膜：包含对应80°扇形，但挖掉中心卡槽圆区，并根据标定挖掉缺口区域。
        """
        center = (R_FIX, R_FIX)
        masks = []
        for start, end in window_angles:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cv2.ellipse(mask, center, (R_FIX-2, R_FIX-2), 0, start, end, 255, -1)
            cv2.circle(mask, center, CENTER_R, 0, -1)
            if self.notch_angles is not None:
                for notch_angle in self.notch_angles:
                    notch_half = 12  # 缺口宽度的一半（度），可配置
                    if self._angle_in_range(notch_angle, start, end):
                        cv2.ellipse(mask, center, (CENTER_R, CENTER_R), 0,
                                    notch_angle - notch_half,
                                    notch_angle + notch_half, 0, -1)
            masks.append(mask)
        return masks[0], masks[1]

    # ================== 红绿像素提取（直方图双峰法） ==================
    def extract_rg_hist(self, img, window_mask):
        """基于 H 直方图双峰分割红绿像素，鲁棒性强，无需 K‑Means"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 排除高光
        highlight_mask = cv2.inRange(hsv, (0, 0, HIGHLIGHT_V_LOW), (180, HIGHLIGHT_S_MAX, 255))
        valid_mask = cv2.bitwise_and(window_mask, cv2.bitwise_not(highlight_mask))
        y, x = np.where(valid_mask > 0)
        if len(y) < 100:
            return 0, 0

        H = hsv[y, x, 0].astype(np.float32)
        S = hsv[y, x, 1].astype(np.float32)
        V = hsv[y, x, 2].astype(np.float32)

        # 黑色像素：V < 60 且 S < 50
        black_mask = (V < 60) & (S < 50)
        # 彩色像素（红/绿）
        color_mask = (~black_mask) & (S > 40)
        H_color = H[color_mask]

        if len(H_color) < 50:
            # 尝试宽松一点
            color_mask = (~black_mask) & (S > 30)
            H_color = H[color_mask]
        if len(H_color) < 50:
            return 0, 0

        # 直方图统计
        hist, bins = np.histogram(H_color, bins=90, range=(0, 180))
        hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')

        # 扩展以处理红色环绕
        hist_ext = np.concatenate([hist_smooth[-20:], hist_smooth, hist_smooth[:20]])
        peaks, _ = find_peaks(hist_ext, distance=15, height=np.max(hist_ext)*0.1)

        if len(peaks) < 2:
            # 回退：根据经验阈值
            red_count = np.sum((H_color < 20) | (H_color > 160))
            green_count = np.sum((H_color > 40) & (H_color < 80))
            return red_count, green_count

        # 将峰值映射回 0~180
        peak_vals = [(p - 20) % 180 for p in peaks]
        red_peak = None
        green_peak = None
        for p in peak_vals:
            if p < 20 or p > 160:
                red_peak = p
            elif 40 < p < 80:
                green_peak = p

        if red_peak is None or green_peak is None:
            red_count = np.sum((H_color < 20) | (H_color > 160))
            green_count = np.sum((H_color > 40) & (H_color < 80))
            return red_count, green_count

        # 环形距离
        def circ_dist(h, target):
            d = np.abs(h - target)
            return np.minimum(d, 180 - d)

        dist_red = circ_dist(H_color, red_peak)
        dist_green = circ_dist(H_color, green_peak)
        red_mask = dist_red < dist_green
        red_count = np.sum(red_mask)
        green_count = np.sum(~red_mask)

        return red_count, green_count

    # ================== 主处理流程 ==================
    def process(self, img):
        """
        完整预处理流水线，返回 (R, G) 特征。
        """
        img_corr = self.correct_ellipse(img)
        img_eq = self.light_normalize(img_corr)
        window_angles = self.fit_window_boundaries(img_eq)
        if window_angles is None or len(window_angles) != 2:
            raise RuntimeError("无法定位两个对称扇形窗口，请检查输入图像。")
        mask1, mask2 = self._create_individual_masks(window_angles)
        R1, G1 = self.extract_rg_hist(img_eq, mask1)
        R2, G2 = self.extract_rg_hist(img_eq, mask2)
        R = (R1 + R2) / 2.0
        G = (G1 + G2) / 2.0
        return R, G
print("preprocess.py")