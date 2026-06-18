#config.py
# 圆盘归一化参数
R_FIX = 400                 # 校正后标准圆半径
IMG_SIZE = 2 * R_FIX        # 输出图像尺寸 800x800

# 几何比例（来自你的测量）
CENTER_RADIUS_RATIO = 1/4.5  # 中心卡槽半径/圆盘半径
CENTER_R = int(R_FIX * CENTER_RADIUS_RATIO)

# 缺口自动标定参数
NUM_CALIB_SAMPLES = 10      # 用于标定缺口的真实图片数量
NOTCH_ANGLE_TOL = 5         # 标定时角度容差（度）
NOTCH_MIN_PIXEL_RATIO = 0.02 # 缺口区域红绿像素占比阈值

# 窗口参数
WINDOW_ANGLE = 80.0         # 露出扇形角度（度）
WINDOW_SYMMETRY_OFFSET = 180.0  # 两个窗口间的对称角度差

# 颜色阈值（HSV，可后期通过样本调整）
RED_LOW1 = (0, 50, 30)
RED_HIGH1 = (10, 255, 255)
RED_LOW2 = (160, 50, 30)
RED_HIGH2 = (180, 255, 255)
GREEN_LOW = (40, 50, 30)
GREEN_HIGH = (80, 255, 255)
BLACK_LOW = (0, 0, 0)
BLACK_HIGH = (180, 60, 80)   # 用于粗定位

# 高光阈值（反光）
HIGHLIGHT_V_LOW = 230
HIGHLIGHT_S_MAX = 40

# Bootstrap
N_BOOTSTRAP = 200           # 自助采样次数
BOOTSTRAP_SAMPLE_FRAC = 1.0 # 每次采样比例（1.0表示有放回抽同样数量）
print(1)