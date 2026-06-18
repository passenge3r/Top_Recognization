import cv2
import matplotlib.pyplot as plt

img_path = "C:/Users/HP/Desktop/data_samples/top/0011_29.9.jpg"
img = cv2.imread(img_path)
if img is None:
    print("读取失败，检查路径")
else:
    print(f"原图形状: {img.shape}, 类型: {img.dtype}")
    # 正确转换颜色空间用于matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("原始图片")
    plt.show()