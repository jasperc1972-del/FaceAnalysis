import cv2
import numpy as np

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片：{img_path}")
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("找不到掌紋輪廓")
    x, y, w, h = cv2.boundingRect(contours[0])
    roi = img[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (128, 128))
    return roi_resized

def gabor_feature(img):
    filters = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kern = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    feats = [cv2.filter2D(img, cv2.CV_8UC3, f) for f in filters]
    return np.concatenate([f.flatten() for f in feats])

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

# 主流程
img1 = preprocess(r"C:\Users\Jasper\PycharmProjects\Face\images\palm1.jpg")
img2 = preprocess(r"C:\Users\Jasper\PycharmProjects\Face\images\palm4.jpg")

feat1 = gabor_feature(img1)
feat2 = gabor_feature(img2)

similarity = cosine_similarity(feat1, feat2)
print(f"掌紋相似度（Cosine Similarity）: {similarity:.4f}")

if similarity > 0.85:
    print("✅ 判定：可能是同一個人")
else:
    print("❌ 判定：可能是不同的人")