import os
from deepface import DeepFace

# 設定圖片資料夾路徑
folder_path = "images"

# 取得資料夾中所有圖片檔案
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

# 逐張分析
for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    try:
        result = DeepFace.analyze(img_path=img_path, actions=["age", "gender", "emotion", "race"])
        print(f"\n📷 圖片：{img_name}")
        print(f"年齡：{result[0]['age']}")
        print(f"性別：{result[0]['gender']}")
        print(f"情緒：{result[0]['dominant_emotion']}")
        print(f"種族：{result[0]['dominant_race']}")
    except Exception as e:
        print(f"\n⚠️ 無法分析 {img_name}：{str(e)}")