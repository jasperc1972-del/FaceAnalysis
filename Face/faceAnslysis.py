from deepface import DeepFace

# 指定圖片路徑
img_path = "girl.jpg"

# 分析人臉屬性：年齡、性別、情緒、種族
result = DeepFace.analyze(img_path=img_path, actions=["age", "gender", "emotion", "race"])

# 顯示結果
print("分析結果：")
print(f"年齡：{result[0]['age']}")
print(f"性別：{result[0]['gender']}")
print(f"情緒：{result[0]['dominant_emotion']}")
print(f"種族：{result[0]['dominant_race']}")