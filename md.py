import numpy as np
import tensorflow as tf

name = [
  "30-30リピーター",
  "オルタネーター",
  "ボセック",
  "CAR",
  "チャージライフル",
  "ディボーション",
  "EVA 8",
  "フラットライン",
  "G7スカウト",
  "ハボック",
  "ヘムロック",
  "ナイフ",
  "クレーバー",
  "ロングボウ",
  "Lスター",
  "マスティフ",
  "モザンビーク",
  "ネメシス",
  "P2020",
  "ピースキーパー",
  "プラウラー",
  "R99",
  "R301",
  "ランページ",
  "RE45",
  "センチネル",
  "スピットファイア",
  "トリプルテイク",
  "ボルト",
  "ウィングマン"
]

model = tf.keras.models.load_model("model")
#model = tf.keras.models.load_model("APEX_AI.h5")

model.summary()

img = tf.keras.utils.load_img(
   "test.png", target_size=(735,735)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "この画像は {} に {:.2f}％類似しています"
    .format(name[np.argmax(score)], 100 * np.max(score))
)

model.save("APEX_AI.keras")