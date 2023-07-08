import numpy as np
import tensorflow as tf

name = {
  "30-30repeater": "30-30リピーター",
  "alternator": "オルタネーター",
  "bosek": "ボセック",
  "car": "CAR",
  "chargerifle":  "チャージライフル",
  "devotion": "ディボーション",
  "eva8": "EVA 8",
  "flatline": "フラットライン",
  "g7": "G7スカウト",
  "havoc": "ハボック",
  "hemlock": "ヘムロック",
  "knife": "ナイフ",
  "kraber": "クレーバー",
  "longbow": "ロングボウ",
  "lstar": "Lスター",
  "mastiff": "マスティフ",
  "mozambique": "モザンビーク",
  "nemesis": "ネメシス",
  "p2020": "P2020",
  "peacekeeper": "ピースキーパー",
  "prowler": "プラウラー",
  "r99": "R99",
  "r301": "R301",
  "rampage": "ランページ",
  "re45": "RE45",
  "sentinel": "センチネル",
  "spitfire": "スピットファイア",
  "tripletake": "トリプルテイク",
  "volt": "ボルト",
  "wingman": "ウィングマン"
}

ds = tf.keras.utils.image_dataset_from_directory("dataset")

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
    .format(name[ds.class_names[np.argmax(score)]], 100 * np.max(score))
)

model.save("APEX_AI.h5")