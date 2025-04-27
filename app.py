import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import os
from io import BytesIO
from PIL import Image
from PIL import ImageOps
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# タイトル
st.title("お名前判定")
st.write("僕の家で飼っているペットの名前がわかる！")

# 学習データのディレクトリからクラス名を取得
class_names = ['mint','mitsumame','tora','rui']

# クラス名と表示名のマッピング辞書
class_display_names = {
    "mint": "みんと",
    "mitsumame": "みつまめ",
    "tora": "とら",
    "rui": "るい"
}

# 画像サイズの指定
IMG_SIZE = (160, 160)

# モデル読み込みをキャッシュ（リソース用キャッシュ）
@st.cache_resource
def load_classifier_model():
    # 非同期ではなく同期関数でモデルを読み込む
    model = load_model('pet_classifier_model.h5')
    return model

loaded_model = load_classifier_model()

# ファイルアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像をRGBに変換して読み込み
    image_data = Image.open(uploaded_file)
    image_data = ImageOps.exif_transpose(image_data).convert("RGB")
    # アップロード画像を表示
    st.image(image_data, caption='アップロードされた画像', use_container_width=True)

    # 画像を前処理
    img = image_data.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # モデルで推論
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_name = class_names[predicted_class]
    result_name = class_display_names.get(class_name, class_name)

    # 結果を表示
    st.success(f"予測結果: {result_name}")
    st.write(f"(クラス: {class_name}, スコア: {predictions[0][predicted_class]:.4f})")