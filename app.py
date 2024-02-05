import streamlit as st

import requests
import pandas as pd
from PIL import Image
import io


def convert_image_to_byte_stream(image_path):
    # 画像ファイルを開く
    with Image.open(image_path) as img:
        # バイトストリームオブジェクトを作成
        byte_stream = io.BytesIO()
        # 画像をバイトストリームに保存する（ここではJPEG形式として保存）
        img.save(byte_stream, format="JPEG")
        # バイトストリームの位置を先頭に戻す
        byte_stream.seek(0)
        return byte_stream


st.title("画像分類 -犬か猫か-")

file = st.file_uploader("画像")
if file:
    image = io.BytesIO(file.read())

    # 使用例
    # image_path = 'Image20240203131824.jpg'
    # byte_stream = convert_image_to_byte_stream(image_path)
    # byte_stream を使用する処理（例：ファイルの保存、ネットワーク経由での送信など）

    files = {
        "image_file": (
            "handwritten_image.jpg",  # 任意のファイル名
            image,  # 画像のバイトデータ
            "image/jpg",  # ファイル形式
        )
    }


# FastAPIサーバーに画像を送信し、予測結果を取得
if st.button("送信"):

    response = requests.post(
        "https://dog-or-cat-g6g6.onrender.com/prediction",
        files=files,
    )
    response_json = response.json()

    st.image(image)

    # 予測されたクラスと確率を取得
    prediction = response_json.get("most_probable_class", "Unknown")
    class_probabilities = response_json.get("class_probabilities", [0.0] * 10)

    # st.subheader(prediction)

    target = ["犬", "猫"]
    # prediction = response.json()['prediction']

    st.write("# この画像は「", str(target[int(prediction)]), "」です")
