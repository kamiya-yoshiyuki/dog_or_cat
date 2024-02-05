from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torchvision.models import resnet18
import pytorch_lightning as pl
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# インスタンス化
app = FastAPI()


# 学習済みのモデルの読み込み
# model = pickle.load(open('model.pkl', 'rb'))


# ネットワークの定義
class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


# VS Codeでの読み込み
model = Net()
model.load_state_dict(torch.load("model_state.pth"))
model.eval()


transform = transforms.Compose(
    [
        transforms.Resize(256),  # 固定
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 固定
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 固定
    ]
)


@app.post("/prediction")
async def predict_image(image_file: UploadFile = File(...)):
    # 画像のバイトデータを受け取り、PIL Imageオブジェクトに変換
    image_bytes = await image_file.read()
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    print(image_file.filename)

    # 画像をモデルに適した形に変換（リサイズ、グレースケール変換、テンソル変換）
    transformed_image = transform(image)

    # 変換された画像をモデルに入力し、生の予測値を取得
    prediction = model(transformed_image.unsqueeze(0))

    # Softmax関数を適用して、生の予測値を確率に変換
    probabilities = F.softmax(prediction, dim=1)

    # 最も確率が高いクラスを決定
    most_probable_class = torch.argmax(probabilities)

    class_probabilities = probabilities.detach().numpy()[0]
    class_probabilities = [round(float(prob), 4) for prob in class_probabilities]

    # 最も確率が高いクラスと各クラスの確率を含む辞書をレスポンスとして返す
    return {
        "most_probable_class": most_probable_class.item(),
        "class_probabilities": class_probabilities,
    }
