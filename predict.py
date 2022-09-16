"""
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
"""
import subprocess
import sys

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
finally:
    import torch
    import torch.utils.data
    import torch.utils.data.distributed

from dataset import ETRIDataset_emo_custom
from networks import *

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
print(torch.__version__)


def main():
    """The main function of the test process for performance measurement."""
    mdl_path = "models/model_best_last.pkl"
    net = EfficientNetV2_emo_clothes().to(DEVICE)
    trained_weights = torch.load(mdl_path, map_location=DEVICE)
    net.load_state_dict(trained_weights)
    print(f"Your model path: {mdl_path}")

    df = pd.read_csv("../data/task1/info_etri20_emotion_test.csv")
    val_dataset = ETRIDataset_emo_custom(
        df, base_path="../data/task1/test/", image_size=380, type="test", transform=None
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    daily_pred_list = np.array([])
    gender_pred_list = np.array([])
    embel_pred_list = np.array([])

    print("start Inference...!")
    with torch.no_grad():
        net.eval()
        for j, sample in enumerate(tqdm(val_dataloader)):
            for key in sample:
                sample[key] = sample[key].to(DEVICE)
            out_daily, out_gender, out_embel, _ = net(sample)

            daily_pred = out_daily
            _, daily_indx = daily_pred.max(1)
            daily_pred_list = np.concatenate(
                [daily_pred_list, daily_indx.cpu()], axis=0
            )

            gender_pred = out_gender
            _, gender_indx = gender_pred.max(1)
            gender_pred_list = np.concatenate(
                [gender_pred_list, gender_indx.cpu()], axis=0
            )

            embel_pred = out_embel
            _, embel_indx = embel_pred.max(1)
            embel_pred_list = np.concatenate(
                [embel_pred_list, embel_indx.cpu()], axis=0
            )

        df["Daily"] = daily_pred_list.astype(int)
        df["Gender"] = gender_pred_list.astype(int)
        df["Embellishment"] = embel_pred_list.astype(int)
        # 제출시 생성 위치는 기본적으로 /home/work/model/prediction.csv로
        df.to_csv("/home/work/model/prediction.csv", index=False)


if __name__ == "__main__":
    main()
