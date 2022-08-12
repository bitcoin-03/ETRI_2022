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
from dataset import ETRIDataset_emo
from networks import *

import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Baseline_ResNet_emo")
parser.add_argument("--version", type=str, default="Baseline_ResNet_emo")
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--lr", default=0.0001, type=float, metavar="N", help="learning rate"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

a = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def main():
    """The main function for model training."""
    if os.path.exists("models") is False:
        os.makedirs("models")

    save_path = "models/" + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 모델은 parser로 network.py에 구현되어 있는 클래스 이름을 입력받아서 생성되게끔 했습니다.
    print("Loading model...")
    # net = EfficientNet_emo().to(DEVICE)
    net = a.model()

    print("Loading data....")
    # 경로는 각자 맞춰주시면 될것같습니다.
    df = pd.read_csv(
        "../TEAM비뜨코인/ETRI_Season3/task1_data/info_etri20_emotion_train.csv"
    )
    train_dataset = ETRIDataset_emo(
        df, base_path="../TEAM비뜨코인/ETRI_Season3/task1_data/train/"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    total_step = len(train_dataloader)
    step = 0
    t0 = time.time()

    print("Preparing Train....")
    for epoch in range(a.epochs):
        net.train()
        t1 = time.time()

        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            step += 1
            for key in sample:
                sample[key] = sample[key].to(DEVICE)

            out_daily, out_gender, out_embel = net(sample)

            loss_daily = criterion(out_daily, sample["daily_label"])
            loss_gender = criterion(out_gender, sample["gender_label"])
            loss_embel = criterion(out_embel, sample["embel_label"])
            loss = loss_daily + loss_gender + loss_embel

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, "
                    "Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Time : {:2.3f}".format(
                        epoch + 1,
                        a.epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        loss_daily.item(),
                        loss_gender.item(),
                        loss_embel.item(),
                        time.time() - t0,
                    )
                )

                t0 = time.time()

        if (epoch + 1) % 10 == 0:
            a.lr *= 0.9
            optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
            print(f"learning rate is decayed... learning rate is {a.lr}")

        if (epoch + 1) % 20 == 0:
            print("Saving Model....")
            torch.save(
                net.state_dict(), save_path + "/model_" + str(epoch + 1) + ".pkl"
            )
            print("OK.")
        print(
            "Epoch {} is finished. Total Time : {:2.3f} \n".format(
                epoch + 1, time.time() - t1
            )
        )


if __name__ == "__main__":
    main()
