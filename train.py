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
### 라이브러리 설치 ####
import subprocess
import sys

try:
    from albumentations import *
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations"])
finally:
    from albumentations import *
from albumentations.pytorch import ToTensorV2

##########################

from dataset import ETRIDataset_emo
from networks import *

import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed
import wandb
import random
import numpy as np
from pathlib import Path
import glob, re




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
# models 안에 저장되는 이름과 wandb 실험 앞에 붙는 이름입니다.
parser.add_argument("--name", default="exp", help="model save at {exp_num}_모델이름")
# 실험에 대한 설명입니다.(실험 구분 목적) wandb에 exp_모델이름 뒤에 붙습니다.
parser.add_argument(
    "--explan", default="", help="experiment description, ex. exp_efficientnet_{explan}"
)
# 10에폭마다 체크포인트를 저장하게 구현하였고, 만약 20에폭까지 학습됐고, 이어서 학습하고 싶으시면
# model_resume_20.pth를 쓰시면 됩니다. 그럼 21에폭부터 학습이 이어집니다.
parser.add_argument("--resume_from", type=str, default=None, help="model_resume_20.pth")


a = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return f"{path}"
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def get_transforms(need=("train", "val")):
    transformations = {}
    if "train" in need:
        transformations["train"] = Compose(
            [
                HorizontalFlip(p=0.5),
                # ShiftScaleRotate(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.5,
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                ColorJitter(p=0.5),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    if "val" in need:
        transformations["val"] = Compose(
            [
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    return transformations


def main():
    """The main function for model training."""
    if os.path.exists("models") is False:
        os.makedirs("models")

    # save_path = "models/" + a.version
    # models 폴더에 exp, exp1, ....expN 으로 저장됩니다.
    save_path = increment_path(os.path.join("models/", a.name))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 모델은 parser로 network.py에 구현되어 있는 클래스 이름을 입력받아서 생성되게끔 했습니다.
    print("Loading model...")
    # net = EfficientNet_emo().to(DEVICE)
    if a.model == "Baseline_ResNet_emo":
        net = Baseline_ResNet_emo.to(DEVICE)
    elif a.model == "EfficientNet_emo":
        net = EfficientNet_emo().to(DEVICE)

    print("Loading data....")
    # 경로는 각자 맞춰주시면 될것같습니다.
    aug = get_transforms()
    df = pd.read_csv(
        "../TEAM비뜨코인/ETRI_Season3/task1_data/info_etri20_emotion_tr_val_simple.csv"
    )
    train_dataset = ETRIDataset_emo(
        df,
        base_path="../TEAM비뜨코인/ETRI_Season3/task1_data/train/",
        transform=aug["train"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=torch.cuda.device_count() * 4,
        pin_memory=True,
    )

    val_dataset = ETRIDataset_emo(
        df,
        base_path="../TEAM비뜨코인/ETRI_Season3/task1_data/train/",
        type="val",
        transform=aug["val"],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=torch.cuda.device_count() * 4,
        pin_memory=True,
    )

    if a.resume_from:
        # 저장했던 중간 모델 정보를 읽습니다.
        path = "models/" + a.resume_from
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.Adam(net.parameters(), lr=checkpoint["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:
        resume_epoch = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)

    # optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    total_step = len(train_dataloader)
    val_total_step = len(val_dataloader)
    step = 0
    val_step = 0
    t0 = time.time()

    # wandb
    """
    project = "본인 프로젝트 이름", 
    name="실험 이름, default=exp_모델명"
    """
    increment_name = save_path.split("/")[-1]
    model_explan = f"{'_'+a.explan if a.explan else ''}"
    wandb.init(
        project="model-test",
        name=f"{increment_name}_{a.model}{model_explan}",
        entity="bitcoin-etri",
        config=a,
    )
    wandb.save()
    wandb.watch(net)
    val_images = []
    print("Preparing Train....")
    for epoch in range(resume_epoch, a.epochs):
        # Train loop
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
                    "Train process "
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
            # wandb
            wandb.log(
                {
                    "loss": loss.item(),
                    "loss_daily": loss_daily.item(),
                    "loss_gender": loss_gender.item(),
                    "loss_embel": loss_embel.item(),
                }
            )
        val_loss_items = {
            'daily_val_loss': [],
            'gender_val_loss': [],
            'embel_val_loss': [],
        }
        # Validation loop
        with torch.no_grad():
            print("Calculating validation results...")
            net.eval()

            for i, val_batch in enumerate(val_dataloader):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(DEVICE)

                val_out_daily, val_out_gender, val_out_embel = net(val_batch)
                
                preds = (
                    torch.argmax(val_out_daily, dim=-1),
                    torch.argmax(val_out_gender, dim=-1),
                    torch.argmax(val_out_embel, dim=-1)
                )
                val_loss_items['daily_val_loss'].append(criterion(val_out_daily, val_batch["daily_label"],7).item())
                val_loss_items['gender_val_loss'].append(criterion(val_out_gender, val_batch["gender_label"],6).item())
                val_loss_items['embel_val_loss'].append(criterion(val_out_embel, val_batch["embel_label"],3).item())

                d = random.randint(0, len(val_dataloader) - 1)
                daily_list = ['실내복', '가벼운 외출', '오피스룩', '격식차린', '이벤트', '교복', '운동복']
                gender_list = ['밀리터리', '매니쉬', '유니섹스', '걸리쉬', '우아한', '섹시한']
                embellishment_list = ['장식이 없는', '포인트 장식이 있는', '장식이 많은']
                
                if len(val_images) < (epoch+1)*2:
                    val_images.append(
                        wandb.Image(
                        val_batch['image'][d],
                        caption='''
                            Pred/Truth
                            일상성 : {} / {}, 
                            성 : {} / {}, 
                            장식성 : {} / {}
                        '''.format(
                            daily_list[preds[0][d].item()],
                            daily_list[val_batch["daily_label"][d]], 
                            gender_list[preds[1][d].item()],
                            gender_list[val_batch["gender_label"][d]], 
                            embellishment_list[preds[2][d].item()],
                            embellishment_list[val_batch["embel_label"][d]]
                            )
                        )
                    )

            val_loss = [np.sum(list(val_loss_items.values())[i])/len(val_dataloader) for i in range(3)]
            print(
                "Validation process "
                "Epoch [{}/{}], Loss: {:.4f}, "
                "Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Time : {:2.3f}".format(
                    epoch + 1,
                    a.epochs,
                    np.sum(val_loss),
                    val_loss[0],
                    val_loss[1],
                    val_loss[2],
                    time.time() - t0,
                )
            )
            t0 = time.time()
            # wandb
            wandb.log(
                {
                    "Examples": val_images,
                    "val_loss": np.sum(val_loss),
                    "val_loss_daily": val_loss[0],
                    "val_loss_gender": val_loss[1],
                    "val_loss_embel": val_loss[2],
                }
            )
        
        if (epoch + 1) % 10 == 0:
            a.lr *= 0.9
            optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
            print(f"learning rate is decayed... learning rate is {a.lr}")

            # 재학습을 위해 10에포크마다 모델 저장
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "lr": a.lr,
                },
                save_path + "/model_resume_" + str(epoch + 1) + ".pth",
            )

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
