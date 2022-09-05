'''
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
'''
### 라이브러리 설치 ####
import subprocess
import sys

try:
    import albumentations as A
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations"])
finally:
    import albumentations as A
from albumentations.pytorch import ToTensorV2

##########################

from dataset import ETRIDataset_emo, ETRIDataset_normalize, ETRIDataset_emo_clothes
from networks import *
from tqdm import tqdm

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

from loss import loss_save
from sklearn.metrics import top_k_accuracy_score

class Config:
    model = "EfficientNetV2_emo_clothes"
    epochs =  100
    lr = 0.00005
    batch_size = 16
    seed = 42
    num_workers = 2
    image_size = 380
    criterion = "focal"

    name = "effV2L_" # model save at {exp_num}_모델이름"
    explan =  "test_clothes/default/pad[b]/coarse,grid[b]," # experiment description, ex. exp_efficientnet_{explan}
    resume_from =  ""  # ex) model_resume_20.pth

    # csv_path = "/content/drive/MyDrive/Fashion-How/data/task1_data/info_etri20_emotion_tr_val_simple.csv"
    csv_path = "/content/drive/MyDrive/Fashion-How/task1_data/info_etri20_emotion_tr_val_stratified_clothes.csv"
    data_path = "/content/drive/MyDrive/Fashion-How/data/task1_data/train/"
    # data_path = "/content/drive/MyDrive/Fashion-How/data/task1_data_masked_white/train/"
    # data_path = "/content/drive/MyDrive/Fashion-How/data/task1_data_masked/train/"


def to_dict(config):
    return {
        'model' : config.model,
        'epochs' : config.epochs,
        'lr' : config.lr,
        'batch_size' : config.batch_size,
        'seed' : config.seed,
        'num_workers' : config.num_workers,
        'image_size' : config.image_size,
        'criterion' : config.criterion,

        'name' : config.name,
        'explan' : config.explan,
        'resume_from' : config.resume_from,

        'csv_path' : config.csv_path,
        'data_path' : config.data_path,
    }
    

def get_transforms(need=("train", "val")):
    transformations = {}
    if "train" in need:
        transformations["train"] = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.CoarseDropout(always_apply=False, p=1.0, max_holes=1, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=(255, 255, 255), mask_fill_value=None)
                A.GridDropout(ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=(0, 0, 0), mask_fill_value=None, always_apply=False, p=0.5),
                A.CoarseDropout(always_apply=False, p=0.5, max_holes=25, max_height=25, max_width=25, min_holes=5, min_height=8, min_width=8, fill_value=(0, 0, 0), mask_fill_value=None)
                # A.HueSaturationValue(
                #     hue_shift_limit=0.2,
                #     sat_shift_limit=0.2,
                #     val_shift_limit=0.2,
                #     p=0.5,
                # ),
                # A.RandomBrightnessContrast(
                #     brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                # ),
                # A.ColorJitter(p=0.5),
            ],
            p=1.0,
        )
    if "val" in need:
        # transformations["val"] = Compose(
        #     [
        #         ToTensorV2(p=1.0),
        #     ],
        #     p=1.0,
        # )
        transformations["val"] = None
    return transformations

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("현재 DEVICE : ",DEVICE)

    set_seeds(Config.seed)

    """ The main function for model training. """
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = increment_path(os.path.join('models/', Config.name))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 모델은 parser로 network.py에 구현되어 있는 클래스 이름을 입력받아서 생성되게끔 했습니다.
    models = {
        "Baseline_ResNet_emo" : Baseline_ResNet_emo,
        "EfficientNet_emo" : EfficientNet_emo,
        "EfficientNet_emo_clothes" : EfficientNet_emo_clothes,
        "EfficientNetV2_emo" : EfficientNetV2_emo,
        "EfficientNetV2_emo_clothes" : EfficientNetV2_emo_clothes,
    }
    print("Loading model...")
    if Config.model in models:
        net = models[Config.model]().to(DEVICE)
    else:
        raise Exception('모델명이 이상해요!!')
    print("Loading data....")
    aug = get_transforms()
    df = pd.read_csv(Config.csv_path)
    
    train_dataset = ETRIDataset_emo_clothes(
        df,
        base_path=Config.data_path,
        image_size=Config.image_size,
        type="train",
        transform=aug["train"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_dataset = ETRIDataset_emo_clothes(
        df,
        base_path=Config.data_path,
        image_size=Config.image_size,
        type="val",
        transform=aug["val"],
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    if Config.resume_from:
        # 저장했던 중간 모델 정보를 읽습니다.
        path = "models/" + Config.resume_from
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])

        # optimizer = torch.optim.Adam(net.parameters(), lr=checkpoint["lr"])
        optimizer = torch.optim.AdamW(net.parameters(), lr=checkpoint["lr"], weight_decay=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:
        resume_epoch = 0
        # optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
        optimizer = torch.optim.AdamW(net.parameters(), lr=Config.lr, weight_decay=0.001)

    # criterion = nn.CrossEntropyLoss().to(DEVICE)
    criterion = loss_save(name = Config.criterion, loss_data = train_dataset, with_clothes = True)
    val_criterion = loss_save(name = Config.criterion, loss_data = val_dataset, with_clothes = True)

    total_step = len(train_dataloader)
    val_total_step = len(val_dataloader)
    step = 0
    val_step = 0
    best_val_acc = 0
    t0 = time.time()

    increment_name = save_path.split('/')[-1]
    model_explan = f"{'_'+Config.explan if Config.explan else ''}"
    config_dict = {**to_dict(Config), **aug}
    wandb.init(
        project="model-test",
        name=f"{increment_name}_{Config.model}{model_explan}",
        entity="bitcoin-etri",
        config=config_dict,
    )
    wandb.save()
    wandb.watch(net)

    targets = {
        'daily': 7,
        'gender': 6,
        'embel': 3,
        'clothes': 14,
    }

    """ define loss scaler for automatic mixed precision """
    # Creates a GradScaler once at the beginning of training.
    print("Preparing Train....")
    for epoch in range(resume_epoch, Config.epochs):
        # Train loop
        net.train()
        t1 = time.time()
        train_acc = {k:0 for k,v in targets.items()}

        scaler = torch.cuda.amp.GradScaler()

        for i, sample in enumerate(tqdm(train_dataloader, desc="tqdm-etri")):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Casts operations to mixed precision 
                step += 1
                for key in sample:
                    sample[key] = sample[key].to(DEVICE)                  

                out_daily, out_gender, out_embel, out_clothes = net(sample)

                loss_daily = criterion['daily'](out_daily, sample['daily_label'])
                loss_gender = criterion['gender'](out_gender, sample['gender_label'])
                loss_embel = criterion['embel'](out_embel, sample['embel_label'])
                loss_clothes = criterion['clothes'](out_clothes, sample['clothes_label'])

                loss = loss_daily + loss_gender + loss_embel
                loss_with_clothes = loss_daily + loss_gender + loss_embel + loss_clothes

                # train top1 accuracy
                top1_acc_cal(train_acc, sample, (out_daily, out_gender, out_embel, out_clothes), targets)

            # Scales the loss, and calls backward() 
            # to create scaled gradients 
            scaler.scale(loss_with_clothes).backward()

            # Unscales gradients and calls 
            # or skips optimizer.step() 
            scaler.step(optimizer)

            # Updates the scale for next iteration 
            scaler.update()

            if (i + 1) % 10 == 0:
                print(
                    "Train process "
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_with_clothes: {:.4f}, "
                    "Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Loss_clothes: {:.4f}, Time : {:2.3f}".format(
                        epoch + 1,
                        Config.epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        loss_with_clothes.item(),
                        loss_daily.item(),
                        loss_gender.item(),
                        loss_embel.item(),
                        loss_clothes.item(), #
                        time.time() - t0,
                    )
                )

                t0 = time.time()
            # wandb
            wandb.log(
                {
                    "loss": loss.item(),
                    "loss_with_clothes" :  loss_with_clothes.item(),
                    "loss_daily": loss_daily.item(),
                    "loss_gender": loss_gender.item(),
                    "loss_embel": loss_embel.item(),
                    "loss_clothes" : loss_clothes.item(),
                }
            )
        acc_div_dataloader(train_acc,len(train_dataloader))

        wandb.log(
            {
                "train_acc_daily": train_acc['daily'],
                "train_acc_gender": train_acc['gender'],
                "train_acc_embel": train_acc['embel'],
                "train_acc_clothes": train_acc['clothes'],

                "train_acc_with_clothes": (train_acc['daily'] + train_acc['gender'] + train_acc['embel'] + train_acc['clothes'])/4,
                "train_acc": (train_acc['daily'] + train_acc['gender'] + train_acc['embel'])/3,
            }
        )
        
        
        with torch.no_grad():
            print("Calculating validation results...")
            net.eval()
            val_images = []
            val_loss_items = {f'{k}_val_loss':[] for k,v in targets.items()}
            print('val_loss_items', val_loss_items)
            val_acc = {k:0 for k,v in targets.items()}
            # Validation loop
            for i, val_batch in enumerate(val_dataloader):
                # i는 미니배치 개수, validation 전체 /batch
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(DEVICE)

                val_out_daily, val_out_gender, val_out_embel, val_out_clothes = net(val_batch)

                preds = (
                    torch.argmax(val_out_daily, dim=-1),
                    torch.argmax(val_out_gender, dim=-1),
                    torch.argmax(val_out_embel, dim=-1),
                    torch.argmax(val_out_clothes, dim=-1),
                )
                                # val loss
                for i, (k, v) in enumerate(targets.items()):
                    temp_pred = (val_out_daily, val_out_gender, val_out_embel, val_out_clothes)
                    val_loss_items[f'{k}_val_loss'].append(val_criterion[f'{k}'](temp_pred[i], val_batch[f'{k}_label']).item())


                daily_list = ['실내복', '가벼운 외출', '오피스룩', '격식차린', '이벤트', '교복', '운동복']
                gender_list = ['밀리터리', '매니쉬', '유니섹스', '걸리쉬', '우아한', '섹시한']
                embellishment_list = ['장식이 없는', '포인트 장식이 있는', '장식이 많은']
                clothes_list = ['Blouse', 'Cardigan', 'Coat', 'Jacket', 'Jumper', 'Knit', 'Onepeice', 'Pants', 'Shirt', 'Skirt', 'Sweater', 'Vest', 'BO', 'BT']

                # val top1 accuracy
                top1_acc_cal(val_acc, val_batch, (val_out_daily, val_out_gender, val_out_embel, val_out_clothes), targets)

                if len(val_images) < 108:
                    image_cnt = 1
                    if len(val_dataloader) <=108:
                        image_cnt = 108//len(val_dataloader)
                        # val_batch : 4 , val_dataloader : 36
                    for j in range(image_cnt):
                        d = random.randint(0, len(val_batch['daily_label'])-1)
                        image_check = image_correct_check(
                            daily_list[preds[0][d].item()],
                            daily_list[val_batch["daily_label"][d]], 
                            gender_list[preds[1][d].item()],
                            gender_list[val_batch["gender_label"][d]], 
                            embellishment_list[preds[2][d].item()],
                            embellishment_list[val_batch["embel_label"][d]],
                            clothes_list[preds[3][d].item()],
                            clothes_list[val_batch["clothes_label"][d]],
                        )
                        if not image_check: continue
                        val_images.append(
                            wandb.Image(
                                val_batch['image'][d],
                                caption='''
                                Pred/Truth
                                일상성 : {} / {}, 
                                성 : {} / {}, 
                                장식성 : {} / {},
                                종류 : {} / {}
                                '''.format(
                                    *image_check
                                    )
                            )
                        )

            val_loss = [np.sum(list(val_loss_items.values())[i])/len(val_dataloader) for i in range(4)]
            acc_div_dataloader(val_acc,len(val_dataloader))
            total_val_acc = (val_acc['daily'] + val_acc['gender'] + val_acc['embel'])/3
            print(
                "Validation process "
                "Epoch [{}/{}], Loss_with_clothes: {:.4f}, Loss: {:.4f}, "
                "Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Loss_clothes: {:.4f}, "
                "val_acc_daily: {:.4f}, val_acc_gender: {:.4f}, val_acc_embel: {:.4f}, val_acc_clothes: {:.4f}, valid_acc_with_clothes: {:.4f}, valid_acc: {:.4f}, Time : {:2.3f}"
                .format(
                    epoch + 1,
                    Config.epochs,
                    np.sum(val_loss),
                    np.sum(val_loss) - val_loss[3],
                    val_loss[0],
                    val_loss[1],
                    val_loss[2],
                    val_loss[3],
                    val_acc['daily'],
                    val_acc['gender'],
                    val_acc['embel'],
                    val_acc['clothes'],
                    (val_acc['daily'] + val_acc['gender'] + val_acc['embel'] + val_acc['clothes'])/4,
                    total_val_acc,
                    time.time() - t0,
                )
            )
            if total_val_acc > best_val_acc: 
                print(
                    f"New best model for val_acc score : {total_val_acc:4.4}! saving the best model.."
                )
                torch.save(
                    net.state_dict(), save_path + "/model_best" + ".pkl"
                )  
                best_val_acc = total_val_acc
                
            t0 = time.time()
            # wandb
            wandb.log(
                {
                    "Examples": val_images,
                    "val_loss_daily": val_loss[0],
                    "val_loss_gender": val_loss[1],
                    "val_loss_embel": val_loss[2],
                    "val_loss_clothes": val_loss[3],
                    "val_loss": np.sum(val_loss) - val_loss[3],
                    "val_loss_with_clothes": np.sum(val_loss),
                    "val_acc_daily": val_acc['daily'],
                    "val_acc_gender": val_acc['gender'],
                    "val_acc_embel": val_acc['embel'],
                    "val_acc_clothes" : val_acc['clothes'],
                    "val_acc_with_clothes" : (val_acc['daily'] + val_acc['gender'] + val_acc['embel'] + val_acc['clothes'])/4,
                    "val_acc" : total_val_acc,
                }
            )
        

        if ((epoch + 1) % 10 == 0):
            Config.lr *= 0.9
            # optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
            optimizer = torch.optim.AdamW(net.parameters(), lr=Config.lr, weight_decay=0.001)
            print(f"learning rate is decayed... learning rate is {Config.lr}")

            # 재학습을 위해 10에포크마다 모델 저장
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "lr": Config.lr,
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

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def acc_div_dataloader(target,div):
    for k, v in target.items():
        target[k] /= div

def top1_acc_cal(target, sample, temp_pred, label_targets):
    for i, (k, v) in enumerate(label_targets.items()):
        target[f'{k}'] += top_k_accuracy_score(
            sample[f'{k}_label'].detach().cpu().numpy(),
            temp_pred[i].detach().cpu().numpy(), k=1,
            labels=[j for j in range(v)]
        )
def image_correct_check(dp,dl,gp,gl,ep,el,cp,cl):
    if dp==dl and gp==gl and ep==el and cp==cl:
        return False
    return [dp,dl,gp,gl,ep,el,cp,cl]


if __name__ == '__main__':
    main()

