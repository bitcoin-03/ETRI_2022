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
from xml.dom import NotFoundErr
from torchvision import transforms
from skimage import io, transform, color
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.utils.data
import numpy as np
import cv2
import os
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image

class ETRIDataset_emo(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

    def __init__(self, df, base_path, image_size, type: str='train', transform = None):
        self.df = df
        self.base_path = base_path
        self.type = type
        if self.type not in ['train', 'val', 'test']:
            raise KeyError(f'Type [{self.type}] is an invalid type')
        self.image_size = image_size
        if isinstance(self.image_size, int) == False:
            raise KeyError(f'Type [{self.image_size}] is an invalid type')

        self.pretransform = A.Compose([
            A.LongestMaxSize(max_size=self.image_size, p=1.0),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p=1.0)
        ])
        self.transform = transform
        self.posttransform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.some = 0

        self.sample = df[df['Split'] == type]
        if self.type is 'test':
            self.some = -1

        self.images = []

        start = time.time()
        for idx, row in tqdm(enumerate(self.sample.itertuples())):
            bbox_xmin = row[3+self.some]
            bbox_ymin = row[4+self.some]
            bbox_xmax = row[5+self.some]
            bbox_ymax = row[6+self.some]
            image = Image.open(base_path + row[2+self.some])
            image = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
            image = np.asarray(image)[...,:3]

            self.images.append(image)
        print(f"{self.type} image load time : {time.time() - start}")

    def __getitem__(self, i):

        image=self.pretransform(image=self.images[i])['image']
        if self.transform:
            image = self.transform(image=image)["image"]
        image=image.astype(np.float32)
        image=self.posttransform(image=image)["image"]

        ret = {}
        ret['image'] = image
        ret['daily_label'] = self.sample['Daily'].iloc[i]   
        ret['gender_label'] = self.sample['Gender'].iloc[i]
        ret['embel_label'] = self.sample['Embellishment'].iloc[i]


        return ret

    def __len__(self):
        return len(self.df[self.df.Split == self.type])


class ETRIDataset_emo_clothes(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

    def __init__(self, df, base_path, image_size, type: str='train', transform = None):
        self.df = df
        self.base_path = base_path
        self.type = type
        if self.type not in ['train', 'val', 'test']:
            raise KeyError(f'Type [{self.type}] is an invalid type')
        self.image_size = image_size
        if isinstance(self.image_size, int) == False:
            raise KeyError(f'Type [{self.image_size}] is an invalid type')

        self.pretransform = A.Compose([
            A.LongestMaxSize(max_size=self.image_size, p=1.0),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p=1.0)
        ])
        self.transform = transform
        self.posttransform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.some = 0
        self.sample = df[df['Split'] == type]
        if self.type is 'test':
            self.some = -1

        self.images = []

        start = time.time()
        for idx, row in tqdm(enumerate(self.sample.itertuples())):
            bbox_xmin = row[3+self.some]
            bbox_ymin = row[4+self.some]
            bbox_xmax = row[5+self.some]
            bbox_ymax = row[6+self.some]
            image = Image.open(base_path + row[2+self.some])
            image = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
            image = np.asarray(image)[...,:3]
            self.images.append(image)
        print(f"{self.type} image load time : {time.time() - start}")

    def __getitem__(self, i):

        image=self.pretransform(image=self.images[i])['image']
        if self.transform:
            image = self.transform(image=image)["image"]
        image=image.astype(np.float32)
        image=self.posttransform(image=image)["image"]

        ret = {}
        ret['image'] = image
        ret['daily_label'] = self.sample['Daily'].iloc[i]   
        ret['gender_label'] = self.sample['Gender'].iloc[i]
        ret['embel_label'] = self.sample['Embellishment'].iloc[i]
        if self.type is not 'test':
            ret['clothes_label'] = self.sample['Clothes'].iloc[i]


        return ret

    def __len__(self):
        return len(self.df[self.df.Split == self.type])

class ETRIDataset_normalize(torch.utils.data.Dataset):
    """Dataset containing emotion categories (Daily, Gender, Embellishment)."""

    def __init__(self, df, base_path, type: str = "train", transform=None):
        self.df = df
        self.base_path = base_path
        self.type = type
        if self.type not in ["train", "val"]:
            raise KeyError(f"Type [{self.type}] is an invalid type")
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(224)
        self.to_tensor = ToTensorV2()
        self.transform = transform

    def __getitem__(self, i):
        sample = self.df[self.df.Split == self.type].iloc[i]
        image = cv2.imread(self.base_path + sample["image_name"])
        if image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        daily_label = sample["Daily"]
        gender_label = sample["Gender"]
        embel_label = sample["Embellishment"]
        bbox_xmin = sample["BBox_xmin"]
        bbox_ymin = sample["BBox_ymin"]
        bbox_xmax = sample["BBox_xmax"]
        bbox_ymax = sample["BBox_ymax"]

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image=image)["image"]
            image = self.background(image, None)
            image = self.to_tensor(image=image)["image"]
            image = image.type(torch.float32)
        else:
            image = self.background(image, None)
            image = self.to_tensor(image=image)["image"]
            image = image.type(torch.float32)

        ret = {}
        ret["image"] = image
        ret["daily_label"] = daily_label
        ret["gender_label"] = gender_label
        ret["embel_label"] = embel_label

        return ret

    def __len__(self):
        return len(self.df[self.df.Split == self.type])

# class ETRIDataset_emo(torch.utils.data.Dataset):
#     """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

#     def __init__(self, df, base_path, type: str='train', transform = None):
#         self.df = df
#         self.base_path = base_path
#         self.type = type
#         if self.type not in ['train', 'val']:
#             raise KeyError(f'Type [{self.type}] is an invalid type')
#         self.bbox_crop = BBoxCrop()
#         self.background = BackGround(224)
#         self.to_tensor = transforms.ToTensor()
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                               std=[0.229, 0.224, 0.225])
#         self.transform = transform

#         # for vis
#         self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
#         self.to_pil = transforms.ToPILImage()

#     def __getitem__(self, i):
#         sample = self.df[self.df.Split == self.type].iloc[i]
#         # image = io.imread(self.base_path + sample['image_name'])
#         image = cv2.imread(self.base_path + sample['image_name'])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # if image.shape[2] != 3:
#         #     image = color.rgba2rgb(image)
#         daily_label = sample['Daily']
#         gender_label = sample['Gender']
#         embel_label = sample['Embellishment']
#         bbox_xmin = sample['BBox_xmin']
#         bbox_ymin = sample['BBox_ymin']
#         bbox_xmax = sample['BBox_xmax']
#         bbox_ymax = sample['BBox_ymax']

#         image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
#         image = self.background(image, None)

#         image_ = image.copy()

#         # image_ = self.to_tensor(image_)
#         # image_ = self.normalize(image_)
#         image_=image_.astype(np.float32)

#         if self.transform:
#             image_ = self.transform(image=image_)["image"]


#         ret = {}
#         ret['ori_image'] = image
#         ret['image'] = image_
#         ret['daily_label'] = daily_label
#         ret['gender_label'] = gender_label
#         ret['embel_label'] = embel_label

#         return ret

#     def __len__(self):
#         return len(self.df[self.df.Split == self.type])

