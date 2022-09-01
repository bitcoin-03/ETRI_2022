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

import torch.utils.data
import numpy as np
import cv2


class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
        fills the remaining part with a black background

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, landmarks, sub_landmarks=None):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode="constant")

        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

            new_image = np.zeros((self.output_size, self.output_size, 3))

            if h > w:
                new_image[:, (112 - new_w // 2) : (112 - new_w // 2 + new_w), :] = img
                landmarks = landmarks + [112 - new_w // 2, 0]
            else:
                new_image[(112 - new_h // 2) : (112 - new_h // 2 + new_h), :, :] = img
                landmarks = landmarks + [0, 112 - new_h // 2]

            if sub_landmarks is not None:
                sub_landmarks = sub_landmarks * [new_w / w, new_h / h]
                if h > w:
                    sub_landmarks = sub_landmarks + [112 - new_w // 2, 0]
                else:
                    sub_landmarks = sub_landmarks + [0, 112 - new_h // 2]
                return new_image, landmarks, sub_landmarks
            else:
                return new_image, landmarks
        else:
            new_image = np.zeros((self.output_size, self.output_size, 3))
            if h > w:
                new_image[:, (112 - new_w // 2) : (112 - new_w // 2 + new_w), :] = img
            else:
                new_image[(112 - new_h // 2) : (112 - new_h // 2 + new_h), :, :] = img

            return new_image


class BBoxCrop(object):
    """Operator that crops according to the given bounding box coordinates."""

    def __call__(self, image, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top : top + new_h, left : left + new_w]

        return image


class ETRIDataset_normalize(torch.utils.data.Dataset):
    """Dataset containing emotion categories (Daily, Gender, Embellishment)."""

    def __init__(self, df, base_path, type: str='train', split_col: str=None, transform = None):
        self.df = df
        self.base_path = base_path
        self.type = type
        if self.type not in ['train', 'val']:
            raise KeyError(f'Type [{self.type}] is an invalid type')
        self.split_col = split_col
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(224)
        self.to_tensor = ToTensorV2()
        self.transform = transform

    def __getitem__(self, i):
        if self.split_col:
            sample = self.df[self.df[self.split_col] == self.type].iloc[i]
        else:
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
        if self.split_col:
            return len(self.df[self.df[self.split_col] == self.type])
        else:
            return len(self.df[self.df.Split == self.type])
