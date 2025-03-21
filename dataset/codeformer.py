from typing import Sequence, Dict, Union
import math
import time

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random


class CodeformerDataset(data.Dataset):

    def __init__(
            self,
            file_list: str,
            out_size: int,
            crop_type: str,
            use_hflip: bool,
            blur_kernel_size: int,
            kernel_list: Sequence[str],
            kernel_prob: Sequence[float],
            blur_sigma: Sequence[float],
            downsample_range: Sequence[float],
            noise_range: Sequence[float],
            jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def generate_sample(self):
        # 定义正态分布的均值和标准差
        mean = 17  # 由于范围变为7到27，均值也相应调整为17
        std_dev = 4

        # 生成正态分布样本
        sample = np.random.normal(mean, std_dev)

        # 四舍五入到最接近的整数
        rounded_sample = round(sample)

        # 确保样本在7到27的范围内
        clipped_sample = max(7, min(rounded_sample, 27))

        # 确保步长为2
        if (clipped_sample - 7) % 2 == 0:
            return clipped_sample
        else:
            # 如果不是偶数步长，调整为下一个或上一个偶数步长
            return clipped_sample + (2 - (clipped_sample - 7) % 2)

    def get_params(self, x, region_num):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size()  # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[
            0]  # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def qrandom(self, x0, region_num):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        x = torch.from_numpy(x0)
        x = x.permute(2, 0, 1)
        x = x.cuda()
        x_c = x.clone()
        EPSILON = 1

        C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x,
                                                                                       region_num)  # -> (C), (C), (C)
        # region percentiles for each channel
        region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)

        region_percentiles_per_channel = region_percentiles.reshape([-1, region_num - 1])
        region_percentiles_pos = (
                    region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1,
                                                                                                               1)
        ordered_region_right_ends_for_checking = \
        torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1) + EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1) + 1e-6], dim=1).sort(1)[
            0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (
                    x.view(C, 1, H, W) >= ordered_region_left_ends)  # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()  # sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)  索引

        # random points inside each region as the proxy for all values in corresponding regions
        proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(),
                                                  device=x.device)  # -1
        proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, region_num])  # -1
        ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (
                    ordered_region_right_ends - ordered_region_left_ends)
        proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
        x = x.clone()
        proxy = proxy_vals.type(x.dtype)
        x = proxy

        x = x.permute(1, 2, 0)
        x = x.cpu()
        x3 = x.numpy()
        return x3

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"

        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
            # 写到这
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        img_gt1 = img_gt.copy()
        random_num = self.generate_sample()
        # numbers = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
        # random_num = random.choice(numbers)

        img_gt2 = self.qrandom(img_gt1, random_num)

        img_lq = img_gt2
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)

        return dict(jpg=target, txt="", hint=source)

    def __len__(self) -> int:
        return len(self.paths)


def main():
    file_list = "/home/wzp/DiffBIR-mainSR/datasavelist/train.list"
    out_size = 512
    crop_type = "center"
    use_hflip = False
    blur_kernel_size = 41
    kernel_list = ['iso', 'aniso']
    kernel_prob = [0.5, 0.5]
    blur_sigma = [0.1, 12]
    downsample_range = [1, 10]
    noise_range = [0, 15]
    jpeg_range = [30, 60]

    dataset = CodeformerDataset(
        file_list=file_list,
        out_size=out_size,
        crop_type=crop_type,
        use_hflip=use_hflip,
        blur_kernel_size=blur_kernel_size,
        kernel_list=kernel_list,
        kernel_prob=kernel_prob,
        blur_sigma=blur_sigma,
        downsample_range=downsample_range,
        noise_range=noise_range,
        jpeg_range=jpeg_range
    )

    sample = dataset[0]
    print("Sample:", sample)

    print("Dataset size:", len(dataset))


if __name__ == "__main__":
    main()