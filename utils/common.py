import random
from typing import Mapping, Any
import importlib
import  torch
from torch import nn
import torchvision.transforms.functional as F


def get_obj_from_str(string: str, reload: bool=False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False


def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

# def low_high_freq_mask(img,threshold=0.1):
def low_high_freq_mask_islow(img):
        # 获取输入图像的形状
        input_shape = img.shape
        # 对输入图像进行傅里叶变换
        img_fft = torch.fft.fftn(img, dim=(-2, -1))

        # 计算频率
        h, w = img.shape[-2:]
        freqs = torch.fft.fftfreq(h)[:, None] + torch.fft.fftfreq(w)[None, :]
        freqs = torch.sqrt(freqs ** 2).to(img.device)

        # 创建掩码
        mask = freqs > 0.01
        # 遮盖低频分量
        img_fft_masked = img_fft * mask

        # 对掩盖后的频域图像进行逆傅里叶变换
        img_masked = torch.fft.ifftn(img_fft_masked, dim=(-2, -1)).real
        # 确保输出图像与输入图像具有相同的形状
        assert img_masked.shape == input_shape, f"输出图像的形状与输入图像的形状不一致，输入图像形状：{input_shape}，输出图像形状：{mask.shape}"
        return img_masked


def low_high_freq_mask_ishigh(img):
    # 获取输入图像的形状
    input_shape = img.shape
    # 对输入图像进行傅里叶变换
    img_fft = torch.fft.fftn(img, dim=(-2, -1))

    # 计算频率
    h, w = img.shape[-2:]
    freqs = torch.fft.fftfreq(h)[:, None] + torch.fft.fftfreq(w)[None, :]
    freqs = torch.sqrt(freqs ** 2).to(img.device)


    # 创建掩码
    mask = freqs < 0.01
    # 遮盖低频分量
    img_fft_masked = img_fft * mask

    # 对掩盖后的频域图像进行逆傅里叶变换
    img_masked = torch.fft.ifftn(img_fft_masked, dim=(-2, -1)).real
    # 确保输出图像与输入图像具有相同的形状
    assert img_masked.shape == input_shape, f"输出图像的形状与输入图像的形状不一致，输入图像形状：{input_shape}，输出图像形状：{mask.shape}"
    return img_masked


def low_pass_filter(img):
    # 随机生成高斯核的大小
    kernel_size = random.choice([0, 3, 5])

    # kernel_size = 5
    # 对图像进行低通滤波
    # filtered_img = F.gaussian_blur(img, kernel_size)

    if kernel_size == 3:
        filtered_img = F.gaussian_blur(img, kernel_size)
    elif kernel_size == 5:
        filtered_img = F.gaussian_blur(img, kernel_size)
    else:
        filtered_img = img

    assert filtered_img.shape == img.shape, f"输出图像的形状与输入图像的形状不一致，输入图像形状：{img.shape}，输出图像形状：{filtered_img.shape}"
    return filtered_img