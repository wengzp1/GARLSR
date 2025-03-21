# GARLSR

### Overview
<img src="./figure/1.png" width=100%>

**Abstract:** 
While large models bring significant performance improvements, they often lead to overfitting. In image super-resolution tasks, diffusion models are one of the representatives of generative capabilities. They usually use large models as the architecture. However, large models face serious overfitting issues when dealing with a small amount of data and highly diverse images. To highlight this phenomenon, we propose a novel Gaussian quantization representation learning method for diffusion models to enhance the model's robustness. By introducing Gaussian quantization representation learning, we can effectively reduce overfitting while maintaining model complexity. On this basis, we have constructed a multi-source infrared image dataset. It is used to emphasize the overfitting issue of large models in small-sample and diverse image reconstruction. To validate the effectiveness of our method in reducing overfitting, we conduct experiments on the constructed multi-source infrared image dataset. The experimental results show that our method outperforms previous super-resolution methods and significantly alleviates the overfitting problem of large models in complex small-sample tasks.

### Visual
<img src="./figure/2.png" width=100%>

## Preparation

### Install

First,create an environment with python = 3.9.

1. Create a new conda environment
```
conda create -n garlsr python=3.9

conda activate garlsr
```

2. Install dependencies
```
pip install -r requirements.txt
```

### Data
Data Preparation
1. Generate file list of training set and validation set.

    ```shell
    python scripts/make_file_list.py \
    --img_folder [hq_dir_path] \
    --val_size [validation_set_size] \
    --save_folder [save_dir_path] \
    --follow_links
    ```
    
    This script will collect all image files in `img_folder` and split them into training set and validation set automatically. You will get two file lists in `save_folder`, each line in a file list contains an absolute path of an image file:
    
    ```
    save_folder
    ├── train.list # training file list
    └── val.list   # validation file list
    ```
2. Configure training set and validation set.

    For image restoration, fill in the following configuration files with appropriate values.

    - [training set](configs/dataset/train.yaml) and [validation set](configs/dataset/val.yaml) for degradation.

### Train
1.Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to provide generative capabilities.

```shell
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
```
2.train
```sh
python train.py --config configs/train_swinir.yaml
```
3.Create the initial model weights.

```shell
python scripts/make_stage2_init_weight.py \
--cldm_config configs/model/cldm.yaml \
--sd_weight [sd_v2.1_ckpt_path] \
--swinir_weight [swinir_ckpt_path] \
--output [init_weight_output_path]
 ```
4.train
```shell
python train.py --config configs/train_cldm.yaml
```



### Test

Run the following script to test the trained model:

```shell
python inference.py \
--input inputs/demo/general \
--config configs/model/cldm.yaml \
--ckpt weights/your_generatecldm.ckpt \
--reload_swinir --swinir_ckpt weights/your_generateswinir.ckpt \
--steps 50 \
--sr_scale 4 \
--color_fix_type wavelet \
--output results/demo/general \
--device cuda [--tiled --tile_size 512 --tile_stride 256]
```


## Acknowledgement
This project is build based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR). We thank the authors for sharing their code.
