# GARLSR

### Overview
<img src="./figure/1.png" width=100%>

**Abstract:** 
While large models bring significant performance improvements, they often lead to overfitting. In image super-resolution tasks, diffusion models are one of the representatives of generative capabilities. They usually use large models as the architecture. However, large models face serious overfitting issues when dealing with a small amount of data and highly diverse images. To highlight this phenomenon, we propose a novel Gaussian quantization representation learning method for diffusion models to enhance the model's robustness. By introducing Gaussian quantization representation learning, we can effectively reduce overfitting while maintaining model complexity. On this basis, we have constructed a multi-source infrared image dataset. It is used to emphasize the overfitting issue of large models in small-sample and diverse image reconstruction. To validate the effectiveness of our method in reducing overfitting, we conduct experiments on the constructed multi-source infrared image dataset. The experimental results show that our method outperforms previous super-resolution methods and significantly alleviates the overfitting problem of large models in complex small-sample tasks.

### Visual
<img src="./figure/2.png" width=100%>

## Preparation

### Install

We test the code on PyTorch 1.10.2 + CUDA 11.3.

1. Create a new conda environment
```
conda create -n scaleupdehazing python=3.7
conda activate scaleupdehazing
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Download

You can download the pretrained models on [[Baidu Drive](https://pan.baidu.com/s/1HPFKJaZ79dsSXOpXxie-7Q?pwd=1234)]

You can download the dataset on [[trinity](https://github.com/chi-kaichen/Trinity-Net)]


The file path should be the same as the following:

```
┬─ save_models
│   ├─ nid
│   │   ├─ scaleupdehazing.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ NID
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       └─ ...
    └─ ... (dataset name)
```


### Train
You can obtain the images after channel transfer:
```sh
python ct.py
```
You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python trainct.py --model scaleupdehazing --dataset NID --exp nid
```
```sh
python trainssl.py --model scaleupdehazing --dataset NID --exp nid
```

### Test

Run the following script to test the trained model:

```sh
python test.py --model scaleupdehazing --dataset NID --exp nid
```

## Acknowledgement
This project is build based on [Dehazeformer](https://github.com/IDKiro/DehazeFormer). We thank the authors for sharing their code.
