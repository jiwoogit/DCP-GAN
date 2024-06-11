# [CVPR 2024] Diversity-aware Channel Pruning for StyleGAN Compression

### [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Diversity-aware_Channel_Pruning_for_StyleGAN_Compression_CVPR_2024_paper.html) / [Arxiv](https://arxiv.org/abs/2403.13548) / [Project Page](https://jiwoogit.github.io/DCP-GAN_site/)

---
**This is the "StyleGAN3" version of our approach.**
This is an **unstable** version, and the code is still in the testing.


## Usage

**To test our code, please follow these steps:**

1. [Setup](#setup)
2. [Pruning](#pruning)
3. [Dataset](#dataset)
4. [Train](#train)
5. [Inference](#inference)
6. [Evaluation](#evaluation)


### Pre-trained weights

If you want to test our lightweight model only, please download the pre-trained model from this [link](https://drive.google.com/drive/folders/189irmL8OMkynCeu4-XLPq8OGGvCNoiFA?usp=sharing) (`StyleGAN3_FFHQ256` dir) and proceed to the Inference step.


## Setup

Our codebase is built on ([xuguodong03/StyleKD](https://github.com/xuguodong03/StyleKD) and [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3)) and has similar architecture and dependencies.

I tested the code in the [pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/1.9.0-cuda11.1-cudnn8-devel/images/sha256-024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1) Docker image.

## Pruning
Copy the teacher weight 'stylegan3-ffhq-256x256.pkl' to the 'weights' directory, then run:
```
python prune_diversity.py \
    --network weights/stylegan3-ffhq-256x256.pkl \
    --outdir weights/pruned_weights
```
- **Pruning ratio ($p_r$)** is controlled by the `--pruning_ratio` parameter. (default: 0.7)
- **Strength of perturbation ($\alpha$)** is controlled by the `--edit_strength` parameter. (default: 5.0)
- **The number of perturbations for each latent vector ($N$)** is controlled through the `--n_direction` parameter. (default: 10)

## Dataset

In StyleGAN3, if you have images in a directory, you need to create a dataset in the form of a "zip" file. To do this, use the .py file `python dataset_tool.py`. In my case, I run:

```
python dataset_tool.py \
    --source /dataset/ffhq \
    --dest /dataset/ffhq.zip \
    --resolution 256x256
```
If you encounter any errors, please refer to the original StyleGAN3 repository.

## Train
This execution assumes a 2-GPU setting.
```
# Train StyleGAN3-T for FFHQ using 2 GPUs.
python train.py --outdir=./training-runs --cfg=stylegan3-t --data=/dataset/ffhq.zip \
    --gpus=2 --batch=16 --gamma=10 --mirror=0 --kimg 25000 \
    --cbase 16384 --teacher_pkl weights/stylegan3-ffhq-256x256.pkl \
    --pruned_pth weights/pruned_weights/div_pruned_0.70.pth \
    --pruning_ratio 0.7 \
    --stylekd=0
```

## Inference
Download the weights from this [link](https://drive.google.com/drive/folders/189irmL8OMkynCeu4-XLPq8OGGvCNoiFA?usp=sharing) or train yourself, and run:
```
python gen_images --network weights/dcp_ffhq_sg3.pkl --seeds 42
```

## Evaluation
### FID
```
python calc_metrics.py --metrics=fid50k_full \
    --data=/workspace/stylegan3/ffhq.zip \
    --mirror=0 \
    --network=/workspace/stylegan3/weights/stylegan3-t-ffhqu-256x256.pkl
```

## Citation
If you find our work useful, please consider citing and star:
```BibTeX
@InProceedings{Chung_2024_CVPR,
    author    = {Chung, Jiwoo and Hyun, Sangeek and Shim, Sang-Heon and Heo, Jae-Pil},
    title     = {Diversity-aware Channel Pruning for StyleGAN Compression},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {7902-7911}
}
```
