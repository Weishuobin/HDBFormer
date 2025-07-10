# HDBFormer: Efficient RGB-D Semantic Segmentation With a Heterogeneous Dual-Branch Framework
🌟 Welcome to the official code repository for [HDBFormer: Efficient RGB-D Semantic Segmentation With a Heterogeneous Dual-Branch Framework](https://arxiv.org/abs/2504.13579). We're excited to share our work with you!

🌟 Our work has been accepted by **IEEE Signal Processing Letters 2024**!

**0. Install**

```bash
conda create -n HDBFormer python=3.10 -y  
conda activate HDBFormer 

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex thop

pip install "numpy<2" --upgrade

```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

Create a folder named datasets in the root directory of the project for storing the two indoor RGB-D semantic segmentation datasets, NYUDepthv2 and SUNRGBD. The datasets should be handled in a way that strictly follows the standard process of the [DFormer](https://github.com/VCIP-RGBD/DFormer) project, for details, please refer to the section on dataset preparation in the project documentation. Links to the relevant datasets are provided below:

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|




- **Checkpoints:** 

 NYUDepthv2 or SUNRGBD trained HDBFormer can be downloaded at:

| HDBFormer| [GoogleDrive](https://drive.google.com/drive/folders/1ds3OILB7-WDe7JwSpuX1OJA6_QqOoJNe?usp=drive_link) | [OneDrive] | [BaiduNetdisk] | 
|:---: |:---:|:---:|:---:|



**2. Train.**

You can change the `local_config' files in the script to choose the model for training. 

If you want to train NYUDepthv2 dataset
```
python train.py --config local_configs.NYUDepthv2.HDBFormer --gpus 1
```
If you want to train SUNRGBD dataset
```
python train.py --config local_configs.SUNRGBD.HDBFormer --gpus 1
```
After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX is depends on the training config.


**3. Eval.**

You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 

If you want to eval NYUDepthv2 dataset
```
python eval.py --config local_configs.NYUDepthv2.HDBFormer --continue_fpath checkpoints/NYUDepthv2_bestmiou
```
If you want to eval SUNRGBD dataset
```
python eval.py --config local_configs.SUNRGBD.HDBFormer --continue_fpath checkpoints/SUNRGBD_bestmiou
```

>  If you have any questions or suggestions about our work, feel free to contact me via e-mail (weishuobin@gmail.com) or raise an issue. 

## Reference
You may want to cite:
```
@article{wei2024hdbformer,
  title={HDBFormer: Efficient RGB-D Semantic Segmentation with A Heterogeneous Dual-Branch Framework},
  author={Wei, Shuobin and Zhou, Zhuang and Lu, Zhengan and Yuan, Zizhao and Su, Binghua},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE}
}
```


### Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1) and [DFormer](https://github.com/VCIP-RGBD/DFormer)   Thanks for their authors.



### License

Code in this repo is for non-commercial use only.
