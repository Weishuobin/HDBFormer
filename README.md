## 1. 🚀 Get Start

**0. Install**

```bash
conda create -n HDBFormer python=3.10 -y  
conda activate HDBFormer 

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 trchaudio=2.1.2 pytorch-cuda=11.8 -c pytc

pip install mmcv==2.1.0 -f https://download.openmmLab.com/mmcv/dist/cu118/torch2.1/index.htn

pip install tqdm opencv-python scipy tensorboardXabulate easydict ftfy regex
```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

By default, you can put datasets into the folder 'datasets' or use 'ln -s path_to_data datasets'.

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|

Compred to the original datasets, we map the depth (.npy) to .png via 'plt.imsave(save_path, np.load(depth), cmap='Greys_r')', reorganize the file path to a clear format, and add the split files (.txt).



- **Checkpoints:** 

 NYUDepth or SUNRGBD trained HDBFormer can be downloaded at:

| HDBFormer| [GoogleDrive] | [OneDrive] | [BaiduNetdisk] | 
|:---: |:---:|:---:|:---:|



**2. Train.**

You can change the `local_config' files in the script to choose the model for training. 
```
python train.py --config <The configuration file to be trained>
```

After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX is depends on the training config.


**3. Eval.**

You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 
```
python eval.py --config <The configuration file to be trained>  --continue_fpath <The location of the weight file>
```



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
