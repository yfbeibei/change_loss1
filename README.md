# IOCFormer (Indiscernible Object Counting in Underwater Scenes)

[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Indiscernible_Object_Counting_in_Underwater_Scenes_CVPR_2023_paper.pdf)]

# Environment
python 3.8.5  
pytorch 1.7.1+cu110 
torchvision 0.8.2+cu110
h5py 3.7.0
opencv-python 4.6.0.66  
scipy   
pillow  
imageio   
nni   
mmcv  
tensorboard  

<!-- # Datasets
- Download JHU-CROWD ++ dataset from [here](http://www.crowd-counting.com/)  
- Download NWPU-Crowd dataset (resized) from [here](https://pan.baidu.com/s/1aqiLFU6lo3F_HqeT6wbEjg), password: 04i4
 -->

# Prepare data
## Download datasets
Download IOCfish5K from [here](https://drive.google.com/file/d/1ETY_AdJB9azzja6L9URN58KtL4OH98SL/view?usp=sharing)

## Generate point map
```cd IOC/data```  
```python prepare_ioc.py --data_path /xxx/xxx/downloaded_dataset```

## Generate image list
```cd IOC```    
```python make_npydata_ioc.py --data_path /xxx/xxx/downloaded_dataset```

# Training 
```cd IOC``` 
Then run
```
  python -m torch.distributed.launch --nproc_per_node=4 --master_port 8219 train_distributed.py --gpu_id '0,1,2,3' \
  --gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-5 \
  --batch_size 8 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
  --dataset cod --crop_size 256 --pre None --test_per_epoch 20  --test_patch --save --save_path exp1 \
   --dm_count  --dilation --branch_merge --branch_merge_way 2 --transformer_flag merge3 
 ``` 
 Please set the path ```exp_save``` (where to save log/model) in ```train_distributed.py``` and ```test.py```.

# Inference
Download the trained weights from [here](https://drive.google.com/file/d/12ah09QN8z6rW9N7esDhm0Cz2nrDWScu_/view?usp=sharing) and put it in ```weights``` folder.
To generate the number reported in the paper, please run:  
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 8219   test.py --dataset cod --pre weights/model.pth \
 --gpu_id 0  --only_dm --dilation --num_queries 700 --crop_size 256 --dm_count \
 --dilation --branch_merge --branch_merge_way 2 --transformer_flag merge3 --with_weights 
```   
Using the provided model, you should get
| IOCfish5K (val set) | MAE | MSE |
| :-------------------- | :-------- | :----- |
| Original paper                   | 15.91   | 34.08     |
 
<!-- # Video Demo
To do -->

# Acknowledgement
Thanks for the following great work:
```
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

```
@inproceedings{meng2021conditional,
  title={Conditional detr for fast training convergence},
  author={Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3651--3660},
  year={2021}
}
```

```
@article{liang2022end,
  title={An end-to-end transformer model for crowd localization},
  author={Liang, Dingkang and Xu, Wei and Bai, Xiang},
  journal={European Conference on Computer Vision},
  year={2022}
}
```
# Reference
If you find this project is useful, please cite:
```
@inproceedings{sun2023indiscernible,
  title={Indiscernible Object Counting in Underwater Scenes},
  author={Sun, Guolei and An, Zhaochong and Liu, Yun and Liu, Ce and Sakaridis, Christos and Fan, Deng-Ping and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13791--13801},
  year={2023}
}
```

# Questions
Please contact sunguolei.kaust@gmail.com