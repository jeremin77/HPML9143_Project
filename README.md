<p align="center">
<a href="https://engineering.nyu.edu/"><img src="https://user-images.githubusercontent.com/68700549/118066006-eaf92080-b36b-11eb-9116-9f8e02a79534.png" align="center" height="100"></a>
</p>

<div align="center"> 
  
## New York University

 </div>

<div align="center"> 
  
# Adan and WDPruning enhanced Vision Transformer
  
#### Project  Repo  for  ECE-GY  9143  Intro To High Performance Machine Learning. Course  Instructor:  Dr.  Parijat Dube

#### Jieming Ma, Siyuan Wang
  
</div> 
<div align="center">

## INTRODUCTION

</div>

<div align="justify"> 
Transformer models have demonstrated their promising potential and achieved excellent performance on a series of computer vision tasks. However, the huge computational cost of vision transformers hinders their deployment and application to edge devices. In this task, we will explore some techniques to reduce training time and Flops but the result does not compromise the accuracy.
 
In order to get familiar with high performance machine learning techniques, we want to implement a state of art optimization algorithm Adan[1] called Adaptive Nesterov Momentum Algorithm and Width and Depth Pruning of VIT[2], a pruning method for Vision Transformer network to reduce computational redundancy while minimizing the accuracy loss. After that, we will make experiments on evaluation between our optimized model and original model to prove the performance benefits.

</div> 

  
## Requirements

</div>

- torch>=1.8.0
- torchvision>=0.9.0
- timm==0.4.9
- h5py
- scipy
- scikit-learn
  
Model preparation: download pre-trained Vit models for pruning:
```
sh download_pretrain.sh
```
  
## Code Structure
  
</div>

- `.py` -- Run for  training
- `run_hpc.py` - Run for normal training
- `.py` - Run for 
- `utilization.py` - See GPU utilization for Current Machines
- `plot_hpml.py` - Show the graph of calculation result 

## Training command

### 1. Train VIT baseline with/without Adan

If you want to run on colab or Jupyter notebook, here is the notebook file:

[Train_Vit](https://github.com/stony0411/Mini-Project-02/tree/main/Stark/experiments/stark_s)

Train on HPC:

```python vit_adan.py```

### 2. Train Width and depth Pruning for VIT 

Prune and Train on HPC:

```
python main_wdpruning.py --arch deit_base --data-set CIFAR10 --batch-size 128 --data-path ../data/ --output_dir /cifar --classifiers 10 
```

Only prune on width:


```
python main_wdpruning.py --arch deit_base --data-set CIFAR10 --batch-size 128 --data-path ../data/ --output_dir /cifar --classifiers 12
```

Test the amout of parameters, GPU throughput of pruned transformer:
```
python masked_parameter_count.py --arch deit_base --pretrained_dir logs/checkpoint.pth --eval_batch_size 1024 --classifiers 10 --classifier_choose 10
```
Note that '--classifier_choose' means choose which classifier to prune. '--classifier_choose 12' means choose the last classifier. 


## Results and evaluation

<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/1.png" width="900" height="300" >
<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/2.png" width="900" height="300" >
<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/3.png" width="900" height="300" >
<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/4.png" width="900" height="300" >
<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/5.png" width="900" height="300" >
<img src="https://github.com/jeremin77/HPML9143_Project/blob/main/results/6.png" width="900" height="300" >

## Reference

[1.Aadn](https://github.com/sail-sg/Adan)

[2.Width and Depth Pruning for Vision Transformers](https://www.aaai.org/AAAI22Papers/AAAI-2102.FangYu.pdf)
