# image scene classification


## 1) 软件概述
   为了完成该image scene classification的任务，经过多次训练与选择， 我最终采用的是resnet深度神经网络，
   特别的我用的是resnet50，并在imagenet finetune的情况下针对该数据集进行训练测试.

## 2) 软硬件要求
软件
- pytorch 0.4.0
- torchvision 0.2.1
- cnn_finetune 0.3
- logging 0.5.1
- numpy, scipy, matplotlib, pandas
- python 3.6(建议直接装anaconda5.1)
- cuda 9.0 + cudnn 7.0

硬件
- cpu 内存8G以上
- NVIDIA GPU内存8G以上
- 硬盘20G以上(存储预训练模型+pytorch、python啥的)


## 3) 部署流程
（1）下载训练数据集解压并更名为image_scene_data, 划分数据集为train和
valid(查看image_scene_data目录下是否有train_list.csv和valid_list.csv, 
如果没有输入以下命令生成, 可更改划分比例,默认9：1)

    python myutils.py

（2）训练，(第一次运行会自动下载imagenet预训练模型)

    python  finetune_scene_train.py  --arch res50 --epochs 100


（3）预测,  (确保模型文件在相应目录下)
 
    python  finetune_scene_train.py --predict --test_dir test_b
  
  (4) 详细参数设置说明,运行以下命令
        
       python finetune_scene_train.py --help
       
   结果：
   
       usage: finetune_scene_train.py [-h] [--arch ARCH] [--num_classes NUM_CLASSES]
                                   [-j N] [--epochs N] [--start_epoch N] [-b N]
                                   [--lr LR] [--momentum M] [--weight_decay W]
                                   [--optim OPTIM] [--print_freq N]
                                   [--resume PATH] [--log_path LOG_PATH]
                                   [--test_dir TEST_DIR] [--pretrained] [-e]
                                   [--predict]

        PyTorch fine tune scene data Training
        
        optional arguments:
          -h, --help            show this help message and exit
          --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                                densenet161 | densenet169 | densenet201 | inception_v3
                                | resnet101 | resnet152 | resnet18 | resnet34 |
                                resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                                vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                                | vgg19_bn (default: resnet50)
          --num_classes NUM_CLASSES
                                num of classes to classify (default: 20)
          -j N, --workers N     number of data loading workers (default: 4)
          --epochs N            number of total epochs to run (default: 50)
          --start_epoch N       manual epoch number (useful on restarts)
          -b N, --batch_size N  mini-batch size (default: 64)
          --lr LR, --learning_rate LR
                                initial learning rate (default 0.0001)
          --momentum M          momentum (default: 0.9)
          --weight_decay W, --wd W
                                weight decay (default: 1e-4)
          --optim OPTIM, --op OPTIM
                                use what optimizer (default: momentum)
          --print_freq N, -p N  print frequency (default: 104 batch)
          --resume PATH         path to latest checkpoint (default: none)
          --log_path LOG_PATH   path to save logs (default: result/res34/log.log)
          --test_dir TEST_DIR   test data dir (default: test_a)
          --pretrained          use pre-trained model (default: true)
          -e, --evaluate        evaluate model on validation set
          --predict             use model to do prediction
    
    

## 4）涉及到的非官方提供的数据集及其获取方式
  无, 所使用数据均是来自比赛提供的数据