# UPerNet
## Model Structure

TODO

## How to use
有两种运行程序的方法，分别描述如下。
### 方式一
命令行方式运行训练程序并手动传入参数：
```
cd UPerNet
conda activate your_conda_environment
python ./train.py --cfg ./config/train.yaml --train_img_dir path/to/your/training/image/directory --train_seg_dir path/to/your/training/segmentation/directory --val_img_dir path/to/your/validation/image/directory --val_seg_dir path/to/your/validation/segmentation/directory --test_img_dir path/to/your/testing/image/directory

... ...

Model Summary:  319 layers; 31349752 parameters; 31349752 gradients; 16.708681728 GFLOPs
    epoch       tot         imgsz          lr         pixel_acc         time(s)
#    1         2.658        448        1.850e-03        0.000                    : 100%|██████████████████| 186/186 [01:37<00:00,  1.91it/s]
#    2         2.495        448        3.697e-03        0.000          97.3      : 100%|██████████████████| 186/186 [01:31<00:00,  2.03it/s]
#    3         2.496        448        5.540e-03        0.000          91.6      : 100%|██████████████████| 186/186 [01:31<00:00,  2.03it/s]
```
其他与训练相关的参数配置统一到UPerNet/config/train.yaml进行设置。
### 方式二
通过在配置文件中配置相关参数运行程序，使用任意文本编辑器，打开并编辑UPerNet/config/train.yaml文件。其中必须要设置的参数有：
+ train_img_dir;
+ train_seg_dir
+ val_img_dir;
+ val_seg_dir;
+ test_img_dir;

然后在命令行中运行训练脚本：

```
cd UPerNet
conda activate your_conda_environment
python ./train.py

... ...

Model Summary:  319 layers; 31349752 parameters; 31349752 gradients; 16.708681728 GFLOPs
    epoch       tot         imgsz          lr         pixel_acc         time(s)
#    1         2.658        448        1.850e-03        0.000                    : 100%|██████████████████| 186/186 [01:37<00:00,  1.91it/s]
#    2         2.495        448        3.697e-03        0.000          97.3      : 100%|██████████████████| 186/186 [01:31<00:00,  2.03it/s]
#    3         2.496        448        5.540e-03        0.000          91.6      : 100%|██████████████████| 186/186 [01:31<00:00,  2.03it/s]
```

### 预训练权重
| model name   | download |   pwd |
| :------------- | :----------: | ------------: |
| cityspace        |    xxx     |         xxx |

使用预训练权重训练自己的数据集：
+ 下载权重文件；
+ 编辑UPerNet/config/train.yaml文件，将```pretrained_model_path```配置项设置为预训练权重的文件路径；
