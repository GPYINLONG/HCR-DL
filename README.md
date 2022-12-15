# 手写体字符识别
本项目为机器学习课程作业，使用了MLP和MiniVGG作为分类器，使用MNIST数据集进行训练和测试，本人的两组实验（分别用MLP和MiniVGG进行识别）保存在save文件夹内，模型参数保存在其中的checkpoints文件夹中，可以直接拿来在test.py中进行测试。

## 快速上手
在进行训练前，先确保将data文件夹下的MNIST数据集解压，并且python环境满足requirements.txt。
>train.py参数如下
> 
><使用MLP训练>
> 
> python train.py --model
MLP
--root 
./data/MNIST_Dataset/
> 

若使用MiniVGG则将model的参数改为MiniVGG即可，日志以及最佳模型参数都会保存在save文件夹的对应目录下。

>test.py参数如下
> 
><使用训练好的MLP模型参数进行测试>
> 
> --model
MLP
--pth_path
<预训练的模型参数保存地址（在checkpoints文件夹里可以找到）>
--root
./data/MNIST_Dataset