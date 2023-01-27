# GAN
## 项目简介
 GAN的全称是Generative adversarial network，中文翻译过来就是生成对抗网络。生成对抗网络其实是两个网络的组合：生成网络（Generator）负责生成模拟数据；判别网络Discriminator）负责判断输入的数据是真实的还是生成的。生成网络要不断优化自己生成的数据让判别网络判断不出来，判别网络也要优化自己让自己判断得更准确。二者关系形成对抗，因此叫对抗网络。

本项目使用3个数据集，分别为MNIST数据集，CIFAR10数据集，CIFAR100数据集，依次检验在对抗下网络生成图片的能力，并对比性能。

## 文件结构
```
    .
    |—— data #数据集
    |—— train #保存的模型
    |—— log #tensorboard保存的文件地址
    *.py #代码文件
    Makefile
    README.md
    environment.yml
    .gitignore
```

## 环境配置
```
conda env create -f environment.yml
```

