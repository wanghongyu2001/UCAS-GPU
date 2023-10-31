# UCAS-GPU
## 运行
首先进入hw11 运行LeNetV1.py生成参数文件，进入hw12，编译mainV1.cu。运行命令```./a.out ../hw11```。可能会出现读取数据集失败的问题，image的数据在/data文件夹里面，这个时候修改main函数中read_mnist_images的参数即可。
