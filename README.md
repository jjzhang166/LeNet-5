# LeNet-5神经网络

### 介绍
根据YANN LECUN的论文《Gradient-based Learning Applied To Document Recognition》设计的LeNet-5神经网络，C语言写成，不依赖任何第三方库。
MNIST手写字符集初代训练识别率97%，多代训练识别率98%。

采用AVX指令集优化，运行速度大幅提升。

### DEMO
main.c文件为MNIST数据集的识别DEMO，直接编译即可运行，训练集60000张，测试集10000张。

### 项目环境
该项目为CMAKE项目，建议采用llvm.org版的clang编译。运行平台的处理器必须支持AVX指令集。

####Linux
0.安装cmake
1.安装clang
2.使用cmake生成makefile
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS=-fopenmp
make
3.运行
./LeNet_5

####macOS
0.安装llvm.org官方原版(非苹果定制版）的clang，安装步骤如下：
在http://llvm.org/releases/download.html下载macOS版clang
使用tar xvz命令解压，并将文件夹内的内容复制进/usr/local文件夹内合并
1.安装cmake
2.使用cmake生成makefile
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS=-fopenmp
make
3.运行
./LeNet_5

####Windows
0.安装Visual Studio
1.安装cmake
2.从官网下载并安装clang,并配置环境变量
3.打开Visual Studio开发人员命令提示，并进入项目文件夹
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS=-fopenmp
make
4.使用editbin修改程序栈大小为8MB
editbin /Stack:8000000 LeNet_5.exe
5.运行
LeNet_5.exe

### API
#####批量训练
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

inputs： 要训练的多个图片对应unsigned char二维数组的数组,指向的二维数组的batchSize倍大小内存空间指针。在MNIST测试DEMO中二维数组为28x28，每个二维数组数值分别为对应位置图像像素灰度值

resMat：结果向量矩阵

labels：要训练的多个图片分别对应的标签数组。大小为batchSize

batchSize:批量训练输入图像（二维数组）的数量

void train_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT],uint8_t *labels, const int batchSize);

####批量预测
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

inputs： 要预测的多个图片对应unsigned char二维数组的数组,指向的二维数组的batchSize倍大小内存空间指针。在MNIST测试DEMO中二维数组为28x28，每个二维数组数值分别为对应位置图像像素灰度值

resMat：结果向量矩阵

labelCount: 结果向量个数

batchSize: 批量预测输入图像（二维数组）的数量

results: 预测结果保存的地址

void predict_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT], uint8_t labelCount, const int batchSize, uint8_t *results);

#####预测
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

input:  输入的图像的数据

labels: 结果向量矩阵指针

labelCount:	结果向量个数

return  返回值为预测的结果

int predict(LeNet5 *lenet, image input, const char(*labels)[LAYER6], int labelCount);

#####初始化
lenet:  LeNet5的权值的指针，LeNet5神经网络的核心

void initial(LeNet5 *lenet);