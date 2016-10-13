# LeNet-5������

### ����
����YANN LECUN�����ġ�Gradient-based Learning Applied To Document Recognition����Ƶ�LeNet-5�����磬C����д�ɣ��������κε������⡣
MNIST��д�ַ�������ѵ��ʶ����97%�����ѵ��ʶ����98%��

����AVXָ��Ż��������ٶȴ��������

### DEMO
main.c�ļ�ΪMNIST���ݼ���ʶ��DEMO��ֱ�ӱ��뼴�����У�ѵ����60000�ţ����Լ�10000�š�

### ��Ŀ����
����ĿΪCMAKE��Ŀ���������llvm.org���clang���롣����ƽ̨�Ĵ���������֧��AVXָ���

####Linux
0.��װcmake
1.��װclang
2.ʹ��cmake����makefile
cmake -DCMAKE_C_COMPILER=clang
make
3.����
./LeNet_5

####macOS
0.��װllvm.org�ٷ�ԭ��(��ƻ�����ư棩��clang����װ�������£�
��http://llvm.org/releases/download.html ����macOS��clang
ʹ��tar xvz�����ѹ�������ļ����ڵ����ݸ��ƽ�/usr/local�ļ����ںϲ�
1.��װcmake
2.ʹ��cmake����makefile
cmake -DCMAKE_C_COMPILER=clang
make
3.����
./LeNet_5

####Windows
0.��װVisual Studio
1.�ӹ������ز���װclang,�����û�������
2.��Visual Studio������Ա������ʾ����������Ŀ�ļ��У�����
make.bat
3.����
LeNet_5.exe

### API
#####����ѵ��
lenet:  LeNet5��Ȩֵ��ָ�룬LeNet5������ĺ���

inputs�� Ҫѵ���Ķ��ͼƬ��Ӧunsigned char��ά���������,ָ��Ķ�ά�����batchSize����С�ڴ�ռ�ָ�롣��MNIST����DEMO�ж�ά����Ϊ28x28��ÿ����ά������ֵ�ֱ�Ϊ��Ӧλ��ͼ�����ػҶ�ֵ

resMat�������������

labels��Ҫѵ���Ķ��ͼƬ�ֱ��Ӧ�ı�ǩ���顣��СΪbatchSize

batchSize:����ѵ������ͼ�񣨶�ά���飩������

void train_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT],uint8_t *labels, const int batchSize);

####����Ԥ��
lenet:  LeNet5��Ȩֵ��ָ�룬LeNet5������ĺ���

inputs�� ҪԤ��Ķ��ͼƬ��Ӧunsigned char��ά���������,ָ��Ķ�ά�����batchSize����С�ڴ�ռ�ָ�롣��MNIST����DEMO�ж�ά����Ϊ28x28��ÿ����ά������ֵ�ֱ�Ϊ��Ӧλ��ͼ�����ػҶ�ֵ

resMat�������������

labelCount: �����������

batchSize: ����Ԥ������ͼ�񣨶�ά���飩������

results: Ԥ��������ĵ�ַ

void predict_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT], uint8_t labelCount, const int batchSize, uint8_t *results);

#####Ԥ��
lenet:  LeNet5��Ȩֵ��ָ�룬LeNet5������ĺ���

input:  �����ͼ�������

labels: �����������ָ��

labelCount:	�����������

return  ����ֵΪԤ��Ľ��

int predict(LeNet5 *lenet, image input, const char(*labels)[LAYER6], int labelCount);

#####��ʼ��
lenet:  LeNet5��Ȩֵ��ָ�룬LeNet5������ĺ���

void initial(LeNet5 *lenet);