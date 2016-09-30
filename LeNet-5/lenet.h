/*
@author : 范文捷
@data    : 2016-09-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api	:

批量训练
void train(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);

预测
uint8 predict(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 count);

初始化
void initial(LeNet5 *lenet);
*/

#pragma once

#define LENGTH_KERNEL0	5
#define LENGTH_KERNEL1	4

#define LENGTH_FEATURE0	28
#define LENGTH_FEATURE1	24//(LENGTH_INPUT - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	12//(LENGTH_INPUT_1 >> 1)
#define LENGTH_FEATURE3	8//(LENGTH_INPUT_2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	4//(LENGTH_INPUT_3 >> 1)
#define LENGTH_FEATURE5	1//(LENGTH_INPUT_4 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE6	1

#define LAYER0			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          84

#define ALPHA 0.05

typedef unsigned char uint8_t;
typedef uint8_t image_t[LENGTH_FEATURE0][LENGTH_FEATURE0];

typedef struct LeNet5
{
	double weight0_1[LAYER0][LAYER1][LENGTH_KERNEL0][LENGTH_KERNEL0];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL0][LENGTH_KERNEL0];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL1][LENGTH_KERNEL1];
	double weight5_6[LAYER5 * LENGTH_FEATURE6 * LENGTH_FEATURE6][OUTPUT];

	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

}LeNet5;

typedef struct Feature
{
	double input[LAYER0][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];
}Feature;

void train_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT],uint8_t *labels, const int batchSize);

void predict_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT], uint8_t labelCount, const int batchSize, uint8_t *labels);

uint8_t predict(LeNet5 *lenet, image_t input, const char(*resMat)[OUTPUT], uint8_t count);

void initial(LeNet5 *lenet);
