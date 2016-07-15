#include "lenet.h"
#include <math.h>
#include <memory.h>
#include <omp.h>

/*
@author : 范文捷
@data	: 2016-04-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》复刻
@api	:

批量训练
void TrainBatch(LeNet5 *lenet, LeNet5 deltas[], image *input, uint8 *result, int batchSize);

训练
void Train(LeNet5 *lenet, image input, uint8 result);

预测
uint8 Predict(LeNet5 *lenet, Feature *features, image input);

初始化
void Initial(LeNet5 *lenet, double(*rand)());
*/

const static char label[LAYER7][LAYER6] =
{
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1 },
	{ -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1 },
	{ -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, +1 },
	{ +1, +1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1, +1, -1, +1, +1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1 },
	{ -1, +1, +1, +1, -1, +1, +1, +1, +1, -1, +1, +1, -1, +1, +1, -1, -1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1, +1, -1, +1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1 },
	{ +1, +1, -1, +1, +1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, -1, +1, -1, -1 },
	{ +1, +1, +1, +1, +1, +1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1 },
	{ +1, -1, +1, +1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, +1, -1 },
	{ +1, +1, +1, -1, +1, +1, -1, +1, -1, +1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, +1, -1, -1, +1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1 },
};

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define SUBSAMPLING_VALID(input,output,weight)																					\
{																																\
	FOREACH(o0,GETLENGTH(output))																								\
		FOREACH(o1,GETLENGTH(*(output)))																						\
			FOREACH(w0,GETLENGTH(weight))																						\
				FOREACH(w1,GETLENGTH(*(weight)))																				\
					(output)[o0][o1] += (input)[o0 * GETLENGTH(weight) + w0][o1 * GETLENGTH(*weight) + w1] * (weight)[w0][w1];	\
}

#define SUBSAMPLING_FULL(input,output,weight)																					\
{																																\
	FOREACH(i0,GETLENGTH(input))																								\
		FOREACH(i1,GETLENGTH(*(input)))																							\
			FOREACH(w0,GETLENGTH(weight))																						\
				FOREACH(w1,GETLENGTH(*(weight)))																				\
					(output)[i0 * GETLENGTH(weight) + w0][i1 * GETLENGTH(*weight) + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}



#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}

#define SUBSAMPLING_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		SUBSAMPLING_VALID(input[x], output[x], weight[x]);						\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define SUBSAMPLING_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		SUBSAMPLING_FULL(outerror[x], inerror[x], weight[x]);				\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		SUBSAMPLING_VALID(input[x], wd[x], outerror[x]);					\
}

#define FULLCONNECT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define FULLCONNECT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

double tanhgrad(double y)
{
	return 1 - y*y;
}

static void normalize(uint8 input[],double output[],int count)
{
	double mean = 0, std = 0;
	FOREACH(i, count)
	{
		mean += input[i];
		std += input[i] * input[i];
	}
	mean /= count;
	std = sqrt(std / count - mean*mean);
	FOREACH(i, count)
		output[i] = (input[i] - mean) / std;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->value0, features->value1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMPLING_FORWARD(features->value1, features->value2, lenet->weight1_2, lenet->bias1_2, action);
	CONVOLUTION_FORWARD(features->value2, features->value3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMPLING_FORWARD(features->value3, features->value4, lenet->weight3_4, lenet->bias3_4, action);
	CONVOLUTION_FORWARD(features->value4, features->value5, lenet->weight4_5, lenet->bias4_5, action);
	FULLCONNECT_FORWARD(features->value5, features->value6, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	FULLCONNECT_BACKWARD(features->value5, errors->value5, errors->value6, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->value4, errors->value4, errors->value5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMPLING_BACKWARD(features->value3, errors->value3, errors->value4, lenet->weight3_4, deltas->weight3_4, deltas->bias3_4, actiongrad);
	CONVOLUTION_BACKWARD(features->value2, errors->value2, errors->value3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMPLING_BACKWARD(features->value1, errors->value1, errors->value2, lenet->weight1_2, deltas->weight1_2, deltas->bias1_2, actiongrad);
	CONVOLUTION_BACKWARD(features->value0, errors->value0, errors->value1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static void load_input(Feature *features, image input)
{
	normalize((uint8 *)input, (double *)features->value0, sizeof(image) / sizeof(uint8));
}

static void load_target(Feature *features, Feature *errors, uint8 result, double(*actiongrad)(double))
{
	double *output = (double *)features->value6;
	double *error = (double *)errors->value6;
	FOREACH(i, GETLENGTH(*label))
		error[i] = (label[result][i] - output[i])*actiongrad(output[i]);
}

static uint8 get_result(Feature *features)
{
	double *output = (double *)features->value6;
	int result = -1;
	double minvalue = 1e+300;
	FOREACH(i, GETLENGTH(label))
	{
		double sum = 0;
		FOREACH(j, GETLENGTH(*label))
			sum += (output[j] - label[i][j])*(output[j] - label[i][j]);
		if (sum < minvalue)
		{
			minvalue = sum;
			result = i;
		}
	}
	return (uint8)result;
}

void TrainBatch(LeNet5* lenet, image * input, uint8 * result, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, input[i]);
		forward(lenet, &features, tanh);
		load_target(&features, &errors, result[i], tanhgrad);
		backward(lenet, &deltas, &errors, &features, tanhgrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 result)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, tanh);
	load_target(&features, &errors, result, tanhgrad);
	backward(lenet, &deltas, &errors, &features, tanhgrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, tanh);
	return get_result(&features);
}

void Initial(LeNet5 *lenet, double(*rand)())
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight1_2; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (LAYER0 + LAYER1))));
	for (double *pos = (double *)lenet->weight1_2; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_SAMPLE * LENGTH_SAMPLE * (LAYER1 + LAYER2))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight3_4; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight3_4; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_SAMPLE * LENGTH_SAMPLE * (LAYER3 + LAYER4))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL1 * LENGTH_KERNEL1 * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + LAYER6)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
