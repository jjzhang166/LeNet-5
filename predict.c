#include "lenet.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

#ifndef __AVX__
#error Please Add Build Variant -mavx
#endif //__AVX__

static void vector_x_matrix(double *src,double *mat,double *des,long height,long width);
static void convolute_valid(double *src,double *conv,double *des,const long dh,const long dw,const long ch,const long cw);

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

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)                                            \
{                                                                                       \
    convolute_valid((double *)(input),(double *)(weight),(double *)(output),            \
        GETLENGTH(output),GETLENGTH(*(output)),GETLENGTH(weight),GETLENGTH(*(weight))); \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)                    \
{                                                                               \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                 \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                            \
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);                 \
	FOREACH(j, GETLENGTH(output))                                               \
		FOREACH(i, GETCOUNT(output[j]))                                         \
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);  \
}


#define SUBSAMP_MAX_FORWARD(input,output)                                                       \
{                                                                                               \
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));                                \
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));                              \
	FOREACH(i, GETLENGTH(output))                                                               \
	FOREACH(o0, GETLENGTH(*(output)))                                                           \
	FOREACH(o1, GETLENGTH(**(output)))                                                          \
	{                                                                                           \
		int x0 = 0, x1 = 0, ismax;                                                              \
		FOREACH(l0, len0)                                                                       \
			FOREACH(l1, len1)                                                                   \
		{                                                                                       \
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);                                                            \
			x1 += ismax * (l1 - x1);                                                            \
		}                                                                                       \
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];                               \
	}                                                                                           \
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)                                                \
{                                                                                                           \
    vector_x_matrix((double *)input,(double *)weight,(double *)output,GETLENGTH(weight),GETLENGTH(*weight));\
	FOREACH(j, GETLENGTH(bias))                                                                             \
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);                                    \
}

static void normalize(uint8_t input[],double output[],int count)
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
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void load_input(Feature *features, image_t input)
{
	normalize((uint8_t *)input, (double *)features->input, sizeof(image_t) / sizeof(uint8_t));
}

static uint8_t get_result(Feature *features, const char(*labels)[OUTPUT], uint8_t count)
{
	double *output = (double *)features->output;
	const int outlen = GETCOUNT(features->output);
	uint8_t result = 0;
	double minvalue = 0;
    FOREACH(j, outlen)
    minvalue += (output[j] - labels[0][j])*(output[j] - labels[0][j]);
	for (uint8_t i = 1; i < count; ++i)
	{
		double sum = 0;
		FOREACH(j, outlen)
			sum += (output[j] - labels[i][j])*(output[j] - labels[i][j]);
		if (sum < minvalue)
		{
			minvalue = sum;
			result = i;
		}
	}
	return result;
}

uint8_t predict(LeNet5 *lenet, image_t input, const char (*resMat)[OUTPUT],uint8_t labelCount)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, tanh);
	return get_result(&features, resMat, labelCount);
}

void initial(LeNet5 *lenet)
{
    srand((unsigned int)time(0));
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = (2.0 / RAND_MAX) * rand() - 1);
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (LAYER0 + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL1 * LENGTH_KERNEL1 * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

const static unsigned long long mask[] ={0x8000000000000000L,0x8000000000000000L,0x8000000000000000L,0x8000000000000000L,0,0,0,0};

static void vector_x_matrix(double *src,double *mat,double *des,const long height,const long width)
{
    const long lastw = width >> 2 << 2;
    asm("vmovupd (%0), %%ymm3;"::"r"(mask):"%ymm3");
    for(long w=0;w<width;w+=4)
    {
        if(w==lastw) asm("vmovupd (%0,%1,8), %%ymm3;"::"r"(mask),"r"(4 - (width & 3)):"%ymm3");
        asm("vmaskmovpd (%0,%1,8), %%ymm3, %%ymm0;"::"r"(des),"r"(w):"%ymm0","%ymm3");
        for(long h=0;h<height;h++)
        {
            asm("                                           \
                vbroadcastsd %0, %%ymm1;                    \
                vmaskmovpd (%1,%2,8), %%ymm3, %%ymm2;       \
                vfmadd231pd %%ymm1, %%ymm2, %%ymm0;         \
                "::"m"(src[h]),"r"(mat),"r"(h * width + w)
                :"%ymm0","%ymm1","%ymm2","%ymm3");
        }
        asm("vmaskmovpd %%ymm0, %%ymm3, (%0, %1, 8);"::"r"(des),"r"(w):"%ymm0","%ymm3");
    }
}

static void convolute_valid(double *src,double *conv,double *des,const long dh,const long dw,const long ch,const long cw)
{
    const long lastw = dw >> 2 << 2,sw = cw + dw - 1;
    asm("vmovupd (%0), %%ymm3;"::"r"(mask):"%ymm3");
    for(long d1 = 0;d1 < dw;d1 += 4)
    {
        if(d1==lastw) asm("vmovupd (%0,%1,8), %%ymm3;"::"r"(mask),"r"(4 - (dw & 3)):"%ymm3");
        for(long d0 = 0;d0 < dh;++d0)
        {
            asm("vmovupd (%0,%1,8),%%ymm0;"::"r"(des),"r"(d0 * dw + d1):"%ymm0","%ymm3");
            for(long c0=0;c0<ch;++c0)
            {
                for(long c1=0;c1<cw;++c1)
                {
                    asm("                                       \
                        vbroadcastsd %0, %%ymm1;                \
                        vmaskmovpd (%1,%2,8), %%ymm3, %%ymm2;   \
                        vfmadd231pd %%ymm2, %%ymm1, %%ymm0;     \
                        "::"m"(conv[c0 * cw + c1]),"r"(src),"r"((c0 + d0) * sw + c1 + d1)
                        :"%ymm0","%ymm1","%ymm2","%ymm3");
                }
            }
            asm("vmaskmovpd %%ymm0, %%ymm3, (%0,%1,8);"::"r"(des),"r"(d0 * dw + d1):"%ymm0","%ymm3");
        }
    }
}
