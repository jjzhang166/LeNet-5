//
//  train.c
//  LeNet-5
//
//  Created by fanwenjie on 2016/9/21.
//  Copyright © 2016年 Fan Wen Jie. All rights reserved.
//

#include "lenet.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define SZYMM   32

#define SZPACK (SZYMM / sizeof(double))

#define MAXSZALIGN(type) (SZYMM - 1 + sizeof(type))


typedef double pack[SZPACK];

typedef struct FeaturePack
{
    pack layer0[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    pack layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    pack layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    pack layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    pack layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    pack layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    pack output[OUTPUT];
}FeaturePack;

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(pack))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

static void convolute_valid1(pack *src,
                            double *conv,
                            pack *des,
                            const long dh,
                            const long dw,
                            const long ch,
                            const long cw)
{
    const long sw = dw + cw - 1;
    for(int d0=0;d0<dh;++d0)
        for(int d1=0;d1<dw;++d1)
        {
            pack *d = des + d0 * dw + d1;
            asm("vmovapd (%0), %%ymm0;"::"r"(d):"%ymm0");
            for(int c0=0;c0<ch;++c0)
                for(int c1=0;c1<cw;++c1)
                {
                    asm("                                   \
                        vmovapd (%0), %%ymm1;               \
                        vbroadcastsd %1, %%ymm2;            \
                        vfmadd231pd %%ymm2, %%ymm1, %%ymm0; \
                        "::"r"(src + (d0 + c0) * sw + d1 + c1),"m"(conv[c0 * cw + c1])
                        :"%ymm0","%ymm1","%ymm2");
                }
            asm("vmovapd %%ymm0, (%0);"::"r"(d):"%ymm0");
        }
}

static void convolute_valid2(pack *src,
                             pack *conv,
                             double *des,
                             const long dh,
                             const long dw,
                             const long ch,
                             const long cw)
{
    const long sw = dw + cw - 1;
    for(int d0 = 0;d0 < dh;++d0)
        for(int d1 = 0;d1 < dw;++d1)
        {
            asm("vxorpd %ymm0, %ymm0, %ymm0;");
            for(int c0=0;c0<ch;++c0)
                for(int c1=0;c1<cw;++c1)
                {
                    asm("                                   \
                        vmovapd (%0), %%ymm1;               \
                        vfmadd231pd (%1), %%ymm1, %%ymm0;   \
                        "::"r"(src + (d0 + c0) * sw + d1 + c1),"r"(conv + c0 * cw + c1)
                        :"%ymm0","%ymm1");
                }
            asm("                               \
                vextractf128 $1, %%ymm0, %%xmm1;\
                vhaddpd %%xmm1, %%xmm0, %%xmm0; \
                vhaddpd %%xmm0, %%xmm0, %%xmm0; \
                vaddsd  (%0), %%xmm0, %%xmm0;   \
                vmovsd  %%xmm0, (%0);           \
                "::"r"(des + d0 * dw + d1)
                :"%ymm0","%ymm1");
        }
}

static void convolute_full(pack *src,
                           double *conv,
                           pack *des,
                           long sh,
                           long sw,
                           long ch,
                           long cw)
{
    const long dw = sw + cw - 1;
    for(int s0 = 0;s0 < sh;++s0)
        for(int s1 = 0;s1 < sw;++s1)
        {
            asm("vmovapd (%0), %%ymm1;"::"r"(src + s0 * sw + s1):"%ymm1");
            for(int c0=0;c0<ch;++c0)
                for(int c1=0;c1<cw;++c1)
                {
                    asm("                                   \
                        vbroadcastsd %0, %%ymm2;            \
                        vmovapd (%1), %%ymm0;               \
                        vfmadd231pd %%ymm2, %%ymm1, %%ymm0; \
                        vmovapd %%ymm0, (%1);               \
                        "::"m"(conv[c0 * cw + c1]),"r"(des + (s0 + c0) * dw + s1 + c1)
                        :"%ymm0","%ymm1","%ymm2");
                }
        }
}

static void vector_x_matrix(pack *src,double *mat,pack *des,long height,long width)
{
    for (int y = 0; y < width; ++y)
    {
        asm("vmovapd (%0), %%ymm0;"::"r"(des + y):"%ymm0");
        for (int x = 0; x < height; ++x)
        {
            asm("                                   \
                vbroadcastsd %1, %%ymm1;            \
                vfmadd231pd (%0), %%ymm1, %%ymm0;   \
                "::"r"(src + x),"m"(mat[x * width + y])
                :"%ymm0","%ymm1");
        }
        asm("vmovapd %%ymm0, (%0);"::"r"(des + y):"%ymm0");
    }
}

static void matrix_x_vector(double *mat,pack *src,pack *des,long height,long width)
{
    for (int x = 0; x < height; ++x)
    {
        asm("vmovapd (%0), %%ymm0;"::"r"(des + x):"%ymm0");
        for (int y = 0; y < width; ++y)
        {
            asm("                                   \
                vbroadcastsd %1, %%ymm1;            \
                vfmadd231pd (%0), %%ymm1, %%ymm0;   \
                "::"r"(src + y),"m"(mat[x * width + y])
                :"%ymm0","%ymm1");
        }
        asm("vmovapd %%ymm0, (%0);"::"r"(des + x):"%ymm0");
    }
}




#define CONVOLUTE_FULL(input,output,weight)                         \
{                                                                   \
    convolute_full((pack *)input,(double *)weight,(pack *)output,   \
        GETLENGTH(input),GETLENGTH(*(input)),                       \
        GETLENGTH(weight),GETLENGTH(*(weight)));                    \
}

#define CONVOLUTE_VALID1(input,output,weight)                       \
{                                                                   \
    convolute_valid1((pack *)input,(double *)weight,(pack *)output, \
        GETLENGTH(output),GETLENGTH(*(output)),                     \
        GETLENGTH(weight),GETLENGTH(*(weight)));                    \
}


#define CONVOLUTE_VALID2(input,output,weight)                       \
{                                                                   \
    convolute_valid2((pack *)input,(pack *)weight,(double *)output, \
        GETLENGTH(output),GETLENGTH(*(output)),                     \
        GETLENGTH(weight),GETLENGTH(*(weight)));                    \
}



#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID1(input[x], output[y], weight[x][y]);				\
	FOREACH(x, GETLENGTH(output))												\
		FOREACH(y, GETCOUNT(output[x]))											\
        FOREACH(i, SZPACK)                                                      \
		((pack *)output[x])[y][i] = action(((pack *)output[x])[y][i] + bias[x]);\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(x, GETCOUNT(inerror))											\
        FOREACH(i,SZPACK)                                                   \
		((pack *)inerror)[x][i] *= actiongrad(((pack *)input)[x][i]);		\
	FOREACH(x, GETLENGTH(outerror))											\
		FOREACH(y, GETCOUNT(outerror[y]))									\
            FOREACH(i, SZPACK)                                              \
                bd[x] += ((pack *)outerror[x])[y][i];						\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID2(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(j, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
        FOREACH(i, SZPACK)                                                                      \
        {                                                                                       \
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[j][o0*len0 + l0][o1*len1 + l1][i] > input[j][o0*len0 + x0][o1*len1 + x1][i];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[j][o0][o1][i] = input[j][o0*len0 + x0][o1*len1 + x1][i];								\
        }                                                                                       \
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(j, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
        FOREACH(i, SZPACK)                                                                      \
        {                                                                                       \
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[j][o0*len0 + l0][o1*len1 + l1][i] > input[j][o0*len0 + x0][o1*len1 + x1][i];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}                                                                                       \
        inerror[j][o0*len0 + x0][o1*len1 + x1][i] = outerror[j][o0][o1][i];						\
        }                                                                                       \
	}																							\
}


#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)                  \
{                                                                             \
    vector_x_matrix((pack *)input,(double *)weight,(pack *)output,            \
        GETLENGTH(weight),GETLENGTH(*(weight)));                              \
	FOREACH(j, GETLENGTH(bias))                                               \
        FOREACH(i, SZPACK)                                                    \
            ((pack *)output)[j][i] = action(((pack *)output)[j][i] + bias[j]);\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
    matrix_x_vector((double *)weight,(pack *)outerror,(pack *)inerror,          \
        GETLENGTH(weight),GETLENGTH(*(weight)));                                \
	FOREACH(j, GETCOUNT(inerror))												\
        FOREACH(i, SZPACK)                                                      \
		((pack *)inerror)[j][i] *= actiongrad(((pack *)input)[j][i]);           \
	FOREACH(j, GETLENGTH(outerror))												\
        FOREACH(i, SZPACK)                                                      \
		bd[j] += ((pack *)outerror)[j][i];                                      \
	FOREACH(x, GETLENGTH(weight))                                               \
        FOREACH(y, GETLENGTH(*weight))                                          \
            FOREACH(i, SZPACK)                                                  \
			wd[x][y] += ((pack *)input)[x][i] * ((pack *)outerror)[y][i];		\
}

static double tanhgrad(double y)
{
    return 1 - y*y;
}

static void forward(LeNet5 *lenet, FeaturePack *featurePack, double(*action)(double))
{
    CONVOLUTION_FORWARD(featurePack->layer0, featurePack->layer1, lenet->weight0_1, lenet->bias0_1, action);
    SUBSAMP_MAX_FORWARD(featurePack->layer1, featurePack->layer2);
    CONVOLUTION_FORWARD(featurePack->layer2, featurePack->layer3, lenet->weight2_3, lenet->bias2_3, action);
    SUBSAMP_MAX_FORWARD(featurePack->layer3, featurePack->layer4);
    CONVOLUTION_FORWARD(featurePack->layer4, featurePack->layer5, lenet->weight4_5, lenet->bias4_5, action);
    DOT_PRODUCT_FORWARD(featurePack->layer5, featurePack->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *delta, FeaturePack *errorPack, FeaturePack *featurePack, double(*actiongrad)(double))
{
    DOT_PRODUCT_BACKWARD(featurePack->layer5, errorPack->layer5, errorPack->output, lenet->weight5_6, delta->weight5_6, delta->bias5_6, actiongrad);
    CONVOLUTION_BACKWARD(featurePack->layer4, errorPack->layer4, errorPack->layer5, lenet->weight4_5, delta->weight4_5, delta->bias4_5, actiongrad);
    SUBSAMP_MAX_BACKWARD(featurePack->layer3, errorPack->layer3, errorPack->layer4);
    CONVOLUTION_BACKWARD(featurePack->layer2, errorPack->layer2, errorPack->layer3, lenet->weight2_3, delta->weight2_3, delta->bias2_3, actiongrad);
    SUBSAMP_MAX_BACKWARD(featurePack->layer1, errorPack->layer1, errorPack->layer2);
    CONVOLUTION_BACKWARD(featurePack->layer0, errorPack->layer0, errorPack->layer1, lenet->weight0_1, delta->weight0_1, delta->bias0_1, actiongrad);
}

static void load_input(FeaturePack *featurePack, image input[],uint8 count)
{
    count = count % 5;
    const long sz = sizeof(*input) / sizeof(***input);
    FOREACH(i, count)
    {
        double mean = 0,std = 0;
        FOREACH(j, GETLENGTH(*input))
        FOREACH(k, GETLENGTH(**input))
        {
            mean += input[i][j][k];
            std += input[i][j][k] * input[i][j][k];
        }
        mean /= sz;
        std = sqrt(std / sz - mean*mean);
        FOREACH(j, GETLENGTH(*input))
        FOREACH(k, GETLENGTH(**input))
        {
            featurePack->layer0[0][j][k][i] = (input[i][j][k] - mean) / std;
        }
    }
}

static void load_target(FeaturePack *featurePack, FeaturePack *errorPack, uint8 *label,const char(*resMat)[OUTPUT],uint8 count, double(*actiongrad)(double))
{
    count = count % 5;
    pack *output = (pack *)featurePack->output;
    pack *error = (pack *)errorPack->output;
    FOREACH(i, GETCOUNT(featurePack->output))
    {
        FOREACH(j, count)
        {
            error[i][j] = (resMat[label[j]][i] - output[i][j])*actiongrad(output[i][j]);
        }
    }
}

static char *align(char *p,const unsigned long align)
{
    const unsigned long mod = (unsigned long)p % align;
    return p + (mod > 0) * (align - mod);
}

void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT], uint8 *labels, int batchSize)
{
    double buffer[sizeof(LeNet5)/sizeof(double)] = { 0 };
    int i = 0;
#pragma omp parallel for
    for (i = 0; i < batchSize / SZPACK; i++)
    {
        char buffer1[MAXSZALIGN(FeaturePack)] = { 0 };
        char buffer2[MAXSZALIGN(FeaturePack)] = { 0 };
        FeaturePack *featurePack = (FeaturePack *)align(buffer1, sizeof(pack));
        FeaturePack *errorPack = (FeaturePack *)align(buffer2, sizeof(pack));
        LeNet5 delta = { 0 };
        load_input(featurePack, inputs + i * SZPACK, SZPACK);
        forward(lenet, featurePack, tanh);
        load_target(featurePack, errorPack, labels + i * SZPACK, resMat, SZPACK, tanhgrad);
        backward(lenet, &delta, errorPack, featurePack, tanhgrad);
        #pragma omp critical
        {
            FOREACH(j, sizeof(LeNet5)/sizeof(double))
            buffer[j] += ((double *)&delta)[j];
        }
    }
    double k = ALPHA / batchSize;
    FOREACH(i, sizeof(LeNet5)/sizeof(double))
    ((double *)lenet)[i] += k * buffer[i];
}
