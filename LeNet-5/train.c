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

//AVX指令集用到的YMM寄存器存储空间大小
#define SZYMM   32
//YMM寄存器可存储双精度浮点数个数
#define SZPACK (SZYMM / sizeof(double))

typedef double pack[SZPACK];

static void convolute_valid1(pack *src,double *conv,pack *des,const long dh,const long dw,const long ch,const long cw);
static void convolute_valid2(pack *src,pack *conv,double *des,const long dh,const long dw,const long ch,const long cw);
static void convolute_full(pack *src,double *conv,pack *des,long sh,long sw,long ch,long cw);
static void vector_x_matrix(pack *src,double *mat,pack *des,long height,long width);
static void matrix_x_vector(double *mat,pack *src,pack *des,long height,long width);
static void subsamp_max_forward(pack *src,pack *des,const long sh,const long sw,const long dh,const long dw);
static void subsamp_max_backward(pack *srcl,pack *desl,pack *src,pack *des,const long sh,const long sw,const long dh,const long dw);


//f(n,align) = min{x|x >= n && x % align==0}
#define ALIGN(n,align) (((align)-1+(n))/(align)*(align))

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

#define FOREACH(i,count) for (long i = 0; i < count; ++i)

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
	FOREACH(x, sizeof(inerror) / sizeof(double))							\
		((double *)inerror)[x] *= actiongrad(((double *)input)[x]);         \
	FOREACH(x, GETLENGTH(outerror))											\
		FOREACH(y, sizeof(outerror[x]) / sizeof(double))					\
            bd[x] += ((double *)outerror[x])[y];                            \
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID2(input[x], wd[x][y], outerror[y]);				\
}




#define SUBSAMP_MAX_FORWARD(input,output)										\
{																				\
	FOREACH(j, GETLENGTH(output))												\
    subsamp_max_forward((pack *)input[j],(pack *)output[j],GETLENGTH(*(input)), \
        GETLENGTH(**(input)),GETLENGTH(*(output)),GETLENGTH(**(output)));       \
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror,output)                     \
{                                                                               \
	FOREACH(j, GETLENGTH(output))												\
    subsamp_max_backward((pack *)output[j],(pack *)input[j],                    \
        (pack *)outerror[j],(pack *)inerror[j],GETLENGTH(*(output)),            \
        GETLENGTH(**(output)),GETLENGTH(*(input)),GETLENGTH(**(input)));        \
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
    SUBSAMP_MAX_BACKWARD(featurePack->layer3, errorPack->layer3, errorPack->layer4,featurePack->layer4);
    CONVOLUTION_BACKWARD(featurePack->layer2, errorPack->layer2, errorPack->layer3, lenet->weight2_3, delta->weight2_3, delta->bias2_3, actiongrad);
    SUBSAMP_MAX_BACKWARD(featurePack->layer1, errorPack->layer1, errorPack->layer2,featurePack->layer2);
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



void train(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT], uint8 *labels, int batchSize)
{
    double dlenet[ALIGN(sizeof(LeNet5),sizeof(pack))] = { 0 };
    int i = 0;
    uint8 szload = SZPACK;
#pragma omp parallel for
    for (i = 0; i < (batchSize + SZPACK - 1) / SZPACK; i++)
    {
        if(szload > batchSize - i * SZPACK)
            szload = batchSize - i * SZPACK;
        char buffer[sizeof(FeaturePack) * 2 + ALIGN(sizeof(LeNet5), sizeof(pack)) + sizeof(pack) - 1] = { 0 };
        FeaturePack *featurePack = (FeaturePack *)ALIGN((unsigned long)buffer, sizeof(pack));
        FeaturePack *errorPack = featurePack + 1;
        LeNet5 *delta = (LeNet5 *)(errorPack + 1);
        load_input(featurePack, inputs + i * SZPACK, szload);
        forward(lenet, featurePack, tanh);
        load_target(featurePack, errorPack, labels + i * SZPACK, resMat, szload, tanhgrad);
        backward(lenet, delta, errorPack, featurePack, tanhgrad);
        #pragma omp critical
        {
            FOREACH(j, sizeof(LeNet5)/sizeof(double))
                dlenet[j] += ((double *)delta)[j];
        }
    }
    const double k = ALPHA / batchSize;
    FOREACH(i, sizeof(LeNet5)/sizeof(double))
    ((double *)lenet)[i] += k * dlenet[i];
}

static void convolute_valid1(pack *src,
                             double *conv,
                             pack *des,
                             const long dh,
                             const long dw,
                             const long ch,
                             const long cw)
{
    const long sw = dw + cw - 1;
    for(long d0=0;d0<dh;++d0)
        for(long d1=0;d1<dw;++d1)
        {
            pack *d = des + d0 * dw + d1;
            asm("vmovapd (%0), %%ymm0;"::"r"(d):"%ymm0");
            for(long c0=0;c0<ch;++c0)
                for(long c1=0;c1<cw;++c1)
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
    for(long d0 = 0;d0 < dh;++d0)
        for(long d1 = 0;d1 < dw;++d1)
        {
            asm("vxorpd %ymm0, %ymm0, %ymm0;");
            for(long c0=0;c0<ch;++c0)
                for(long c1=0;c1<cw;++c1)
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
    for(long s0 = 0;s0 < sh;++s0)
        for(long s1 = 0;s1 < sw;++s1)
        {
            asm("vmovapd (%0), %%ymm1;"::"r"(src + s0 * sw + s1):"%ymm1");
            for(long c0=0;c0<ch;++c0)
                for(long c1=0;c1<cw;++c1)
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
    for (long y = 0; y < width; ++y)
    {
        asm("vmovapd (%0), %%ymm0;"::"r"(des + y):"%ymm0");
        for (long x = 0; x < height; ++x)
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
    for (long x = 0; x < height; ++x)
    {
        asm("vmovapd (%0), %%ymm0;"::"r"(des + x):"%ymm0");
        for (long y = 0; y < width; ++y)
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

static void subsamp_max_forward(pack *src,pack *des,
                                const long sh,const long sw,
                                const long dh,const long dw)
{
    const long lh = sh / dh,lw = sw / dw;
    for(long d0 = 0;d0 < dh;++d0)
        for(long d1 = 0;d1 < dw;++d1)
        {
            asm("vmovapd (%0), %%ymm0;"::"r"(src + d0 * lh * sw + d1 * lw):"%ymm0");
            for(long l = 1;l < lh * lw;++l)
                asm("vmaxpd (%0), %%ymm0, %%ymm0"::"r"(src + (d0 * lh + l / lw) * sw + d1 * lw + l % lw):"%ymm0");
            asm("vmovapd %%ymm0, (%0);"::"r"(des + d0 * dw + d1):"%ymm0");
        }
}

static void subsamp_max_backward(pack *srcl,pack *desl,
                                 pack *src,pack *des,
                                 const long sh,const long sw,
                                 const long dh,const long dw)
{
    const long lh = dh / sh,lw = dw / sw;
    for(long s0 = 0;s0 < sh;++s0)
        for(long s1 = 0;s1 < sw;++s1)
        {
            long index = s0 * sw + s1;
            asm("                       \
                vmovapd (%0), %%ymm0;   \
                vmovapd (%1), %%ymm1;   \
                "::"r"(src + index),"r"(srcl + index)
                :"%ymm0","%ymm1");
            for(long l = 0;l < lh * lw;++l)
            {
                index = (s0 * lh + l / lw) * dw + s1 * lw + l % lw;
                asm("                               \
                    vmovapd (%1), %%ymm2;           \
                    vcmpeqpd %%ymm2, %%ymm1, %%ymm2;\
                    vmaskmovpd %%ymm0, %%ymm2, (%0);\
                    vorpd %%ymm2, %%ymm1, %%ymm1;   \
                    "::"r"(des + index),"r"(desl + index)
                    :"%ymm0","%ymm1","%ymm2");
            }
        }
}
