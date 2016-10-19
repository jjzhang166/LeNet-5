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

#ifndef __AVX__
#error Please Add Build Variant -mavx
#endif //__AVX__

//AVX指令集用到的YMM寄存器存储空间大小
#define SZYMM   32
//YMM寄存器可存储双精度浮点数个数
#define SZPACK (SZYMM / sizeof(double))

typedef double pack_t[SZPACK];

static void convolute_valid1(pack_t *src,double *conv,pack_t *des,const long dh,const long dw,const long ch,const long cw);
static void convolute_valid2(pack_t *src,pack_t *conv,double *des,const long dh,const long dw,const long ch,const long cw);
static void convolute_full(pack_t *src,double *conv,pack_t *des,const long sh,const long sw,const long ch,const long cw);
static void vector_x_matrix(pack_t *src,double *mat,pack_t *des,const long height,const long width);
static void matrix_x_vector(double *mat,pack_t *src,pack_t *des,const long height,const long width);
static void subsamp_max_forward(pack_t *src,pack_t *des,const long sh,const long sw,const long dh,const long dw);
static void subsamp_max_backward(pack_t *srcl,pack_t *desl,pack_t *src,pack_t *des,const long sh,const long sw,const long dh,const long dw);
static void get_result(pack_t output[OUTPUT], const char(*resMat)[OUTPUT], const uint8_t labelCount, uint8_t labels[SZPACK], uint8_t szpack);


//f(n,align) = min{x|x >= n && x % align == 0}
#define ALIGN(n,align) (((align)-1+(n))/(align)*(align))

typedef struct FeaturePack
{
    pack_t layer0[LAYER0][LENGTH_FEATURE0][LENGTH_FEATURE0];
    pack_t layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    pack_t layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    pack_t layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    pack_t layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    pack_t layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    pack_t output[OUTPUT];
}FeaturePack;

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(pack_t))

#define FOREACH(i,count) for (long i = 0; i < count; ++i)

#define CONVOLUTE_FULL(input,output,weight)								\
{																		\
    convolute_full((pack_t *)input,(double *)weight,(pack_t *)output,   \
        GETLENGTH(input),GETLENGTH(*(input)),							\
        GETLENGTH(weight),GETLENGTH(*(weight)));						\
}

#define CONVOLUTE_VALID1(input,output,weight)							\
{																		\
    convolute_valid1((pack_t *)input,(double *)weight,(pack_t *)output, \
        GETLENGTH(output),GETLENGTH(*(output)),							\
        GETLENGTH(weight),GETLENGTH(*(weight)));						\
}


#define CONVOLUTE_VALID2(input,output,weight)							\
{																		\
    convolute_valid2((pack_t *)input,(pack_t *)weight,(double *)output, \
        GETLENGTH(output),GETLENGTH(*(output)),							\
        GETLENGTH(weight),GETLENGTH(*(weight)));						\
}



#define CONVOLUTION_FORWARD(input,output,weight,bias,action)						\
{																					\
	for (int x = 0; x < GETLENGTH(weight); ++x)										\
		for (int y = 0; y < GETLENGTH(*weight); ++y)								\
			CONVOLUTE_VALID1(input[x], output[y], weight[x][y]);					\
	FOREACH(x, GETLENGTH(output))													\
		FOREACH(y, GETCOUNT(output[x]))												\
        FOREACH(i, SZPACK)															\
		((pack_t *)output[x])[y][i] = action(((pack_t *)output[x])[y][i] + bias[x]);\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);          \
	FOREACH(x, sizeof(inerror) / sizeof(double))							\
		((double *)inerror)[x] *= actiongrad(((double *)input)[x]);         \
	FOREACH(x, GETLENGTH(outerror))											\
		FOREACH(y, sizeof(outerror[x]) / sizeof(double))					\
            bd[x] += ((double *)outerror[x])[y];                            \
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID2(input[x], wd[x][y], outerror[y]);				\
}




#define SUBSAMP_MAX_FORWARD(input,output)											\
{																					\
	FOREACH(j, GETLENGTH(output))													\
    subsamp_max_forward((pack_t *)input[j],(pack_t *)output[j],GETLENGTH(*(input)), \
        GETLENGTH(**(input)),GETLENGTH(*(output)),GETLENGTH(**(output)));			\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror,output)				\
{																		\
	FOREACH(j, GETLENGTH(output))										\
    subsamp_max_backward((pack_t *)output[j],(pack_t *)input[j],		\
        (pack_t *)outerror[j],(pack_t *)inerror[j],GETLENGTH(*(output)),\
        GETLENGTH(**(output)),GETLENGTH(*(input)),GETLENGTH(**(input)));\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)						\
{																					\
    vector_x_matrix((pack_t *)input,(double *)weight,(pack_t *)output,				\
        GETLENGTH(weight),GETLENGTH(*(weight)));									\
	FOREACH(j, GETLENGTH(bias))														\
        FOREACH(i, SZPACK)															\
            ((pack_t *)output)[j][i] = action(((pack_t *)output)[j][i] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
    matrix_x_vector((double *)weight,(pack_t *)outerror,(pack_t *)inerror,      \
        GETLENGTH(weight),GETLENGTH(*(weight)));                                \
	FOREACH(j, GETCOUNT(inerror))												\
        FOREACH(i, SZPACK)                                                      \
		((pack_t *)inerror)[j][i] *= actiongrad(((pack_t *)input)[j][i]);       \
	FOREACH(j, GETLENGTH(outerror))												\
        FOREACH(i, SZPACK)                                                      \
		bd[j] += ((pack_t *)outerror)[j][i];                                    \
	FOREACH(x, GETLENGTH(weight))                                               \
        FOREACH(y, GETLENGTH(*weight))                                          \
            FOREACH(i, SZPACK)                                                  \
			wd[x][y] += ((pack_t *)input)[x][i] * ((pack_t *)outerror)[y][i];	\
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

static void load_input(pack_t(*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0], image_t input[],uint8_t count)
{
    count %= SZPACK + 1;
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
            layer0[0][j][k][i] = (input[i][j][k] - mean) / std;
        }
    }
}

static void load_target(pack_t *output, pack_t *error, uint8_t *labels,const char(*resMat)[OUTPUT],uint8_t count, double(*actiongrad)(double))
{
    count %= SZPACK + 1;
    FOREACH(i, GETLENGTH(*resMat))
    {
        FOREACH(j, count)
        {
            error[i][j] = (resMat[labels[j]][i] - output[i][j])*actiongrad(output[i][j]);
        }
    }
}

void train_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT],uint8_t *labels, const int batchSize)
{
	uint8_t szload = SZPACK;
    double deltasum[sizeof(LeNet5) / sizeof(double)] = {0};
    const double k = ALPHA / batchSize;
#pragma omp parallel for
    for (int i = 0; i < (batchSize + SZPACK - 1) / SZPACK; i++)
    {
		szload -= (i == batchSize / SZPACK) * (SZPACK - batchSize % SZPACK);
        char buffer[sizeof(FeaturePack) * 2 + ALIGN(sizeof(LeNet5), sizeof(pack_t)) + sizeof(pack_t) - 1] = { 0 };
        FeaturePack *featurePack = (FeaturePack *)ALIGN((unsigned long long)buffer, sizeof(pack_t));
        FeaturePack *errorPack = featurePack + 1;
        LeNet5 *delta = (LeNet5 *)(errorPack + 1);
        load_input(featurePack->layer0, inputs + i * SZPACK, szload);
        forward(lenet, featurePack, tanh);
        load_target(featurePack->output, errorPack->output, labels + i * SZPACK, resMat, szload, tanhgrad);
        backward(lenet, delta, errorPack, featurePack, tanhgrad);
        #pragma omp critical
        {
            FOREACH(j, sizeof(LeNet5)/sizeof(double))
                deltasum[j] += ((double *)delta)[j];
        }
    }
    FOREACH(j, sizeof(LeNet5)/sizeof(double))
        ((double *)lenet)[j] += k * deltasum[j];
}

void predict_batch(LeNet5 *lenet, image_t *inputs, const char(*resMat)[OUTPUT],uint8_t labelCount, const int batchSize, uint8_t *results)
{
	uint8_t szload = SZPACK;
#pragma omp parallel for
	for (int i = 0; i < (batchSize + SZPACK - 1) / SZPACK; i++)
	{
		szload -= (i == batchSize / SZPACK) * (SZPACK - batchSize % SZPACK);
		char buffer[sizeof(FeaturePack) + sizeof(pack_t) - 1] = { 0 };
		FeaturePack *featurePack = (FeaturePack *)ALIGN((unsigned long long)buffer, sizeof(pack_t));
		load_input(featurePack->layer0, inputs + i * SZPACK, szload);
		forward(lenet, featurePack, tanh);
		get_result(featurePack->output, resMat, labelCount, results + i*SZPACK, szload);
	}
}

static void get_result(pack_t output[OUTPUT], const char(*resMat)[OUTPUT], const uint8_t labelCount, uint8_t labels[SZPACK], uint8_t szpack)
{
	szpack %= SZPACK + 1;
	const static long long maxvalue = 0x7FFFFFFFFFFFFFFFL;
	unsigned long long result[4] = { 0 };
	asm("vbroadcastsd %0, %%ymm0;"::"m"(maxvalue) : "%ymm0");
	for (long long j = 0; j < labelCount; ++j)
	{
		asm("vxorpd %ymm1, %ymm1, %ymm1;");
		for (long i = 0; i < OUTPUT; ++i)
		{
			long temp = resMat[j][i];
			asm("                                       \
                vcvtsi2sd %0, %%xmm2, %%xmm2;           \
                vmovlhps %%xmm2, %%xmm2, %%xmm2;        \
                vinsertf128 $1, %%xmm2, %%ymm2, %%ymm2; \
                vsubpd (%1), %%ymm2, %%ymm2;            \
                vfmadd231pd %%ymm2, %%ymm2, %%ymm1;     \
                "::"m"(temp), "r"(output + i)
				: "%ymm0", "%ymm1", "%ymm2");
		}
		asm("									\
            vcmpltpd %%ymm0, %%ymm1, %%ymm2;	\
            vminpd %%ymm1, %%ymm0, %%ymm0;		\
            vbroadcastsd %0, %%ymm1;			\
            vmaskmovpd %%ymm1, %%ymm2, (%1);	\
            "::"m"(j), "r"(result)
			: "%ymm0", "%ymm1", "%ymm2", "memory");
	}
	for (long i = 0; i < szpack; ++i)
		labels[i] = (uint8_t)result[i];
}

static void convolute_valid1(pack_t *src, double *conv, pack_t *des, const long dh, const long dw, const long ch, const long cw)
{
    const long sw = dw + cw - 1;
    for(long d0=0;d0<dh;++d0)
        for(long d1=0;d1<dw;++d1)
        {
            pack_t *d = des + d0 * dw + d1;
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

static void convolute_valid2(pack_t *src, pack_t *conv, double *des, const long dh,const long dw, const long ch, const long cw)
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
                vaddpd %%xmm1, %%xmm0, %%xmm0;  \
                vhaddpd %%xmm0, %%xmm0, %%xmm0; \
                vaddsd  (%0), %%xmm0, %%xmm0;   \
                vmovsd  %%xmm0, (%0);           \
                "::"r"(des + d0 * dw + d1)
                :"%ymm0","%ymm1");
        }
}

static void convolute_full(pack_t *src, double *conv, pack_t *des, const long sh, const long sw, const long ch, const long cw)
{
	const long dw = sw + cw - 1;
	for (long s0 = 0; s0 < sh; ++s0)
		for (long s1 = 0; s1 < sw; ++s1)
		{
			asm("vmovapd (%0), %%ymm0;"::"r"(src + s0 * sw + s1) : "%ymm0");
			for (long c0 = 0; c0 < ch; ++c0)
				for (long c1 = 0; c1 < cw; ++c1)
				{
					asm("                                   \
                        vbroadcastsd %0, %%ymm1;            \
                        vfmadd213pd (%1), %%ymm0, %%ymm1;   \
                        vmovapd %%ymm1, (%1);               \
                        "::"m"(conv[c0 * cw + c1]), "r"(des + (s0 + c0) * dw + s1 + c1)
						: "%ymm0", "%ymm1");
				}
		}
}



static void vector_x_matrix(pack_t *src, double *mat, pack_t *des, const long height, const long width)
{
	for (long y = 0; y < width; ++y)
	{
		asm("vmovapd (%0), %%ymm0;"::"r"(des + y) : "%ymm0");
		for (long x = 0; x < height; ++x)
		{
			asm("                                   \
                vbroadcastsd %1, %%ymm1;            \
                vfmadd231pd (%0), %%ymm1, %%ymm0;   \
                "::"r"(src + x), "m"(mat[x * width + y])
				: "%ymm0", "%ymm1");
		}
		asm("vmovapd %%ymm0, (%0);"::"r"(des + y) : "%ymm0");
	}
}

static void matrix_x_vector(double *mat, pack_t *src, pack_t *des, const long height, const long width)
{
	for (long x = 0; x < height; ++x)
	{
		asm("vmovapd (%0), %%ymm0;"::"r"(des + x) : "%ymm0");
		for (long y = 0; y < width; ++y)
		{
			asm("                                   \
                vbroadcastsd %1, %%ymm1;            \
                vfmadd231pd (%0), %%ymm1, %%ymm0;   \
                "::"r"(src + y), "m"(mat[x * width + y])
				: "%ymm0", "%ymm1");
		}
		asm("vmovapd %%ymm0, (%0);"::"r"(des + x) : "%ymm0");
	}
}

static void subsamp_max_forward(pack_t *src, pack_t *des, const long sh, const long sw, const long dh, const long dw)
{
	const long lh = sh / dh, lw = sw / dw;
	for (long d0 = 0; d0 < dh; ++d0)
		for (long d1 = 0; d1 < dw; ++d1)
		{
			asm("vmovapd (%0), %%ymm0;"::"r"(src + d0 * lh * sw + d1 * lw) : "%ymm0");
			for (long l = 1; l < lh * lw; ++l)
				asm("vmaxpd (%0), %%ymm0, %%ymm0"::"r"(src + (d0 * lh + l / lw) * sw + d1 * lw + l % lw) : "%ymm0");
			asm("vmovapd %%ymm0, (%0);"::"r"(des + d0 * dw + d1) : "%ymm0");
		}
}

static void subsamp_max_backward(pack_t *srcl, pack_t *desl, pack_t *src, pack_t *des, const long sh, const long sw, const long dh, const long dw)
{
	const long lh = dh / sh, lw = dw / sw;
	for (long s0 = 0; s0 < sh; ++s0)
		for (long s1 = 0; s1 < sw; ++s1)
		{
			long index = s0 * sw + s1;
			asm("                       \
                vmovapd (%0), %%ymm0;   \
                vmovapd (%1), %%ymm1;   \
                "::"r"(src + index), "r"(srcl + index)
				: "%ymm0", "%ymm1");
			for (long l = 0; l < lh * lw; ++l)
			{
				index = (s0 * lh + l / lw) * dw + s1 * lw + l % lw;
				asm("                               \
                    vmovapd (%1), %%ymm2;           \
                    vcmpeqpd %%ymm2, %%ymm1, %%ymm2;\
                    vmaskmovpd %%ymm0, %%ymm2, (%0);\
                    vorpd %%ymm2, %%ymm1, %%ymm1;   \
                    "::"r"(des + index), "r"(desl + index)
					: "%ymm0", "%ymm1", "%ymm2");
			}
		}
}