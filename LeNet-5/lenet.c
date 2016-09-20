#include "lenet.h"
#include <memory.h>
#include <float.h>
#include <time.h>
#include <stdlib.h>
#include <stdalign.h>
//#include <omp.h>
#include <math.h>
#include <stdio.h>
static void vector_x_matrix(double *src,double *mat,double *des,long height,long width);
static void matrix_x_vector(double *mat,double *src,double *des,long height,long width);
static void scalar_fma_vector(double k,double *src,double *des,long length);
static void convolute_valid(double *src,double *conv,double *des,const long dh,const long dw,const long ch,const long cw);
static void convolute_full(double *src,double *conv,double *des,const long sh,const long sw,const long ch,const long cw);

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

//#define CONVOLUTE_VALID(input,output,weight)											\
//{																						\
//	FOREACH(o0,GETLENGTH(output))														\
//		FOREACH(o1,GETLENGTH(*(output)))												\
//			FOREACH(w0,GETLENGTH(weight))												\
//				FOREACH(w1,GETLENGTH(*(weight)))										\
//					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
//}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

//#define CONVOLUTE_VALID(input,output,weight)                    \
//{                                                               \
//const double mask[4] = {                                    \
//(GETLENGTH(*(output))&3)>0,                             \
//(GETLENGTH(*(output))&3)>1,                             \
//(GETLENGTH(*(output))&3)>2,                             \
//};                                                          \
//asm("vmovdqu (%0), %%ymm3;"::"r"(mask));                    \
//for(int o0=0;o0<GETLENGTH(output);++o0)                     \
//{                                                           \
//for(int o1=0;o1<GETLENGTH(*(output));o1+=4)             \
//{                                                       \
//asm("vxorpd %ymm0, %ymm0, %ymm0;");                 \
//for(int w0=0;w0<GETLENGTH(weight);++w0)             \
//{                                                   \
//for(int w1=0;w1<GETLENGTH(*weight);++w1)        \
//{                                               \
//asm("                                       \
//vmovdqu         (%0),   %%ymm1;         \
//vbroadcastsd    %1,     %%ymm2;         \
//vfmadd231pd     %%ymm1, %%ymm2, %%ymm0; \
//"::"r"((input)[o0 + w0] + o1 + w1),     \
//"m"(weight[w0][w1]):"%ymm0","%ymm3");   \
//}                                               \
//}                                                   \
//if(o1==GETLENGTH(*(output))>>2<<2)                  \
//asm("vmulpd %ymm3, %ymm0, %ymm0;");             \
//asm("                                               \
//vaddpd  (%0),   %%ymm0, %%ymm0;                 \
//vmovdqu %%ymm0, (%0);                           \
//"::"r"(output[o0]+o1):"%ymm0");                 \
//}                                                       \
//}                                                           \
//}


#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
    for (int x = 0; x < GETLENGTH(weight); ++x)									\
        for (int y = 0; y < GETLENGTH(*weight); ++y)							\
            convolute_valid((double *)input[x], (double *)weight[x][y],         \
                (double *)output[y], GETLENGTH(output[y]),GETLENGTH(*output[y]),\
                GETLENGTH(weight[x][y]),GETLENGTH(*weight[x][y]));              \
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
            convolute_valid((double *)input[x], (double *)outerror[y],      \
            (double *)wd[x][y], GETLENGTH(wd[x][y]),GETLENGTH(*wd[x][y]),   \
            GETLENGTH(outerror[y]),GETLENGTH(*outerror[y]));                \
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}


#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)                                                \
{                                                                                                           \
    vector_x_matrix((double *)input,(double *)weight,(double *)output,GETLENGTH(weight),GETLENGTH(*weight));\
    FOREACH(j, GETLENGTH(bias))                                                                             \
        ((double *)output)[j] = action(((double *)output)[j] + bias[j]);                                    \
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)                                    \
{                                                                                                               \
    matrix_x_vector((double *)weight,(double *)outerror,(double *)inerror,GETLENGTH(weight),GETLENGTH(*weight));\
    FOREACH(i, GETCOUNT(inerror))                                                                               \
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                                             \
    scalar_fma_vector(1,(double *)outerror,bd,GETLENGTH(outerror));                                             \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                                                 \
        scalar_fma_vector(((double *)input)[x],((double *)outerror),wd[x],GETLENGTH(*weight));                  \
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
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static void load_input(Feature *features, image input)
{
	normalize((uint8 *)input, (double *)features->input, sizeof(image) / sizeof(uint8));
}

static void load_target(Feature *features, Feature *errors, const char *label, double(*actiongrad)(double))
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	const int outlen = GETCOUNT(features->output);
	FOREACH(i, outlen)
		error[i] = (label[i] - output[i])*actiongrad(output[i]);
}

static uint8 get_result(Feature *features, const char(*labels)[OUTPUT], uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double minvalue = DBL_MAX;
	for (uint8 i = 0; i < count; ++i)
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

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT], uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, tanh);
		load_target(&features, &errors, resMat[labels[i]], tanhgrad);
		backward(lenet, &deltas, &errors, &features, tanhgrad);
		#pragma omp critical
		{
            scalar_fma_vector(1, (double *)&deltas, buffer, GETCOUNT(LeNet5));
		}
	}
    scalar_fma_vector(ALPHA / batchSize, buffer, (double *)lenet, GETCOUNT(LeNet5));
}

void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, tanh);
	load_target(&features, &errors, resMat[label], tanhgrad);
	backward(lenet, &deltas, &errors, &features, tanhgrad);
    scalar_fma_vector(ALPHA , (double *)&deltas, (double *)lenet, GETCOUNT(LeNet5));
}

uint8 Predict(LeNet5 *lenet, image input, const char (*resMat)[OUTPUT],uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, tanh);
	return get_result(&features, resMat, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL0 * LENGTH_KERNEL0 * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL1 * LENGTH_KERNEL1 * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}


#if (__GNUC__ && (__i386__ || __x86_64__))

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

static void matrix_x_vector(double *mat,double *src,double *des,const long height,const long width)
{
    const long lastw = width >> 2 << 2;
    asm("                           \
        vmovupd (%0), %%ymm4;       \
        vmovupd (%0,%1,8), %%ymm5;  \
        "::"r"(mask),"r"(4 - (width & 3)):"%ymm4","%ymm5");
    for(long h=0;h<height;h++)
    {
        asm("vxorpd %ymm0, %ymm0, %ymm0;");
        asm("vmovapd %ymm4, %ymm3;");
        for(long w=0;w<width;w+=4)
        {
            if(w==lastw) asm("vmovapd %ymm5, %ymm3;");
            asm("                                       \
                vmaskmovpd (%0,%1,8), %%ymm3, %%ymm1;   \
                vmaskmovpd (%2,%3,8), %%ymm3, %%ymm2;   \
                vfmadd231pd %%ymm2, %%ymm1, %%ymm0;     \
                "::"r"(src),"r"(w),"r"(mat),"r"(h * width + w)
                :"%ymm0","%ymm1","%ymm2","%ymm3","%ymm4","%ymm5");
        }
        asm("                                       \
            vextractf128 $1, %%ymm0, %%xmm1;        \
            vhaddpd %%xmm1, %%xmm0, %%xmm0;         \
            vhaddpd %%xmm0, %%xmm0, %%xmm0;         \
            vmovsd (%0,%1,8), %%xmm1;               \
            vaddpd %%xmm1, %%xmm0, %%xmm0;          \
            vmovsd %%xmm0, (%0,%1,8);               \
            "::"r"(des),"r"(h)
            :"%ymm0","%ymm1","%ymm2","%ymm4","%ymm5");
    }
}

static void scalar_fma_vector(double k,double *src,double *des,long length)
{
    long i = length & 3;
    asm("vbroadcastsd %0, %%ymm3;"::"m"(k));
    if(i)
    {
        asm("                                   \
            vmovupd (%0,%1,8), %%ymm4;          \
            vmaskmovpd (%2), %%ymm4, %%ymm1;    \
            vmaskmovpd (%3), %%ymm4, %%ymm2;    \
            vfmadd231pd %%ymm3, %%ymm2, %%ymm1; \
            vmaskmovpd %%ymm1, %%ymm4, (%2);    \
            "::"r"(mask),"r"(4 - i),"r"(des),"r"(src)
            :"%ymm3"
            );
    }
    for(;i<length;i+=4)
    {
        asm("                                       \
            vmovupd (%0,%2,8), %%ymm1;              \
            vfmadd231pd (%1,%2,8), %%ymm3, %%ymm1;  \
            vmovupd %%ymm1, (%0,%2,8);              \
            "::"r"(des),"r"(src),"r"(i)
            :"%ymm3"
            );
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


static void convolute_full(double *src,double *conv,double *des,const long sh,const long sw,const long ch,const long cw)
{
    const long lastw = sw >> 2 << 2,dw = sw + cw - 1;
    asm("vmovupd (%0), %%ymm3;"::"r"(mask):"%ymm3");
    for(long s1=0;s1<sw;s1+=4)
    {
        if(s1==lastw) asm("vmovupd (%0,%1,8), %%ymm3;"::"r"(mask),"r"(4 - (sw & 3)):"%ymm3");
        for(long s0=0;s0<sh;++s0)
        {
            asm("vmaskmovpd (%0,%1,8), %%ymm3, %%ymm2;"::"r"(src),"r"(s0*sw+s1):"%ymm2","%ymm3");
            for(long c0=0;c0<ch;++c0)
            {
                for(long c1=0;c1<cw;++c1)
                {
                    asm("                                       \
                        vbroadcastsd %2, %%ymm1;                \
                        vmaskmovpd (%0,%1,8), %%ymm3, %%ymm0;   \
                        vfmadd231pd %%ymm1, %%ymm2, %%ymm0;     \
                        vmaskmovpd %%ymm0, %%ymm3, (%0,%1,8);   \
                        "::"r"(des),"r"((s0+c0)*dw+s1+c1),"m"(conv[c0*cw+c1])
                        :"%ymm0","%ymm1","%ymm2","%ymm3");
                }
            }
        }
    }
}

#else

static void vector_x_matrix(double *src,double *mat,double *des,long height,long width)
{
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y)
            des[y] += src[x] * mat[x * width + y];
}

static void matrix_x_vector(double *mat,double *src,double *des,long height,long width)
{
    for (int x = 0; x < height; ++x)
        for (int y = 0; y < width; ++y)
            des[x] += src[y] * mat[x * width + y];
}

static void scalar_fma_vector(double k,double *src,double *des,long length)
{
    for(int x = 0; x < length; ++x)
        des[x] += k * src[x];
}

static void convolute_valid(double *src,double *conv,double *des,long dh,long dw,long ch,long cw)
{
    const long sw = dw + cw - 1;
    for(int d0=0;d0<dh;++d0)
        for(int d1=0;d1<dw;++d1)
            for(int c0=0;c0<ch;++c0)
                for(int c1=0;c1<cw;++c1)
                    des[d0*dw+d1]+=src[(d0+c0)*sw+d1+c1]*conv[c0*cw+c1];
}

static void convolute_full(double *src,double *conv,double *des,long sh,long sw,long ch,long cw)
{
    const long dw = sw + cw - 1;
    for(int s0=0;s0<sh;++s0)
        for(int s1=0;s1<sw;++s1)
            for(int c0=0;c0<ch;++c0)
                for(int c1=0;c1<cw;++c1)
                    des[(s0+c0)*dw+s1+c1]+=src[s0*sw+s1]*conv[c0*cw+c1];
}
#endif
