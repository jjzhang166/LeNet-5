#include "lenet.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>

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