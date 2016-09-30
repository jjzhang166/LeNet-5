#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

const static char resMat[10][OUTPUT] =
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


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image_t *train_data, uint8_t *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		train_batch(lenet, train_data + i, resMat, train_label + i, batch_size);
		//if (i * 100 / total_size > percent)
		//	printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image_t *test_data, uint8_t *test_label,int total_size)
{
	int right = 0;
	uint8_t *result = (uint8_t *)calloc(COUNT_TEST, sizeof(uint8_t));
	predict_batch(lenet, test_data, resMat, 10, total_size, result);
	for (int i = 0; i < total_size; ++i)
	{
		//uint8_t l = test_label[i];
		//int p = predict(lenet, test_data[i], resMat, 10);
		//right += l == p;
		right += test_label[i] == result[i];
	}
	free(result);
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, const char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}


void foo()
{
	image_t *train_data = (image_t *)calloc(COUNT_TRAIN, sizeof(image_t));
	uint8_t *train_label = (uint8_t *)calloc(COUNT_TRAIN, sizeof(uint8_t));
	image_t *test_data = (image_t *)calloc(COUNT_TEST, sizeof(image_t));
	uint8_t *test_label = (uint8_t *)calloc(COUNT_TEST, sizeof(uint8_t));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		initial(lenet);
	
	time_t t0 = time(0),t1,t2;
	printf("Start Train...\n");
	int batches[] = { 300 };
	for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
	t1 = time(0);
    printf("Train Count:%d\nTrain Time:%us\n", COUNT_TRAIN,(unsigned)(t1 - t0));
	printf("Start Test...\n");
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	t2 = time(0);
	printf("Test Accuracy:%d/%d\nTest Time:%us\n",right, COUNT_TEST, (unsigned)(t2 - t1));
	printf("Total Time:%us\n", (unsigned)(t2 - t0));
	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
#if (WIN32 || WIN64)
	system("pause");
#endif
}

int main()
{
	foo();
	return 0;
}
