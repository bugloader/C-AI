#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_NUM 25
#define HIDDEN_LAYER_NUM 2
#define HIDDEN_LAYER_SIZE 16
#define OUTPUT_NUM 10
#define LEARNING_RATE 0.1

double preferredOutput[OUTPUT_NUM];

struct Sample
{
    double input[INPUT_NUM],output[OUTPUT_NUM];
};

struct InputCell
{
    double value;
} inputLayer[INPUT_NUM];

struct FirstHiddenCell
{
    double bias;
    double weight[INPUT_NUM];
    double value;
} firstHiddenLayer[HIDDEN_LAYER_SIZE];

struct HiddenCell
{
    double bias;
    double weight[HIDDEN_LAYER_SIZE];
    double value;
} hiddenLayers[HIDDEN_LAYER_NUM-1][HIDDEN_LAYER_SIZE];

struct OutputCell
{
    double bias;
    double weight[HIDDEN_LAYER_SIZE];
    double value;
} outputLayer[OUTPUT_NUM];

struct BPData
{
    double dCHiddenBIAS[HIDDEN_LAYER_NUM][HIDDEN_LAYER_SIZE];
    double dCOutputBIAS[OUTPUT_NUM];
    double dCFirstHiddenWeight[HIDDEN_LAYER_SIZE][INPUT_NUM];
    double dCHiddenRestWeight[HIDDEN_LAYER_NUM-1][HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
    double dCOutputWeight[OUTPUT_NUM][HIDDEN_LAYER_SIZE];
};

double logistic(double rawValue)
{
    return 1/(1+pow(2.89,-rawValue));
}

void initialize()
{
    for(int i=0; i<HIDDEN_LAYER_SIZE; i++) //1st hidden layer
    {
        firstHiddenLayer[i].bias = logistic(rand()%4*(rand()%2?-1:1));
        for(int j=0; j<INPUT_NUM; j++)
        {
            firstHiddenLayer[i].weight[j]=logistic(rand()%4*(rand()%2?-1:1));
        }
    }
    for(int i=0; i<HIDDEN_LAYER_NUM-1; i++)//layers
    {
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            hiddenLayers[i][j].bias=logistic(rand()%4*(rand()%2?-1:1));
            for(int k=0; k<HIDDEN_LAYER_SIZE; k++)
            {
                hiddenLayers[i][j].weight[k]=logistic(rand()%4*(rand()%2?-1:1));
            }
        }
    }
    for(int i=0; i<OUTPUT_NUM; i++)//output
    {
        outputLayer[i].bias = logistic(rand()%4*(rand()%2?-1:1));
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            outputLayer[i].weight[j] = logistic(rand()%4*(rand()%2?-1:1));
        }
    }
}

void feedForward(struct Sample pic)
{
    for(int i=0; i<INPUT_NUM; i++)//input
    {
        inputLayer[i].value = pic.input[i];
    }
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        preferredOutput[i] = pic.output[i];
    }
    for(int i=0; i<HIDDEN_LAYER_SIZE; i++)//calculate 1st hiddenlayer's value
    {
        firstHiddenLayer[i].value = firstHiddenLayer[i].bias;
        for(int j=0; j<INPUT_NUM; j++)//transverse input
        {
            firstHiddenLayer[i].value += inputLayer[j].value * firstHiddenLayer[i].weight[j];
        }
        firstHiddenLayer[i].value = logistic(firstHiddenLayer[i].value);
    }
    for(int i=0; i<HIDDEN_LAYER_SIZE; i++)//calculate 1st hiddenlayer's value
    {
        hiddenLayers[0][i].value = hiddenLayers[0][i].bias;
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)//transverse input
        {
            hiddenLayers[0][i].value += firstHiddenLayer[j].value * hiddenLayers[0][i].weight[j];
        }
        hiddenLayers[0][i].value = logistic(hiddenLayers[0][i].value);
    }
    if(HIDDEN_LAYER_NUM>2)
    {
        for(int i=1; i<HIDDEN_LAYER_NUM-1; i++)//layers
        {
            for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
            {
                hiddenLayers[i][j].value=hiddenLayers[i][j].bias;
                for(int k=0; k<HIDDEN_LAYER_SIZE; k++)
                {
                    hiddenLayers[i][j].value += hiddenLayers[i-1][k].value * hiddenLayers[i][j].weight[k];
                }
                hiddenLayers[i][j].value = logistic(hiddenLayers[i][j].value);
            }
        }
    }
    for(int i=0; i<OUTPUT_NUM; i++)//calculate outputlayer's value
    {
        outputLayer[i].value = outputLayer[i].bias;
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)//transverse 2nd hiddenlayer
        {
            outputLayer[i].value += hiddenLayers[HIDDEN_LAYER_NUM-2][j].value * outputLayer[i].weight[j];
        }
        outputLayer[i].value = logistic(outputLayer[i].value);
    }
}

double cost()
{
    double cost=0,difference;
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        difference = (outputLayer[i].value-preferredOutput[i]);
        cost+= difference*difference;
    }
    return cost/2;
}

void deepBP(struct BPData data,int layer,int index,double tempDCBH)
{
    double nextDCBH;
    for(int i=0; i<HIDDEN_LAYER_SIZE; i++)
    {
        data.dCHiddenRestWeight[layer][index][i] += tempDCBH*hiddenLayers[layer-2][i].value;
        nextDCBH = tempDCBH*hiddenLayers[layer][index].weight[i] * (1-hiddenLayers[layer-2][i].value)*hiddenLayers[layer-2][i].value;
        data.dCHiddenBIAS[layer-1][i] += nextDCBH;
        if(layer>0)
        {
            deepBP(data,layer-1,i,nextDCBH);
        }
        else
        {
            for(int j=0; j<INPUT_NUM; j++)
            {
                data.dCFirstHiddenWeight[i][j] += nextDCBH * inputLayer[j].value;
            }
        }
    }
}

struct BPData bp()
{
    struct BPData data;
    double tempDCBHEnd;
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        data.dCOutputBIAS[i]=(outputLayer[i].value-preferredOutput[i])*(1-outputLayer[i].value) * outputLayer[i].value;
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            data.dCOutputWeight[i][j] = data.dCOutputBIAS[i]*hiddenLayers[HIDDEN_LAYER_NUM-2][j].value;
            tempDCBHEnd = data.dCOutputBIAS[i]*outputLayer[i].weight[j] * (1-hiddenLayers[HIDDEN_LAYER_NUM-2][j].value) * hiddenLayers[HIDDEN_LAYER_NUM-2][j].value;
            data.dCHiddenBIAS[HIDDEN_LAYER_NUM-2][j] += tempDCBHEnd;
            deepBP(data,HIDDEN_LAYER_NUM-2,j,tempDCBHEnd);
        }
    }
    return data;
}

void learn(int dataNum,struct Sample *pics)
{
    struct BPData total,temp;
    memset(&total,0x00,sizeof(total));
    for(int n=0; n<dataNum; n++)
    {
        feedForward(pics[n]);
        temp = bp();
        for(int i=0; i<HIDDEN_LAYER_NUM; i++)
        {
            for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
            {
                total.dCHiddenBIAS[i][j] += temp.dCHiddenBIAS[i][j];
            }
        }
        for(int i=0; i<OUTPUT_NUM; i++)
        {
            total.dCOutputBIAS[i] += temp.dCOutputBIAS[i];
            for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
            {
                total.dCOutputWeight[i][j] += temp.dCOutputWeight[i][j];
            }
        }
        for(int i=0; i<HIDDEN_LAYER_SIZE; i++)
        {
            for(int j=0; j<INPUT_NUM; j++)
            {
                total.dCFirstHiddenWeight[i][j] += temp.dCFirstHiddenWeight[i][j];
            }
        }
        for(int i=0; i<HIDDEN_LAYER_NUM-1; i++)
        {
            for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
            {
                for(int k=0; k<HIDDEN_LAYER_SIZE; k++)
                {
                    total.dCHiddenRestWeight[i][j][k] += temp.dCHiddenRestWeight[i][j][k];
                }
            }
        }
    }
    for(int i=0; i<HIDDEN_LAYER_NUM-1; i++)
    {
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            hiddenLayers[i][j].bias -= total.dCHiddenBIAS[i+1][j] * LEARNING_RATE;
        }
    }
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        outputLayer[i].bias -= total.dCOutputBIAS[i] * LEARNING_RATE;
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            outputLayer[i].weight[j] -= total.dCOutputWeight[i][j] * LEARNING_RATE;
        }
    }
    for(int i=0; i<HIDDEN_LAYER_SIZE; i++)
    {
        firstHiddenLayer[i].bias -= total.dCHiddenBIAS[0][i] * LEARNING_RATE;
        for(int j=0; j<INPUT_NUM; j++)
        {
            firstHiddenLayer[i].weight[j] -= total.dCFirstHiddenWeight[i][j] * LEARNING_RATE;
        }
    }
    for(int i=0; i<HIDDEN_LAYER_NUM-1; i++)
    {
        for(int j=0; j<HIDDEN_LAYER_SIZE; j++)
        {
            for(int k=0; k<HIDDEN_LAYER_SIZE; k++)
            {
                hiddenLayers[i][j].weight[k] -= total.dCHiddenRestWeight[i][j][k] * LEARNING_RATE;
            }
        }
    }
}

void test(struct Sample pic)
{
    feedForward(pic);
    printf("cost:%llf\n",cost());
    for(int i=0;i<OUTPUT_NUM;i++)
    {
        printf(" %d:%llf\n",i,outputLayer[i].value);
    }
}

struct Sample inputSample()
{
    struct Sample result;
    for(int i=0;i<INPUT_NUM;i++)
    {
        scanf("%lf",&result.input[i]);
    }
    for(int i=0;i<OUTPUT_NUM;i++)
    {
        scanf("%lf",&result.output[i]);
    }
    return result;
}

int main()
{
    //srand((unsigned)time(NULL));
    initialize();
    struct Sample pic[2];
    int t;
    while(1)
    {
        pic[0]=inputSample();
        pic[1]=inputSample();
        test(pic[0]);
        test(pic[1]);
        scanf("%d",&t);
        for(int i=0;i<t;i++)
        {
             learn(2,pic);
        }
    }
    return 0;
}
/*
0 1 1 0 0
0 1 1 0 0
0 1 1 0 0
0 1 1 0 0
0 0 0 0 0
0 1 0 0 0 0 0 0 0 0

0 1 0.9 1 0
0 1 0 1 0
0 1 1 1 0
0 1 0.1 1 0
0 0.8 1 0.9 0
0 0 0 0 0 0 0 0 1 0

0 0 0 0 0
0 1 1 1 0
0 1 0 1 0
0 1 1 1 0
0 0 0 0 0
1 0 0 0 0 0 0 0 0 0

0 1 1 0 0
0 1 1 0 0
0 1 1 0 0
0 1 1 0 0
0 0 0 0 0
0 1 0 0 0 0 0 0 0 0
*/
