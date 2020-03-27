#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_NUM 25
#define HIDDEN_LAYER_NUM 2
#define HIDDEN_LAYER_SIZE 16
#define HIDDEN1_NUM 16
#define HIDDEN2_NUM 16
#define OUTPUT_NUM 10
#define LEARNING_RATE 0.1

double inputValue[INPUT_NUM];
double preferredOutput[OUTPUT_NUM];

struct InputCell
{
    double value;
} inputLayer[INPUT_NUM];

struct HiddenCell
{
    double bias;
    double weight[INPUT_NUM];
    double value;
} hiddenLayers[HIDDEN_LAYER_NUM][HIDDEN_LAYER_SIZE];

struct HiddenCell1
{
    double bias;
    double weight[INPUT_NUM];
    double value;
} hiddenLayer1[HIDDEN1_NUM];

struct HiddenCell2
{
    double bias;
    double weight[HIDDEN1_NUM];
    double value;
} hiddenLayer2[HIDDEN2_NUM];

struct OutputCell
{
    double bias;
    double weight[HIDDEN2_NUM];
    double value;
} outputLayer[OUTPUT_NUM];

struct BPData
{
    double dCbH1[HIDDEN1_NUM];
    double dCbH2[HIDDEN2_NUM];
    double dCbO[OUTPUT_NUM];
    double dCwH1[HIDDEN1_NUM][INPUT_NUM];
    double dCwH2[HIDDEN2_NUM][HIDDEN1_NUM];
    double dCwO[OUTPUT_NUM][HIDDEN2_NUM];
};

double logistic(double rawValue)
{
    return 1/(1+pow(2.89,-rawValue));
}

void initialize()
{
    for(int i=0; i<HIDDEN1_NUM; i++)//425
    {
        hiddenLayer1[i].bias = logistic(rand()%4*(rand()%2?-1:1));
        for(int j=0; j<INPUT_NUM; j++)
        {
            hiddenLayer1[i].weight[j] = logistic(rand()%4*(rand()%2?-1:1));
        }
    }
    for(int i=0; i<HIDDEN2_NUM; i++)//272
    {
        hiddenLayer2[i].bias = logistic(rand()%4*(rand()%2?-1:1));
        for(int j=0; j<HIDDEN1_NUM; j++)
        {
            hiddenLayer2[i].weight[j] = logistic(rand()%4*(rand()%2?-1:1));
        }
    }
    for(int i=0; i<OUTPUT_NUM; i++)//170
    {
        outputLayer[i].bias = logistic(rand()%4*(rand()%2?-1:1));
        for(int j=0; j<HIDDEN2_NUM; j++)
        {
            outputLayer[i].weight[j] = logistic(rand()%4*(rand()%2?-1:1));
        }
    }
}

void feedForward()
{
    for(int i=0; i<INPUT_NUM; i++)//input
    {
        inputLayer[i].value = inputValue[i];
    }
    for(int i=0; i<HIDDEN1_NUM; i++)//calculate 1st hiddenlayer's value
    {
        hiddenLayer1[i].value = hiddenLayer1[i].bias;
        for(int j=0; j<INPUT_NUM; j++)//transverse input
        {
            hiddenLayer1[i].value += inputLayer[j].value * hiddenLayer1[i].weight[j];
        }
        hiddenLayer1[i].value = logistic(hiddenLayer1[i].value);
    }
    for(int i=0; i<HIDDEN2_NUM; i++)//calculate 2nd hiddenlayer's value
    {
        hiddenLayer2[i].value = hiddenLayer2[i].bias;
        for(int j=0; j<HIDDEN1_NUM; j++)//transverse 1st hiddenlayer
        {
            hiddenLayer2[i].value += hiddenLayer1[j].value * hiddenLayer2[i].weight[j];
        }
        hiddenLayer2[i].value = logistic(hiddenLayer2[i].value);
    }
    for(int i=0; i<OUTPUT_NUM; i++)//calculate outputlayer's value
    {
        outputLayer[i].value = outputLayer[i].bias;
        for(int j=0; j<HIDDEN2_NUM; j++)//transverse 2nd hiddenlayer
        {
            outputLayer[i].value += hiddenLayer2[j].value * outputLayer[i].weight[j];
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
/* for sigmoid is logistic function
N: value
dN[a][j] / dbias[a-1][j] = (1-N[a][j])*N[a][j]
dN[a][j] / dW[a-1][j][i] = N[a-1][i] * dN[a][j]/dbias[a-1][j]
dN[a][j] / dN[a-1][i] = rate[a-1][j][i] * dN[a][j]/dbias[a-1][j]
dC/dN[a][j] = N[a][j]-Z

dC/dbias[a-1][j] = dC/dN[a][j] * dN[a][j]/dbias[a-1][j]

dC/dW[a-1][j][i] = dC/dN[a][j] * dN[a][j]/dW[a-1][j][i]

dC/dN[a-1][i] = dC/dN[a][j] * dN[a][j]/dN[a-1][i]

dC/dbias[a-2][j] =  dC/dN[a-1][i] * dN[a-1][j] / dbias[a-2][j]

dC/dW[a-2][j][i] = dC/dN[a-1][i] * dN[a-1][j] / dW[a-2][j]

dc/dbO = dc/do * do/dbo = (o - z)*(1-o)*o
dc/dwo = dc/dbo* h2[j]

*/
struct BPData bp()
{
    struct BPData data;
    double tempDCBH2,tempDCBH1;
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        data.dCbO[i]=(outputLayer[i].value-preferredOutput[i])*(1-outputLayer[i].value) * outputLayer[i].value;
        for(int j=0; j<HIDDEN2_NUM; j++)
        {
            data.dCwO[i][j] = data.dCbO[i]*hiddenLayer2[j].value;
            tempDCBH2 = data.dCbO[i]*outputLayer[i].weight[j] * (1-hiddenLayer2[j].value) * hiddenLayer2[j].value;
            data.dCbH2[j] += tempDCBH2;
            for(int k=0; k<HIDDEN1_NUM; k++)
            {
                data.dCwH2[j][k] += tempDCBH2 * hiddenLayer1[k].value;
                tempDCBH1 = tempDCBH2*hiddenLayer2[j].weight[k] * (1-hiddenLayer1[k].value)*hiddenLayer1[k].value;
                data.dCbH1[k] +=tempDCBH1;
                for(int l=0;l<INPUT_NUM;l++)
                {
                    data.dCwH1[k][l] += tempDCBH1 * inputLayer[l].value;
                }
            }
        }
    }
    return data;
}

int main()
{
    srand((unsigned)time(NULL));
    printf("Hello world!\n");
    return 0;
}
