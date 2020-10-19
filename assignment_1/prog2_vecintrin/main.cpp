#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
#include <CycleTimer.h>
using namespace std;

extern void absSerial(float *values, float *output, int N);
extern void absVector(float *values, float *output, int N);
extern void clampedExpSerial(float *values, int *exponents, float *output, int N);
extern void clampedExpVector(float *values, int *exponents, float *output, int N);
extern float arraySumSerial(float *values, int N);
extern float arraySumVector(float *values, int N);

#define EXP_MAX 1024

Logger CMU418Logger;

void usage(const char *progname);
void initValue(float *values, int *exponents, float *output, float *gold, unsigned int N);
void absSerial(float *values, float *output, int N);
void absVector(float *values, float *output, int N);
void clampedExpSerial(float *values, int *exponents, float *output, int N);
void clampedExpVector(float *values, int *exponents, float *output, int N);
float arraySumSerial(float *values, int N);
float arraySumVector(float *values, int N);
bool verifyResult(float *values, int *exponents, float *output, float *gold, int N);

int main(int argc, char *argv[])
{
  int N = 16;
  bool printLog = false;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
      {"size", 1, 0, 's'},
      {"log", 0, 0, 'l'},
      {"help", 0, 0, '?'},
      {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "s:l?", long_options, NULL)) != EOF)
  {

    switch (opt)
    {
    case 's':
      N = atoi(optarg);
      if (N <= 0)
      {
        printf("Error: Workload size is set to %d (<0).\n", N);
        return -1;
      }
      break;
    case 'l':
      printLog = true;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }

  float *values = new float[N + VECTOR_WIDTH];
  int *exponents = new int[N + VECTOR_WIDTH];
  float *output = new float[N + VECTOR_WIDTH];
  float *gold = new float[N + VECTOR_WIDTH];
  initValue(values, exponents, output, gold, N);

  //clampedExpSerial(values, exponents, gold, N);
  //clampedExpVector(values, exponents, output, N);
  double minSerial = 1e30;
  double minVector = 1e30;

  double startTime = CycleTimer::currentSeconds();
  clampedExpSerial(values, exponents, gold, N);
  double endTime = CycleTimer::currentSeconds();
  minSerial = endTime - startTime;

  startTime = CycleTimer::currentSeconds();
  clampedExpVector(values, exponents, output, N);
  endTime = CycleTimer::currentSeconds();

  minVector = endTime - startTime;

  printf("[clampedExp serial]:\t\t[%.3f] ms\n", minSerial * 1000);
  printf("[clampedExp Vector]:\t\t[%.3f] ms\n", minVector * 1000);
  //absSerial(values, gold, N);
  //absVector(values, output, N);

  printf("\e[1;31mCLAMPED EXPONENT\e[0m (required) \n");
  bool clampedCorrect = verifyResult(values, exponents, output, gold, N);
  if (printLog)
    CMU418Logger.printLog();
  CMU418Logger.printStats();

  printf("************************ Result Verification *************************\n");
  if (!clampedCorrect)
  {
    printf("@@@ Failed!!!\n");
  }
  else
  {
    printf("Passed!!!\n");
  }

  printf("\n\e[1;31mARRAY SUM\e[0m (bonus) \n");
  if (N % VECTOR_WIDTH == 0)
  {
    float sumGold = arraySumSerial(values, N);
    float sumOutput = arraySumVector(values, N);

    double startTime = CycleTimer::currentSeconds();
    arraySumSerial(values, N);
    double endTime = CycleTimer::currentSeconds();
    minSerial = endTime - startTime;

    startTime = CycleTimer::currentSeconds();
    arraySumVector(values, N);
    endTime = CycleTimer::currentSeconds();
    minVector = endTime - startTime;

    printf("[sum serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    printf("[sum Vector]:\t\t[%.3f] ms\n", minVector * 1000);

    float epsilon = 0.1;
    bool sumCorrect = abs(sumGold - sumOutput) < epsilon * 2;
    if (!sumCorrect)
    {
      printf("Expected %f, got %f\n.", sumGold, sumOutput);
      printf("@@@ Failed!!!\n");
    }
    else
    {
      printf("Passed!!!\n");
    }
  }
  else
  {
    printf("Must have N % VECTOR_WIDTH == 0 for this problem (VECTOR_WIDTH is %d)\n", VECTOR_WIDTH);
  }

  delete[] values;
  delete[] exponents;
  delete[] output;
  delete gold;

  return 0;
}

void usage(const char *progname)
{
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -s  --size <N>     Use workload size N (Default = 16)\n");
  printf("  -l  --log          Print vector unit execution log\n");
  printf("  -?  --help         This message\n");
}

void initValue(float *values, int *exponents, float *output, float *gold, unsigned int N)
{

  for (unsigned int i = 0; i < N + VECTOR_WIDTH; i++)
  {
    // random input values
    values[i] = -1.f - 0.01f * static_cast<float>(rand()) / RAND_MAX;
    exponents[i] = rand() % EXP_MAX;
    output[i] = 0.f;
    gold[i] = 0.f;
  }
}

bool verifyResult(float *values, int *exponents, float *output, float *gold, int N)
{
  int incorrect = -1;
  float epsilon = 0.00001;
  for (int i = 0; i < N + VECTOR_WIDTH; i++)
  {
    if (abs(output[i] - gold[i]) > epsilon)
    {
      incorrect = i;
      break;
    }
  }

  //printf("%d\n", N + VECTOR_WIDTH);

  if (incorrect != -1)
  {
    if (incorrect >= N)
      printf("You have written to out of bound value!\n");

    printf("Wrong calculation at value[%d]!\n", incorrect);
    printf("value  = ");
    for (int i = 0; i < N + VECTOR_WIDTH; i++)
    {
      printf("% f ", values[i]);
    }
    printf("\n");

    printf("exp    = ");
    for (int i = 0; i < N + VECTOR_WIDTH; i++)
    {
      printf("% 9d ", exponents[i]);
    }
    printf("\n");

    printf("output = ");
    for (int i = 0; i < N + VECTOR_WIDTH; i++)
    {
      printf("% f ", output[i]);
    }
    printf("\n");

    printf("gold   = ");
    for (int i = 0; i < N + VECTOR_WIDTH; i++)
    {
      printf("% f ", gold[i]);
    }
    printf("\n");
    return false;
  }
  printf("Results matched with answer!\n");
  return true;
}
