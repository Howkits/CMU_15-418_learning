#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void printCudaTerm(int * input,int length)
{
    printf("printing.....:");
    int * output=new int[length];
    cudaMemcpy(output, input, length * sizeof(int),
               cudaMemcpyDeviceToHost);
    for(int i=0;i<length;i++)
    {
        printf("%d ",output[i]);
    }
    printf("\n");
}

__global__ void scan(int* device_start, int length, int* device_result)   //my solution
{
    __shared__ int temp[128*4];

    int i=blockIdx.x*blockDim.x+threadIdx.x;
    // printf("from thread %d:",threadIdx.x);
    if (i<length)
    {
        temp[threadIdx.x]=device_start[i];
    }
    for(unsigned int stride=1;stride<=length/2;stride*=2)
    {
        __syncthreads();
        int index=(threadIdx.x+1)*stride*2-1;
        if(index<length)
            temp[index]+=temp[index-stride];
        __syncthreads();
    }    
    
    temp[length-1]=0;

    for(unsigned int stride=length/2;stride>0;stride/=2)
    {
        __syncthreads();
        int index1=(threadIdx.x+1)*stride*2-1;
        
        if((index1<length)&&((index1-stride)<length))
        { 
            int t=temp[index1];
            temp[index1]=temp[index1-stride];
            temp[index1-stride]=t;
            temp[index1]+=temp[index1-stride];
        }
        __syncthreads();
    }
    
    if(i<length)
        device_result[i]=temp[i];
    
    // printf("res:%d\n",device_result[i]);
}

__global__ void exclusive_scan_gpu(int* input, int* output, int n)    //offical solution with Avoiding Bank Conflicts
{
    __shared__ int temp[4 * 64];
    int thid_global = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    int thid = threadIdx.x;
  
    {  
      int offset = 1;
      //temp[2 * thid] = input[2 * thid_global];
      //temp[2 * thid + 1] = input[2 * thid_global + 1];
      
      int aind = thid;
      int bind = thid + n / 2;
      int bankOffsetA = CONFLICT_FREE_OFFSET(aind);
      int bankOffsetB = CONFLICT_FREE_OFFSET(bind);
      temp[aind + bankOffsetA] = input[thid_global];
      temp[bind + bankOffsetB] = input[thid_global + n / 2];  
       
  
      for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          ai += CONFLICT_FREE_OFFSET(ai);
          bi += CONFLICT_FREE_OFFSET(bi);
          temp[bi] += temp[ai];
        }
        offset *= 2;
      }
  
      if (thid == 0) { 
        //temp[n - 1] = 0;
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
      }
  
      for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          ai += CONFLICT_FREE_OFFSET(ai);
          bi += CONFLICT_FREE_OFFSET(bi);
          int t = temp[ai];
          temp[ai] = temp[bi];
          temp[bi] += t;
        }
      }
  
      __syncthreads();
      //output[2 * thid_global] = temp[2 * thid];
      //output[2 * thid_global + 1] = temp[2 * thid + 1];
      //printf("%d:%d %d:%d\n", 2 * thid_global, output[2 * thid_global], 2 * thid_global + 1, output[2 * thid_global + 1]);
      output[thid_global] = temp[aind + bankOffsetA];
      output[thid_global + n / 2] = temp[bind + bankOffsetB];
    }
}

__global__ void add_base_gpu(int* device_input, int* device_output, int block_index) 
{
    int block_last_element = block_index * 128 * 2 - 1;
    
    int base = device_input[block_last_element] + device_output[block_last_element];
    
    int thid = block_index * blockDim.x + threadIdx.x;
  
    device_output[2 * thid] += base;
    device_output[2 * thid + 1] += base;
}


void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */

    const int threadsPerBlock = 128;
    // const int threadsPerBlock = 16;
    // int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    int blocks =  length / (threadsPerBlock * 2);
    if(blocks==0)
        blocks=1;
    // scan<<<blocks,threadsPerBlock>>>(device_start,threadsPerBlock,device_result);
    exclusive_scan_gpu<<<blocks,threadsPerBlock>>>(device_start,device_result,length/blocks);

    cudaThreadSynchronize();

    for(int i=1;i<blocks;i++)
    {
        add_base_gpu<<<1,threadsPerBlock>>>(device_start,device_result,i);
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    // exclusive_scan(device_input, end - inarray, device_result);
    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void mark_repeats_gpu(int *device_input,int length,int *flags)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<length-1)
    {
        if(device_input[i]==device_input[i+1])
            flags[i]=1;
        else
            flags[i]=0;
    }
}

__global__ void  get_repeat_results(int* input,int* flags,int* flags_scanned,int length,int* output)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if((i<length-1)&&(flags_scanned[i]<flags_scanned[i+1]))
    {
        output[flags_scanned[i]]=i;
    }
}

int find_repeats(int *device_input, int length, int N, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    
    const int threadsPerBlock = 128;
    // const int threadsPerBlock = 16;
    // const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    int blocks = length/(threadsPerBlock);
    if(blocks==0)
        blocks=1;

    int *flags;
    int *flags_scanned;
    cudaMalloc((void **)&flags,length*sizeof(int));
    cudaMalloc((void **)&flags_scanned,length*sizeof(int));
    
    mark_repeats_gpu<<<blocks,threadsPerBlock>>>(device_input,length,flags);
    
    exclusive_scan(flags,length,flags_scanned);

    // cudaThreadSynchronize();

    // for (int i = 1; i < blocks; i++)
    //     add_base_gpu<<<1, threadsPerBlock>>>(flags, flags_scanned, i);

    // printCudaTerm(flags,length);
    // printCudaTerm(flags_scanned,length);

    get_repeat_results<<<blocks,threadsPerBlock>>>(device_input,flags,flags_scanned,N,device_output);

    // cudaThreadSynchronize();

    // printCudaTerm(device_output,length);

    // printf("===========================\n");
    
    int *output=new int[length];
    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);
    if(N<=1)
        return 0;
    int ans=1;
    for(int i=1;i<N;i++)
    {
        if(output[i]>0&&output[i]>output[i-1])
            ans++;
        else
            break;
    }

    return ans;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, rounded_length, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}

