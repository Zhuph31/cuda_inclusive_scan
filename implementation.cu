#include "implementation.h"

#include "stdio.h"
#include <cstdint>

void printSubmissionInfo() {
  // This will be published in the leaderboard on piazza
  // Please modify this field with something interesting
  char nick_name[] = "Penghui Zhu Steven";

  // Please fill in your information (for marking purposes only)
  char student_first_name[] = "Penghui";
  char student_last_name[] = "Zhu";
  char student_student_number[] = "1009763551";

  // Printing out team information
  printf("*********************************************************************"
         "**********************************\n");
  printf("Submission Information:\n");
  printf("\tnick_name: %s\n", nick_name);
  printf("\tstudent_first_name: %s\n", student_first_name);
  printf("\tstudent_last_name: %s\n", student_last_name);
  printf("\tstudent_student_number: %s\n", student_student_number);
}

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

int nextPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

__global__ void prescan_arbitrary(int *output, int *input, int n,
                                  int powerOfTwo) {
  extern __shared__ int temp[]; // allocated on invocation
  int threadID = threadIdx.x;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  if (threadID < n) {
    temp[ai + bankOffsetA] = input[ai];
    temp[bi + bankOffsetB] = input[bi];
  } else {
    temp[ai + bankOffsetA] = 0;
    temp[bi + bankOffsetB] = 0;
  }

  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0;
       d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (threadID == 0) {
    temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] =
        0; // clear the last element
  }

  for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (threadID < n) {
    output[ai] = temp[ai + bankOffsetA];
    output[bi] = temp[bi + bankOffsetB];
  }
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n,
                                              int powerOfTwo) {
  extern __shared__ int temp[]; // allocated on invocation
  int threadID = threadIdx.x;

  if (threadID < n) {
    temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
    temp[2 * threadID + 1] = input[2 * threadID + 1];
  } else {
    temp[2 * threadID] = 0;
    temp[2 * threadID + 1] = 0;
  }

  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0;
       d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (threadID == 0) {
    temp[powerOfTwo - 1] = 0;
  } // clear the last element

  for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (threadID < n) {
    output[2 * threadID] = temp[2 * threadID]; // write results to device memory
    output[2 * threadID + 1] = temp[2 * threadID + 1];
  }
}

__global__ void prescan_large(int *output, int *input, int n, int *sums) {
  extern __shared__ int temp[];

  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = input[blockOffset + ai];
  temp[bi + bankOffsetB] = input[blockOffset + bi];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadID == 0) {
    sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[blockOffset + ai] = temp[ai + bankOffsetA];
  output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_large_unoptimized(int *output, int *input, int n,
                                          int *sums) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  extern __shared__ int temp[];
  temp[2 * threadID] = input[blockOffset + (2 * threadID)];
  temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadID == 0) {
    sums[blockID] = temp[n - 1];
    temp[n - 1] = 0;
  }

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[blockOffset + (2 * threadID)] = temp[2 * threadID];
  output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

__global__ void add(int *output, int length, int *n) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

int THREADS_PER_BLOCK = 1024;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scanLargeDeviceArray(int *output, int *input, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao);

float scan(int *output, int *input, int length, bool bcao) {
  const int arraySize = length * sizeof(int);

  // start timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  if (length > ELEMENTS_PER_BLOCK) {
    scanLargeDeviceArray(output, input, length, bcao);
  } else {
    scanSmallDeviceArray(output, input, length, bcao);
  }

  // end timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  return elapsedTime;
}

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
  int remainder = length % (ELEMENTS_PER_BLOCK);
  if (remainder == 0) {
    scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
  } else {
    // perform a large scan on a compatible multiple of elements
    int lengthMultiple = length - remainder;
    scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

    // scan the remaining elements and add the (inclusive) last element of the
    // large scan to this
    int *startOfOutputArray = &(d_out[lengthMultiple]);
    scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder,
                         bcao);

    add<<<1, remainder>>>(startOfOutputArray, remainder,
                          &(d_in[lengthMultiple - 1]),
                          &(d_out[lengthMultiple - 1]));
  }
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
  int powerOfTwo = nextPowerOfTwo(length);

  if (bcao) {
    prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(
        d_out, d_in, length, powerOfTwo);
  } else {
    prescan_arbitrary_unoptimized<<<1, (length + 1) / 2,
                                    2 * powerOfTwo * sizeof(int)>>>(
        d_out, d_in, length, powerOfTwo);
  }
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
  const int blocks = length / ELEMENTS_PER_BLOCK;
  const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

  int *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));

  if (bcao) {
    prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(
        d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
  } else {
    prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK,
                                2 * sharedMemArraySize>>>(
        d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
  }

  const int sumsArrThreadsNeeded = (blocks + 1) / 2;
  if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
    // perform a large scan on the sums arr
    scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
  } else {
    // only need one block to scan sums arr so can use small scan
    scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
  }

  add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

  cudaFree(d_sums);
  cudaFree(d_incr);
}

/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions,
 * kernels or allocate temporary memory. However, you must not modify other
 * files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
  int32_t *input = const_cast<int32_t *>(d_input);
  printf("calling parllel implementation, size:%lu\n", size);
  scan(d_output, input, size, false);
}