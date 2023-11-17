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

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

int THREADS_PER_BLOCK = 256;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

int getPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

__global__ void prescan_arbitrary(int *output, const int *input, int n,
                                  int powerOfTwo) {
  extern __shared__ int temp[];
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
  for (int d = powerOfTwo >> 1; d > 0; d >>= 1) {
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
    temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;
  }

  for (int d = 1; d < powerOfTwo; d *= 2) {
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

__global__ void prescan_large(int *output, const int *input, int n, int *sums) {
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
  for (int d = n >> 1; d > 0; d >>= 1) {
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

  for (int d = 1; d < n; d *= 2) {
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

__global__ void add(int *output, int length, const int *n) {
  output[blockIdx.x * length + threadIdx.x] += n[blockIdx.x];
}

__global__ void add(int *output, int length, const int *n1, const int *n2) {
  output[blockIdx.x * length + threadIdx.x] += n1[blockIdx.x] + n2[blockIdx.x];
}

void scanSmallDeviceArray(int *d_out, const int *d_in, int length) {
  int powerOfTwo = getPowerOfTwo(length);

  prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(
      d_out, d_in, length, powerOfTwo);
}

void scanLargeDeviceArray(int *d_out, const int *d_in, int length);
void scanLargeEvenDeviceArray(int *d_out, const int *d_in, int length) {
  const int blocks = length / ELEMENTS_PER_BLOCK;
  const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

  int *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));

  prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(
      d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

  const int sumsArrThreadsNeeded = (blocks + 1) / 2;
  if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
    // perform a large scan on the sums arr
    scanLargeDeviceArray(d_incr, d_sums, blocks);
  } else {
    // only need one block to scan sums arr so can use small scan
    scanSmallDeviceArray(d_incr, d_sums, blocks);
  }

  add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

  cudaFree(d_sums);
  cudaFree(d_incr);
}

void scanLargeDeviceArray(int *d_out, const int *d_in, int length) {
  int remainder = length % (ELEMENTS_PER_BLOCK);
  if (remainder == 0) {
    scanLargeEvenDeviceArray(d_out, d_in, length);
  } else {
    // perform a large scan on a compatible multiple of elements
    int lengthMultiple = length - remainder;
    scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);

    // scan the remaining elements and add the (inclusive) last element of the
    // large scan to this
    int *startOfOutputArray = &(d_out[lengthMultiple]);
    scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]),
                         remainder);

    add<<<1, remainder>>>(startOfOutputArray, remainder,
                          &(d_in[lengthMultiple - 1]),
                          &(d_out[lengthMultiple - 1]));
  }
}

void scan(int *output, const int *input, int length) {
  if (length > ELEMENTS_PER_BLOCK) {
    scanLargeDeviceArray(output, input, length);
  } else {
    scanSmallDeviceArray(output, input, length);
  }
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
  scan(d_output, d_input, size);
}