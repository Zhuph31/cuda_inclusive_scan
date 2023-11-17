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

int block_threads = 256;
int block_elems = block_threads * 2;

int get_power_of_two(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

__global__ void prescan_arbitrary(int *output, const int *input, int n,
                                  int power_of_two) {
  extern __shared__ int temp[];
  int thread_id = threadIdx.x;

  int ai = thread_id;
  int bi = thread_id + (n / 2);
  int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
  int bank_offset_b = CONFLICT_FREE_OFFSET(bi);

  if (thread_id < n) {
    temp[ai + bank_offset_a] = input[ai];
    temp[bi + bank_offset_b] = input[bi];
  } else {
    temp[ai + bank_offset_a] = 0;
    temp[bi + bank_offset_b] = 0;
  }

  int offset = 1;
  for (int d = power_of_two >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thread_id == 0) {
    temp[power_of_two - 1 + CONFLICT_FREE_OFFSET(power_of_two - 1)] = 0;
  }

  for (int d = 1; d < power_of_two; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (thread_id < n) {
    output[ai] = temp[ai + bank_offset_a];
    output[bi] = temp[bi + bank_offset_b];
  }
}

__global__ void prescan_large(int *output, const int *input, int n, int *sums) {
  extern __shared__ int temp[];
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * n;

  int ai = thread_id;
  int bi = thread_id + (n / 2);
  int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
  int bank_offset_b = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bank_offset_a] = input[block_offset + ai];
  temp[bi + bank_offset_b] = input[block_offset + bi];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (thread_id == 0) {
    sums[block_id] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[block_offset + ai] = temp[ai + bank_offset_a];
  output[block_offset + bi] = temp[bi + bank_offset_b];
}

__global__ void add(int *output, int length, const int *n) {
  output[blockIdx.x * length + threadIdx.x] += n[blockIdx.x];
}

__global__ void add(int *output, int length, const int *n1, const int *n2) {
  output[blockIdx.x * length + threadIdx.x] += n1[blockIdx.x] + n2[blockIdx.x];
}

void scan_small(int *output, const int *input, int length) {
  int power_of_two = get_power_of_two(length);
  prescan_arbitrary<<<1, (length + 1) / 2, 2 * power_of_two * sizeof(int)>>>(
      output, input, length, power_of_two);
}

void scan_large(int *output, const int *input, int length);
void scan_equal(int *output, const int *input, int length) {
  const int blocks = length / block_elems;
  const int sharedMemSize = block_elems * sizeof(int);

  int *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));

  prescan_large<<<blocks, block_threads, 2 * sharedMemSize>>>(
      output, input, block_elems, d_sums);

  if ((blocks + 1) / 2 > block_threads) {
    scan_large(d_incr, d_sums, blocks);
  } else {
    scan_small(d_incr, d_sums, blocks);
  }
  add<<<blocks, block_elems>>>(output, block_elems, d_incr);

  cudaFree(d_sums);
  cudaFree(d_incr);
}

void scan_large(int *output, const int *input, int length) {
  int remainder = length % (block_elems);
  int even_length = length - remainder;
  scan_equal(output, input, even_length);
  if (remainder > 0) {
    scan_small(&(output[even_length]), &(input[even_length]), remainder);
    add<<<1, remainder>>>(&(output[even_length]), remainder,
                          &(input[even_length - 1]),
                          &(output[even_length - 1]));
  }
}

void scan(int *output, const int *input, int length) {
  if (length > block_elems) {
    scan_large(output, input, length);
  } else {
    scan_small(output, input, length);
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