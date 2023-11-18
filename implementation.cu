#include "implementation.h"

#include "stdio.h"
#include <cstdint>
#include <driver_types.h>

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

int block_threads = 4;
int block_elems = block_threads * 2;

int get_power_of_two(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

__global__ void prescan_arbitrary(int *output, const int *input, int n,
                                  int power_of_two, bool is_inclusive) {
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
    int target_val = 0;
    if (is_inclusive) {
      // in inclusive mode, instead of 0, set as corresponding value in input to
      // make the result inclusive
      target_val = power_of_two - 1 < n ? input[power_of_two - 1] : 0;
    }
    temp[power_of_two - 1 + CONFLICT_FREE_OFFSET(power_of_two - 1)] =
        target_val;
  }

  for (int d = 1; d < power_of_two; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      int ai_no_offset = ai, bi_no_offset = bi;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      if (is_inclusive) {
        // when assigning bi to ai, first minus the input value at bi, then add
        // the input value at ai to make the result inclusive
        int input_ai = ai_no_offset < n ? input[ai_no_offset] : 0;
        int input_bi = bi_no_offset < n ? input[bi_no_offset] : 0;
        temp[ai] = temp[bi] - input_bi + input_ai;
      } else {
        temp[ai] = temp[bi];
      }
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (thread_id < n) {
    output[ai] = temp[ai + bank_offset_a];
    output[bi] = temp[bi + bank_offset_b];
  }
}

__global__ void prescan_large(int *output, const int *input, int n, int *sums,
                              bool is_inclusive) {
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

    // in inclusive mode, instead of 0, set as corresponding value in input to
    // make the result inclusive
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = is_inclusive ? input[n - 1] : 0;
  }

  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      int ai = offset * (2 * thread_id + 1) - 1;
      int bi = offset * (2 * thread_id + 2) - 1;
      int ai_no_offset = ai, bi_no_offset = bi;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      if (is_inclusive) {
        int input_ai = input[ai_no_offset], input_bi = input[bi_no_offset];
        temp[ai] = temp[bi] - input_bi + input_ai;
      } else {
        temp[ai] = temp[bi];
      }
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[block_offset + ai] = temp[ai + bank_offset_a];
  output[block_offset + bi] = temp[bi + bank_offset_b];
}

__global__ void add(int *output, int block_elems, const int *n) {
  output[blockIdx.x * block_elems + threadIdx.x] += n[blockIdx.x];
}

__global__ void add(int *output, const int *n1) { output[threadIdx.x] += *n1; }

__global__ void add(int *output, const int *n1, const int *n2) {
  output[threadIdx.x] += *n1 + *n2;
}

__global__ void exclusive_to_inclusive(int *output, int block_threads,
                                       int length, const int *input) {
  extern __shared__ int temp[];

  int thread_id = threadIdx.x;
  int pos = thread_id;
  // printf("pos:%d, length:%d\n", pos, length);
  while (pos < length) {
    temp[pos] = output[pos];
    pos += block_threads;
  }

  pos = thread_id;
  __syncthreads();

  while (pos < length) {
    if (pos < length - 1) {
      output[pos] = temp[pos + 1];
    }
    if (pos == length - 1) {
      output[pos] = temp[pos] + input[pos];
    }
    pos += block_threads;
  }
}

void scan_small(int *output, const int *input, int length, bool is_inclusive) {
  int power_of_two = get_power_of_two(length);
  prescan_arbitrary<<<1, (length + 1) / 2, 2 * power_of_two * sizeof(int)>>>(
      output, input, length, power_of_two, is_inclusive);
}

void scan_large(int *output, const int *input, int length, bool is_inclusive);
void scan_equal(int *output, const int *input, int length, bool is_inclusive) {
  const int blocks = length / block_elems;
  const int sharedMemSize = block_elems * sizeof(int);

  int *sums, *incr;
  cudaMalloc((void **)&sums, blocks * sizeof(int));
  cudaMalloc((void **)&incr, blocks * sizeof(int));

  prescan_large<<<blocks, block_threads, 2 * sharedMemSize>>>(
      output, input, block_elems, sums, is_inclusive);

  // always scan incr in an exclusive mode
  if ((blocks + 1) / 2 > block_threads) {
    scan_large(incr, sums, blocks, false);
  } else {
    scan_small(incr, sums, blocks, false);
  }

  // int32_t h_array[blocks];
  // cudaMemcpy(h_array, sums, blocks * sizeof(int32_t),
  // cudaMemcpyDeviceToHost); printf("sum:%d\n", h_array[0]);
  // cudaMemcpy(h_array, incr, blocks * sizeof(int32_t),
  // cudaMemcpyDeviceToHost); printf("incr:%d\n", h_array[0]);

  // int32_t h_output[8];
  // cudaMemcpy(h_output, output, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 8; ++i) {
  //   printf("%d,", h_output[i]);
  // }
  // printf("\n");

  add<<<blocks, block_elems>>>(output, block_elems, incr);

  cudaFree(sums);
  cudaFree(incr);
}

void scan_large(int *output, const int *input, int length, bool is_inclusive) {
  int remainder = length % (block_elems);
  int even_length = length - remainder;
  scan_equal(output, input, even_length, is_inclusive);
  if (remainder > 0) {
    scan_small(&(output[even_length]), &(input[even_length]), remainder,
               is_inclusive);

    // int32_t h_array[1];
    // cudaMemcpy(h_array, &(output[even_length]), sizeof(int32_t),
    //            cudaMemcpyDeviceToHost);
    // printf("small:%d\n", h_array[0]);

    if (is_inclusive) {
      add<<<1, remainder>>>(&(output[even_length]), &(output[even_length - 1]));
    } else {
      // input[even_length-1] is needed in exclusive mode to make the former
      // result inclusive
      add<<<1, remainder>>>(&(output[even_length]), &(input[even_length - 1]),
                            &(output[even_length - 1]));
    }
  }
}

void inclusive_scan(int *output, const int *input, int length) {
  if (length > block_elems) {
    scan_large(output, input, length, true);
  } else {
    scan_small(output, input, length, true);
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
  inclusive_scan(d_output, d_input, size);
  // int sharedMemoryBytes = size * sizeof(int32_t);
  // cudaFuncSetAttribute(exclusive_to_inclusive,
  //                      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      sharedMemoryBytes);
  // exclusive_to_inclusive<<<1, 512, size * sizeof(int32_t)>>>(d_output, 512,
  //                                                            size, d_input);
  // cudaDeviceSynchronize();
}