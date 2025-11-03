#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    /* Build K stack for keys[0..i] by vertical concat in HBM */
    Matrix *k_stack = nullptr;
    if (i == 0) {
      k_stack = keys[0];
    } else {
      Matrix *cur = keys[0];
      for (size_t j = 1; j <= i; ++j) {
        Matrix *next_concat = matrix_memory_allocator.Allocate("k_concat");
        gpu_sim.Concat(cur, keys[j], next_concat, /*axis=*/0, Position::kInGpuHbm);
        if (cur != keys[0]) {
          gpu_sim.ReleaseMatrix(cur);
        }
        cur = next_concat;
      }
      k_stack = cur;
    }

    /* Build V stack for values[0..i] by vertical concat in HBM */
    Matrix *v_stack = nullptr;
    if (i == 0) {
      v_stack = values[0];
    } else {
      Matrix *cur = values[0];
      for (size_t j = 1; j <= i; ++j) {
        Matrix *next_concat = matrix_memory_allocator.Allocate("v_concat");
        gpu_sim.Concat(cur, values[j], next_concat, /*axis=*/0, Position::kInGpuHbm);
        if (cur != values[0]) {
          gpu_sim.ReleaseMatrix(cur);
        }
        cur = next_concat;
      }
      v_stack = cur;
    }

    /* Move operands needed for MatMul to shared memory */
    gpu_sim.MoveMatrixToSharedMem(current_query);
    if (k_stack->GetPosition() != Position::kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(k_stack);

    /* Transpose K to get K^T in SRAM */
    gpu_sim.Transpose(k_stack, Position::kInSharedMemory);

    /* logits = Q * K^T (shape (i+1, i+1)) in SRAM */
    Matrix *logits = matrix_memory_allocator.Allocate("logits");
    gpu_sim.MatMul(current_query, k_stack, logits);

    /* Softmax row-wise on logits to produce attn (i+1, i+1) in SRAM */
    Matrix *attn = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix *row_mat = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(logits, row, row_mat, Position::kInSharedMemory);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_mat, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      if (row == 0) {
        attn = row_soft;
      } else {
        Matrix *new_attn = matrix_memory_allocator.Allocate("attn_concat");
        gpu_sim.Concat(attn, row_soft, new_attn, /*axis=*/0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(attn);
        attn = new_attn;
      }
      /* release temporaries */
      gpu_sim.ReleaseMatrix(row_mat);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      if (row != 0) {
        gpu_sim.ReleaseMatrix(row_soft);
      }
    }

    /* Prepare V stack in SRAM for final matmul */
    if (v_stack->GetPosition() != Position::kInSharedMemory)
      gpu_sim.MoveMatrixToSharedMem(v_stack);

    /* out = softmax(QK^T) * V (shape (i+1, d)) in SRAM */
    Matrix *out = matrix_memory_allocator.Allocate("out");
    gpu_sim.MatMul(attn, v_stack, out);

    /* Release intermediates no longer needed */
    gpu_sim.ReleaseMatrix(k_stack);
    gpu_sim.ReleaseMatrix(logits);
    gpu_sim.ReleaseMatrix(attn);

    /* Move result to HBM for commit */
    gpu_sim.MoveMatrixToGpuHbm(out);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*out);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
