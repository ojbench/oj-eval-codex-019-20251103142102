#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *k_sram_stack = nullptr; // persistent across rounds (m x d)
  Matrix *k_sram_t = nullptr;     // persistent transpose (d x m)
  Matrix *v_sram_stack = nullptr; // persistent across rounds
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

    /* Prepare progressive K/V stacks in SRAM by appending current rows */
    // Prepare K row copy in HBM then move to SRAM
    Matrix *k_row_sram = matrix_memory_allocator.Allocate("k_row_sram");
    gpu_sim.Copy(keys[i], k_row_sram, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(k_row_sram);
    if (i == 0) {
      k_sram_stack = k_row_sram;
    } else {
      Matrix *new_k_stack = matrix_memory_allocator.Allocate("k_stack_sram");
      gpu_sim.Concat(k_sram_stack, k_row_sram, new_k_stack, /*axis=*/0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_sram_stack);
      gpu_sim.ReleaseMatrix(k_row_sram);
      k_sram_stack = new_k_stack;
    }

    // Maintain K^T incrementally: transpose current row to (d x 1) and concat as new column
    Matrix *k_row_t = matrix_memory_allocator.Allocate("k_row_t");
    gpu_sim.Copy(k_sram_stack, k_row_t, Position::kInSharedMemory); // careful: we want to copy the last row, not whole stack
    // Extract last row from k_sram_stack instead
    gpu_sim.ReleaseMatrix(k_row_t);
    Matrix *last_row = matrix_memory_allocator.Allocate("k_last_row");
    gpu_sim.GetRow(k_sram_stack, i, last_row, Position::kInSharedMemory);
    Matrix *last_row_t = matrix_memory_allocator.Allocate("k_last_row_t");
    gpu_sim.Copy(last_row, last_row_t, Position::kInSharedMemory);
    gpu_sim.Transpose(last_row_t, Position::kInSharedMemory);
    gpu_sim.ReleaseMatrix(last_row);
    if (i == 0) {
      k_sram_t = last_row_t;
    } else {
      Matrix *new_k_t = matrix_memory_allocator.Allocate("k_t_sram");
      gpu_sim.Concat(k_sram_t, last_row_t, new_k_t, /*axis=*/1, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_sram_t);
      gpu_sim.ReleaseMatrix(last_row_t);
      k_sram_t = new_k_t;
    }

    // Prepare V row copy similarly
    Matrix *v_row_hbm = matrix_memory_allocator.Allocate("v_row_hbm");
    gpu_sim.Copy(values[i], v_row_hbm, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(v_row_hbm);
    if (i == 0) {
      v_sram_stack = v_row_hbm;
    } else {
      Matrix *new_v_stack = matrix_memory_allocator.Allocate("v_stack_sram");
      gpu_sim.Concat(v_sram_stack, v_row_hbm, new_v_stack, /*axis=*/0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_sram_stack);
      gpu_sim.ReleaseMatrix(v_row_hbm);
      v_sram_stack = new_v_stack;
    }

    /* Copy current query to avoid mutating original, then move copy to SRAM */
    Matrix *q_copy = matrix_memory_allocator.Allocate("q_copy");
    gpu_sim.Copy(current_query, q_copy, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(q_copy);

    /* logits = Q * K^T (shape (i+1, i+1)) in SRAM */
    Matrix *logits = matrix_memory_allocator.Allocate("logits");
    gpu_sim.MatMul(q_copy, k_sram_t, logits);

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

    /* out = softmax(QK^T) * V (shape (i+1, d)) in SRAM */
    Matrix *out = matrix_memory_allocator.Allocate("out");
    gpu_sim.MatMul(attn, v_sram_stack, out);

    /* Release intermediates no longer needed */
    gpu_sim.ReleaseMatrix(logits);
    gpu_sim.ReleaseMatrix(attn);
    gpu_sim.ReleaseMatrix(q_copy);

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
