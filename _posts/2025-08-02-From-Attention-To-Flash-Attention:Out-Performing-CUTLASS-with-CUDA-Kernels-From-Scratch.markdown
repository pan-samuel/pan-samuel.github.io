---
layout: post
title:  "Attention To Flash Attention: Out Performing SPDA with CUDA Kernels From Scratch"
date:   2025-08-02 01:04:05 -0400
categories: jekyll update
---

This post covers my final project from last semester to write an optimized Flash Attention kernel allgorithm. The goal is to compute attention as fast as possible written with pure CUDA C++. Inspired by  Siboehm's great blog and others, I'll detail iterative optiimized algorithms where the final kernel achieves a significant speedup and outperforms the PyTorch 2.7.0 scaled_dot_product_attention on fp32. 

| Kernel | Time (ms) | TFLOP/s | Performance vs. PyTorch SDPA |
|:---|---:|---:|---:|
| Manual Attention (Eager) | 51.26 | 4.02 | 24% |
| PyTorch SDPA (Flash Backend) | 12.06 | 17.09 | 100%|
| Kernel 1: Naive Row-per-Thread | 1122.45 | 0.37 | 2% |
| Kernel 2: Coalesced Memory | 223.02 | 1.85 | 11% |
| Kernel 3: Sequence Parallelism | 122.48 | 3.37 | 20% |
| Kernel 4: 2D Tiling & Register Caching | 37.28 | 11.06 | 65% |
| Kernel 5: Bank Conflict & Vectorization | 21.68 | 19.02 | 111% |
| Kernel 6: Occupancy Tuning | **20.51** | **20.10** | **118%** |

## Background
At the core of transformer-based models, the [attention mechanism](https://arxiv.org/pdf/1706.03762) has arguably become the most important deep learning algorithm in recent history. From LLMs to diffusion models, the algorithm is paramount to modern AI systems in its ability to parallelize computation and selectively focus on relevant parts of the input space through learned query-key-value interactions.

### Self-Attention Mechanism
The self-attention mechanism operates through a series of matrix multiplications and a softmax normalization. For a single attention head, the computation is expressed as:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $$Q$$, $$K$$, and $$V$$ represent the query, key, and value matrices respectively, and $$O$$ is the output attention matrix.

Let's define:
- $$N$$ = sequence length
- $$d$$ = head dimension  
- $$B$$ = batch size
- $$H$$ = number of heads

The query, key, and value tensors start with shape $$(B, N, H \times d)$$. In multi-head attention, we reshape and transpose the last dimension of the tensors to separate heads, resulting in shape $$(B, H, N, d)$$.

Since multi-head attention is fully parallelizable across both batch and head dimensions, we can focus our analysis on a single head processing a single batch, reducing our work with 2D matrices of shape $$(N \times d)$$ for $$Q$$, $$K$$, and $$V$$.

Going back to the attention equation, we can break it down into multiple stages:


$$S = QK^T \in \mathbb{R}^{N \times N}$$ \\
$$P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$$  \\
$$O = PV \in \mathbb{R}^{N \times d}$$ 

$$S$$ is called the raw attention scores, $$P$$ is the attention weights, and $$O$$ is the output matrix.

### Softmax
The softmax function converts and normalizes raw scores (logits) into a probability distribution. 

If you recall from your early ML or math classes, the softmax function is defined as:

$$O_i = \frac{e^{x_i}}{\sum_{k=1}^{N} e^{x_k}}$$

Here, we take an input vector $$X$$ with $$N$$ elements and return and output vector $$O$$ of the same shape. Each element in the input vector is transformed as shown above. Normally, we work we 2D matricies, so we apply the softmax rowise or across the last dimension. Note: the denominator term of the equation is often refered  as the normalization term or norm. 

However, when $$x_i$$ values are large, $$e^{x_i}$$ can easily overflow the precision of floating-point numbers. To fix this, the softmax equation is modified by subtracting each element $$x_i$$ by the maximum value of the vector, $$x_{max}$$, before the exponentiation. This shifts the values into a numerically stable range without changing the final output probabilities. The new equation is:

$$O_i = \frac{e^{(x_i - x_{max})}}{\sum_{k=1}^{N} e^{(x_k - x_{max})}}$$

Optimizing how we compute this stable softmax is important to the flash attention implementation and will be discussed later.

### Attention is Memory bound (1)
Before going into the Flash Attention algorithm, we need to understand why standard attention implementations are fundamentally limited — specifically by memory bandwidth. This problem comes from the intermediate attention score matrix $$S$$ ($$N \times N$$), which grows quadratically with sequence length.

If we consider a sequence length of $$N=4096$$ with precision of `bfloat16`, the attention score matrix alone requires $$4096^2 \times 2 = 33.6$$ MB of memory. This seems managable, but real systems process much longer sequences, which also scale quadratically. A sequence length of 16K would require over 500 MB and a 64K sequence would demand 8 GB just for the attention matrix computation. The memory challenge becomes worse when we consider that transformers invovle multiple attention heads, large batch sizes, and possible intermediate activations for gradient computation for backpropogation. 

Flash Attention allows us to compute the output without having to materialize the entire attention matrix at any one point. It reformulates this using tiling and "IO-aware" memory transactions without storing the full $$S$$ matrix.

## GPU Memory Hierarcy
Modern GPUs contain multiple levels of memory, each with different performance characteristics, which Flash Attention capitalizes to achieve its speedups.
#### High Bandwidth Memroy (HBM)
HBM is the GPU's main memory, analogous to RAM in a CPU system. This memory is the headline memory that GPUs promote when talking about VRAM. For example, a NVIDIA A100 GPU contains 40-80 GB of HBM with a bandwidth of approximately 1.5-2.0 TB/s. While "high bandwidth" may suggests fast access, HBM is actually the slowest memory tier that GPU kernels regularly interact with. 

The input tensors $$Q$$, $$K$$, and $$V$$ typically reside in HBM, along with the final output $$O$$

#### Static RAM (SRAM)
SRAM is the GPU's on-chip memory, including registers, L1/L2 caches, and shared memory accessible to CUDA thread blocks. A100 GPUs only have 192 KB of SRAM per [streaming multiprocessor](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor), but this memory operates at roughly 19 TB/s, or roughly 10 times faster than HBM.


### Attention is Memory Bound (2)
We've seen how standard attention implementations suffer from large intermediate activation memory requirements, but it also struggles due to its memory access patterns in regards to the GPU memory hierarchy. 

The core issue is how the algorithm handles $$Q, K, V$$ reads and intermediate activation ($$S, P$$) writes. Consider what happens during a typical attention computation: the algorithm first reads the entire $$Q$$ matrix $$(N \times d)$$ and $$K$$ matrix $$(N \times d)$$ from HBM into GPU compute units, performs the matrix multiplication to produce $$S$$ $$(N \times N)$$, then immediately writes this large intermediate result back to HBM. This write operation is necessary because the GPU's fast SRAM cannot accommodate the full $$(N \times N)$$ matrix (for a sequence length of 2048, this matrix requires 8.4 MB, far exceeding the typical 192 KB of SRAM available per streaming multiprocessor).

Then during the softmax computation, the algorithm reads the entire $$S$$ matrix back from HBM, applies the softmax operation row-wise (find row maximums, normalize, compute exponentials), then writes the resulting probability matrix $$P$$ bac k to HBM. Finally, the algorithm reads both $$P$$ and $$V$$ from HBM to compute the final matrix multiplication $$PV$$, writing the output $$O$$ back to HBM. In total, we are bounded by $$\Omega(Nd + N^2)$$ HBM accesses.

This pattern creates write-allocate pressure, where the GPU repeatedly allocates HBM bandwidth to store intermediate results that will be consumed almost immediately. Each of these large matrix transfers saturates the memory controllers, creating queuing delays that compound across the entire computation. When additional operations like masking or dropout are applied, these requires another full read-modify-write cycle through HBM.

The arithmetic intensity of this computation, or the ratio of floating-point operations to bytes transferred, is incredibly low. While the matrix multiplications themselves are compute-intensive, the softmax and element-wise operations perform smaller amounts of computation relative to the amount of data movement required. This creates a scenario where the GPU's thousands of compute cores sit idle, waiting for data to arrive from the memory subsystem that operates at a fraction of their potential throughput. 

## Flash Attention
### IO-Aware Algorithm Design
The term "IO-aware" refers to algorithm design that explicitly accounts for the cost of data movement between different memory tiers. 
FlashAttention exemplifies IO-aware design by minimizing HBM accesses while maximizing the computational work performed on data once it's loaded into SRAM. The algorithm orchestrates data movement to avoid the repeated reading and writing of intermediate results that standard attention implementations' suffer from by fusing the entire attention computation into a single kernel. Once **blocks** of $$Q$$, $$K$$, and $$V$$ are loaded into SRAM, the algorithm performs the matrix multiplication, applies the softmax (using the online technique), and accumulates the contribution to the output—all within the fast on-chip memory.

### Online Softmax
FlashAttention's other improvement uses a technique called online softmax, which computes the softmax function without requiring access to the complete input vector. 

Traditional softmax computation requires three passes over the input data:
1. First pass: Compute the maximum value, $$m$$, across all elements to ensure numerical stability. 
2. Second pass: Compute the normalization term by summing all exponential values $$l = \sum_{k=1}^{N} e^{(x_i - m)}$$
3. Third pass: Compute the output probabilities by dividing each exponential by the normalization term $$\frac{e^{(x_i - m)}}{l}$$

Online softmax reduces this to two passes by maintaining running statistics. As we iterate through the input vector, we simultaneously track the current maximum value and a running sum of the partial normalization term. This is because we can derive a recurrence relation that allows us to update our partial normalization term when we encounter a new maximum value. I won't go into the math this but [this](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) contain a good proof for the derivation.

The following is the recurrence relations:

$$m_i=max⁡(m_{i−1},x_i)$$ \\
$$d_i = d_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}$$

where $$m_i$$ represents the maximum value seen through position $$i$$, and $$d_i$$ represents the running normalization term. The exponential factor $$e^{m_{i-1} - m_i}$$ is a correction term, effectively "re-normalizing" all previously processed elements when we discover a new maximum. [Umar](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=475s) has a really great video on this section as well.

Then, the 2-pass algorithm over the input vector reduces to:
1. First pass: Update the $$m_i$$ and $$l_i$$ based on the current element $$x_i$$.
2. Second pass: Compute the probability for each element by calculating $$\frac{e^{(x_i - m)}}{l}$$

In Pytorch, the code looks like:

{% highlight python %}
def online_softmax(x):
  m = float('-inf')
  l = 0
  for x_i in x:
    m_i = max(m, x_i)
    l = l * math.exp(m - m_i) + math.exp(x_i - m_i)
    m = m_i
  o = []
  for x_i in x:
    o.append(math.exp(x_i - m) / l)
  return o
{% endhighlight %}

This online approach is crucial for Flash Attention because it allows us to compute softmax values for blocks of the attention matrix without needing to see the entire matrix at once.

## Tiling
FlashAttention's second improvement applies tiling to attention computation. Instead of materializing the complete $$(N \times N)$$ attention matrix, the algorithm processes the computation in smaller, manageable blocks that fit within the GPU's fast SRAM.

The algorithm divides the input matrices $$Q$$, $$K$$, and $$V$$ into blocks of size $$(B_r \times d)$$ and $$(B_c \times d)$$ respectively, where the $$B_r, B_c$$ are chosen to maximize SRAM utilization. For each block combination, FlashAttention loads the relevant data from HBM to SRAM, performs the attention computation entirely within the fast memory and accumulates the results using the online softmax technique.

This tiling approach decreases the memory access from $$O(Nd + N^2)$$ HBM reads and writes to $$O(\frac{N^2 d^2}{M})$$, where $$M$$ represents the SRAM size. Since we're processing all combinations of query and key-value blocks, we end up with approximately $$O(\frac{N^2 d^2}{M^2})$$ computational loops. Each loop moves $$O(M)$$ data, giving us our total data transfer of $$O(\frac{N^2 d^2}{M})$$. For typical values of the head dimension $$d$$ and SRAM size $$M$$, this is a big reduction in expensive memory operations.

#### Backwards pass
During the backward pass, standard attention implementations require access to the intermediate attention matrix $$P$$ to compute gradients with respect to $$Q$$, $$K$$, and $$V$$. Storing this matrix requires $$O(N^2)$$ memory and expensive HBM accesses.

Instead of storing the intermediate attention matrix, Flash Attention uses selective recomputation, where algorithm stores only the output $$O$$ and the softmax normalization statistics (the maximum and normalization terms from the online softmax computation). During the backward pass, these values are used for efficient recomputation of the attention matrix in small blocks that fit within SRAM.

This increases the total number of FLOPs by recomputing values that were previously stored. However, the reduction in HBM accesses more than compensates for the additional computation, resulting in faster wall-clock times.  

The recomputation strategy also reduces the peak memory footprint from $$O(N^2)$$ to $$O(N)$$ so we can process much longer sequences within the same memory constraints, translating into the ability to train larger models or process longer contexts without running out of GPU memory.

## Flash Attention Algorithm

Here is the Flash Attention algorithm:

![Alt text](/assets/FA-algo1.png "Flash Attention Algorithm")

#### Algorithm Setup (Lines 1-4)
The algorithm begins by determining optimal block sizes based on the available SRAM capacity. The algorithm then partitions the input matrices:

$$Q$$ matrix: Divided into $$T_r = \lceil \frac{N}{B_r} \rceil$$ row blocks of size $$B_r \times d$$ \\
$$K$$ and $$V$$ matrices: Divided into $$T_c = \lceil \frac{N}{B_c} \rceil$$ column blocks of size $$B_c \times d$$ \\
Output matrix $$O$$: Partitioned into $$T_r$$ blocks to match the Q partitioning

The algorithm also initializes the running statistics arrays $$\ell$$ (normalization terms) and $$m$$ (maximum values) and are initialized to zero and negative infinity respectively.

![Alt text](/assets/FA-algo-diagram.png "Flash Attention Algorithm")

#### Outer Loop: Processing K,V Blocks (Lines 5-6)
The algorithm's outer loop iterates through each K,V block pair $$(K_j, V_j)$$. For each iteration, the current K and V blocks are loaded from HBM into the GPU's fast SRAM which will be reused across multiple Q blocks in the inner loop.

#### Inner Loop: Processing Q Blocks (Lines 7-8)
For each $$K,V$$ block pair loaded in SRAM, the algorithm iterates through all $$Q$$ blocks. Each $$Q$$ block, $$Q_i$$, is loaded along with its corresponding output block $$O_i$$ and running statistics $$\ell_i, m_i$$.

#### Attention Computation (Lines 9-11)
This computation happens entirely in SRAM. 
Matrix Multiplication (Line 9): The algorithm computes the attention scores $$S_{ij} = Q_i K_j^T$$, producing a $$B_r \times B_c$$ block of the full attention matrix.
Block-wise Softmax (Line 10): Get the row-wise maximum $$\tilde{m}{ij}$$ and the exponential values $$\tilde{P}{ij}-\tilde{m}{ij}$$, along with the row sums $$\tilde{\ell}_{ij}$$. 
Online Statistics Update (Line 11): Update the running maximum $$m_i^{new}$$ and normalization term $$\$ell_i^{new}$$ using the recurrence relations. When a new maximum is discovered, all previously processed contributions are dynamically rescaled using the correction factor $$e^{m_i - m_i^{new}}$$.

#### Output Update and Storage (Lines 12-13)
The final step updates the output matrix using the newly computed attention block. The update formula in line 12 accounts for the rescaling needed when the running maximum changes: 

$$O_i \leftarrow \text{diag}(\ell_i^{new})^{-1}(\text{diag}(\ell_i)e^{m_i - m_i^{new}} O_i + e^{\tilde{m}_{ij} - m_i^{new}} \tilde{P}_{ij} V_j)$$ 

This formula ensures that previously accumulated contributions in $$O_i$$ are properly rescaled when new maximum values are discovered, while adding the contribution from the current block computation.
The algorithm then writes the updated output block and statistics back to HBM.

## Custom Kernels
Note: You should have a minimal understanding of GPU architecture and CUDA (what threads, blocks, etc) are for the following kernels to make sense. [Github](https://github.com/pan-samuel/flash-attention/tree/main/FA1-fp32)

### Kernel 1 - Minimal FA Implementation
Implementation 1 has inspiration from the [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) repository, though it differs in some ways. This kernel implements a single-pass attention algorithm that uses the core principles of Flash Attention while remaining simple enough to analyze and optimize systematically.

The kernel's execution model is straightforward: each thread processes one complete row of the output matrix. For multi-head attention, we launch one thread block per head per batch element. This requires launching batch_size × num_heads total blocks to process the entire input.
The launch configuration looks like:

{% highlight c++ %}
// Each thread block handles one batch-head combination
dim3 grid(BATCH_SIZE, NUM_HEAD);
// Each thread processes one output row
dim3 blockDim(B_r);
flashattention<<<grid, blockDim>>>(Q, K, V, O, m, l, bs, nh, N, dim);
{% endhighlight %}

This kernel is that we store the running statistics $$m_i$$ and $$l_i$$ in thread registers rather than shared memory. Since each thread processes its row sequentially, these values don't need to be shared across threads, and register storage eliminates the memory overhead of shared memory allocation.

Below is the pseudocode for the main loop. Try to take your time to understand how each part fits into the flash attention algorithm, we iterate over this later on:

{% highlight c++ %}
for (int j = 0; j < Tc; ++j) {  // (Line 5)
    // global memory to shared memory transfer (Line 6)
    // each thread loads a row
    K_smem[tid, :] = K[KV_tile_row + tid, :]; //K_smem (Bc x d)
    V_smem[tid, :] = V[KV_tile_row + tid, :]; //V_smem (Bc x d)
    
    // synchronize across the thread block in between
    // writing shared memory and reading shared memory
    __syncthreads();

    // (Line 7)
    for (int i = j; i < Tr; ++i) 
    {
      // global memory to shared memory transfer (Line 8)
        Q_smem[tid, :] = Q[Q_tile_row + tid, :]; //Q_smem (Br x d)
        
        // load previous m and l values into registers
        float m_i = m[lm_tile_row + tid]; // m and l are in global memory,  
        float l_i = l[lm_tile_row + tid]; // the shape of m and l is (N,1)

        // On chip computation of S = QK^T, row_m = m_ij(S) (Line 9)
        // We iterate over all of columns in K_smem for a (1 x d) x (d x Bc) = (1 x Bc) row a thread computes, which is what we want 
        for (int x = 0; x < Bc; ++x) {
          //dot product between the row the thread is resopnsible in Q and a column in K
            float sum = Q_smem[tid, :] * K_smem[x, :];
            S_ij_smem[tid, x] = sum * softmax_scale;
            //find the current maximum of S_ij (Line 10)
            m_ij = max(sum, m_ij)
        }

        // Stabilze pre attention matrix for softmax (Line 10)
        // P = exp(S - m_ij), l_ij = rowsum(P)
        S_ij_smem[tid, :] = expf(S_ij_smem[tid, :] - m_ij);
        l_ij = sum(S_ij_smem[tid, :]);
        
        // Find the new m and l (Line 11)
        float m_new = max(m_i, m_ij);
        float l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij;

        // Compute O and write to HBM (Line 12)
        // We sequentially compute all elements within a row (1 x d)
        for (int x = 0; x < d; ++x) {
            O_acc = S_ij_smem[tid, :] * V_smem[:, x];
            O[Q_tile_row + tid, x] = (1 / l_new) * (l_i * exp(m_i - m_new) *        
                                            O[Q_tile_row + tid, x] + exp(m_ij - m_new) * O_acc);
        }
        // Line 13
        m[lm_tile_row + tid] = m_new;
        l[lm_tile_row + tid] = l_new;
    }
    __syncthreads();
}
{% endhighlight %}

The performance in this kernel is limited by its memory access pattern. On a GPU, maximum memory bandwidth is achieved when consecutive threads in a **warp** (a group of 32 threads that execute in lockstep) access consecutive data elements in global memory. This is called a **coalesced access**. In our kernel, each thread loads an entire row of $$d$$ elements. This means thread `tid` and `tid+1` access memory locations that are $$d*sizeof(float)$$ bytes apart—a **strided access**. This forces the hardware to issue many separate, inefficient memory transactions

The result is a catastrophic drop in memory bandwidth. This kernel achieves a global memory throughput of only **13.12 GB/s**, 1.7% of the A6000's peak bandwidth of 768 GB/s. Furthermore, the profiler warns: `This kernel grid is too small to fill the available resources on this device`. With only `8 * 12 = 96` thread blocks launched, we aren't even giving the GPU enough work to do. This results in a runtime of 1122 ms, over 90 times slower than PyTorch's optimized version.


## Kernel 2 - Flash Attention Implemetation
While Kernel 1 establishes a functional baseline, it suffers from a severe underutilization of the GPU's parallelism and memory bandwidth. Our second kernel addresses this by reworking the thread-to-data mapping to achieve coalesced memory access.

The primary bottleneck in the first kernel is its strided memory access pattern. If we instead use coalesced access, it allows the memory controller to satisfy requests for an entire warp in a single transaction. 

Instead of one thread loading an entire $$d$$-element row, we parallelize the loading process. We assign a block of contiguous threads to load a contiguous block of data. For example, threads `tid`, `tid+1`, ..., `tid+31` now load 32 consecutive `float` values from a row in `K` into shared memory. This is a perfectly coalesced access. This cooperative loading is enabled by increasing the number of threads per block. 

This cooperative loading strategy is enabled by increasing the number of threads per block from $$Bc$$ to $$Br \times Bc$$. This larger thread block gives us the granularity needed to assign threads to individual data elements for loading, and then re-purpose them to compute a single element of the `S_ij` tile of size $$Br \times Bc$$.

Since the work for a single output row is now distributed across multiple threads, the running statistics `m_i` and `l_i` must be computed cooperatively. In this kernel, we perform the row-wise `max` and `sum` for the softmax normalization using a simple sequential loop within each thread over the `S_ij` tile row stored in shared memory.

{% highlight c++ %}
for (int i = j; i < Tr; ++i) 
    {
      //  global memory to shared memory coalesed transfer (Line 8)
        Q_smem[:, :] = Q[QO_tile_row:QO_tile_row+Br, :]; //Q_smem (Br x d)
        O_smem[:, :] = O[QO_tile_row:QO_tile_row+Br, :]; //O_smem (Br x d)
        
        // load previous m and l values into registers
        float m_i = m[lm_tile_row + tid]; // m and l are in global memory,  
        float l_i = l[lm_tile_row + tid]; // the shape of m and l is (N,1)

        // On chip computation of S_ij = QK^T, row_m = m_ij(S) (Line 9)
        // We compute one S_ij value with a row from Q (1Xd) and a column from K^T (dx1)
        S_ij_smem[tid_row, tid_col] = Q_smem[tid_row, :] * K_smem[:, tid_col] * scale;

        // Iterate over a row to find the max
        m_ij = rowmax(S_ij_smem[tid_row, :]);

        // All threads compute exp for their columns
        S_ij_smem[tid_row, tid_col] = __expf(S_ij_smem[tid_row, tid_col] - m_ij);

        // Compute l_ij (sum of exponentials) by iterativing over tid_row

        l_ij = rowsum(S_ij_smem[tid_row, :]);

        // Find the new m and l (Line 11)
        float m_new = max(m_i, m_ij);
        float l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij;

        // Compute O and write to HBM (Line 12)
        // We sequentially compute all elements within a row (1 x d)
        for (int x = 0; x < d; ++x) {
            O_acc = S_ij_smem[tid, :] * V_smem[:, x];
            O[Q_tile_row + tid, x] = (1 / l_new) * (l_i * exp(m_i - m_new) *        
                                            O[Q_tile_row + tid, x] + exp(m_ij - m_new) * O_acc);
        }
        // Line 13
        m[lm_tile_row + tid] = m_new;
        l[lm_tile_row + tid] = l_new;
    }
    __syncthreads();
{% endhighlight %}

This change yields a massive improvement. Global memory throughput jumps to **109.42 GB/s**, and the runtime drops to **223 ms**—a 5x speedup. However, it's still far from optimal.

A quick note on reduction strategy: I tried parallel reduction using warp shuffle primitives (`__shfl_down_sync`) as I assumed it would be faster. However, in this case, the iterative nature of the shuffle created a dependency chain, where each shuffle must wait for the previous one to complete. In the implementaiton that I used, the compiler is incredibly effective at optimizing the simple reduction loop, unrolling it and issuing wide, vectorized 128-bit shared memory loads (`LDS.U.128`). The sequential loop runs in 223 ms, while the shuffle-based version is slightly slower at 226 ms. You can see a comparison of the generated SASS here for the [shuffles primitives](https://godbolt.org/z/vGj3TvbaM) and for the [thread reduction](https://godbolt.org/z/TcsWqs3Eh).

## Kernel 3 - Parallelizing over Sequence Length

If you've been following the logic of flash attention closely, you might spot an obvious opportunity for even more parallelism. The idea comes in reexamining the loops and how data is reused. By focusing on what happens for a single output tile, we can see a more efficient structure.

The figure below shows the algorithm's execution flow. We can visualize the two nested loops with the outer loop iterating horizontally across the key/value blocks (indexed by $$j$$), and the inner loop that iterates vertically down the query/output blocks (indexed by $$i$$). For each block $$K_j$$ and $$V_j$$, the inner loop processes every block $$Q_i$$ to update every output block $$O_i$$.

![Alt text](/assets/FlashAttention_Sequence.png "Flash Attention Algorithm")

However, let's focus on how a single output tile, say $$O_1$$ (the tile at $$i=1$$), is computed. To get its final value, $$O_1$$ must accumulate results from its interaction with every key/value tile: $$Q_0$$, $$Q_1$$, $$Q_2$$/$$K_0$$, $$K_1$$, $$K_2$$, and so on. In our second kernel's loop structure, this means that the corresponding query tile $$Q_1$$ and the output tile $$O_1$$ are loaded from HBM into SRAM, updated, and written back to HBM in every single iteration of the outer $$j$$ loop. As you can see in the diagram, the block of $$Q$$ at $$i=1$$ is used when $$j=1$$, and then it's used again when $$j=2$$. This repeated HBM traffic for $$Q$$ and $$O$$ is a major source of inefficiency that we can eliminate.

So the computation for each output tile $$O_i$$ is independent of every other output tile $$O_{k \neq i}$$. This is also an embarrassingly parallel problem. We can redesign our kernel so that each thread block is responsible for computing one final output tile of size $$B_r \times d$$. This means we are now parallelizing over the sequence length dimension, $$N$$. A thread block will load its assigned query tile $$Q_i$$ into SRAM just once. Then, it will loop through all the necessary key/value tiles $$K_j$$ and $$V_j$$, performing the matrix multiplications, online softmax updates, and accumulating the result for $$O_i$$ entirely in fast on-chip memory. Only when the loop is finished and the output tile is complete is it written back to HBM.

  ![Alt text](/assets/FlashAttention_3.png "Flash Attention Algorithm")

This change has two benefits. First, we drastically reduce HBM traffic by eliminating the redundant loads of $$Q$$ and read-modify-write cycles for $$O$$. Second, we massively increase the number of independent thread blocks. Our previous kernel was limited by `batch_size * num_heads`. For a long sequence but small batch (a common scenario in LLMs), this leads to low GPU occupancy. For our A6000 with 84 SMs, Kernel 2 might only launch `8 * 12 = 96` blocks, barely enough to keep the machine busy. By parallelizing over sequence length, Kernel 3 launches $$(B \times H \times N / B_r) = (8 \times 12 \times 4096 / 32) = 12,288$$ blocks, ensuring the entire GPU is saturated with work.

This optimization improves performance to **122 ms** (3.37 TFLOP/s), a nearly 2x speedup over Kernel 2. However, we are still far from the GPU's peak performance.


We are heavily limited by the number of warps we can schedule on a multiprocessor, because on an A6000, max warps per multiprocessor is 48, which is 1536 threads, but because we launch our kernel with 32*32=1024 threads for each block, we are  only using a singular block for each multiprocessor. So this kernel is limited by the number of threads per block where, we cannot load more than one block per SM, giving us a final occupancy of 32 active warps / 48 max active warps = 66%. A 66% occupancy is not too bad, so this doesn’t explain why our kernel runs so slow.

## Kernel 4 - 2D Blocktiling for Increased Arithmatic Intensity

By parallelizing over the sequence length, we ensured the GPU's streaming multiprocessors (SMs) were saturated with independent blocks of work. The kernel is faster, but a look under the hood with a profiler reveals a new bottleneck. To understand this, consider that an A100 GPU has 108 SMs, each equipped with four warp schedulers that can issue instructions to 32 threads (a warp) in parallel per clock cycle. With over 13,000 threads potentially making progress simultaneously, the hardware is designed for massive parallelism. However, achieving this theoretical peak requires that warps remain in an "Active" state, receiving new instructions every cycle. In reality, warps frequently stall for various reasons—they may be waiting for memory operations to complete, dependencies to resolve, or computational resources to become available. The profiler's "Warp State Statistics" section reveals exactly where our kernel is spending its time, and for Kernel 3, the story is clear: warps are spending an enormous number of cycles stalled on MIO Throttle, indicating we're bottlenecked by memory throughput rather than computational capacity.

![Alt text](/assets/NCU_1.png "Nvidia Nsight Compute Profile")

This problem can be understood through arithmetic intensity and shared memory transactions. In Kernel 3, to compute a single output value, each thread performs roughly one multiply-add for every two values it loads from shared memory. 

{% highlight c++ %}
float s_ij_val = 0.0f;
for (int k = 0; k < d; k++) {
  s_ij_val += Q_smem[s_row * d + k] * K_smem[s_col * d + k]; // We load 2 elements from shared memory
}                                                           // to perform 1 FMA (Fused Multiply-Add) operation
{% endhighlight %}

The arithmetic units are constantly stalling, waiting for data to arrive from Shared Memory and we're bottlednecked by instruction dependency latency. This low ratio of math-to-memory-access chokes the SMEM pipeline for both the attention computation and output computation.

Kernel 4 introduces **2D micro-tiling** to solve this. Each thread is now responsible for computing a small `TM x TN` tile of the output (e.g., 4x4), not just a single scalar. The inner computation is moved from shared memory into thread registers. A thread loads a `1 x TM` fragment of Q and a `1 x TN` fragment of K into its private registers. It then performs an outer product on these register vectors. A single value from the Q fragment is reused `TN` times, and a value from the K fragment is reused `TM` times.

{% highlight c++ %}
for (int k = 0; k < d; ++k) {
  // Load slices into registers
  for (int mm = 0; mm < TM; ++mm) Q_reg[mm] = ...;
  for (int nn = 0; nn < TN; ++nn) K_reg[nn] = ...;

  // Perform outer product from registers
  for (int mm = 0; mm < TM; ++mm)
      for (int nn = 0; nn < TN; ++nn)
          acc[mm][nn] += Q_reg[mm] * K_reg[nn];
}
{% endhighlight %}

This change dramatically increases the arithmetic intensity at the shared memory level. For a `4x4` tile, we now perform $$TM \times TN=16$$ FMAs for every $$TM+TN=8$$ shared memory loads, a 4x improvement in the FMA-to-load ratio. Now, the number of shared memory load instructions (`LDS`) executed by the kernel decreases from $$2.9\times10^9$$ in Kernel 3 to just $$4,81\times10^8$$ in Kernel 4. The result is a transformative **3.3x speedup**, bringing the runtime down to **37.28 ms** (11.06 TFLOP/s).

![Alt text](/assets/FlashAttention_4.png "FA 4")

Of course, this is not a free. This technique increases register pressure, and making the tiles too large can lead to register spilling or new stall reasons as the scheduler handles more dependencies. The optimal tile size (TM=4, TN=4 was a good balance on my hardware) is a matter of empirical tuning. 

![Alt text](/assets/NCU_2.png "Nvidia Nsight Compute Profile")


## Kernel 5 - Small Changes
### Shared Memory Conflicts
With the MIO Throttle stall addressed, our kernel's performance tripled. Yet, it was still only about half as fast as the optimized PyTorch implementation. A new run through Nsight Compute shows the bottleneck has changed. The "Warp State Statistics" profile now shows that warps are no longer waiting on the MIO instruction queue; instead, they are stalled by Stall Short Scoreboard.

The NVIDIA documentation says:

"Warps was stalled waiting for a scoreboard dependency on MIO (memory input/output) operation... The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. ... Verify if there are shared memory operations and reduce bank conflicts, if applicable."

Also, the high L1/TEX cache throughput we were also seeing (around 95%) can be a secondary effect of bank conflicts, so the profiler is giving us instruction to investigate shared memory bank conflicts.

#### Shared Memory Banks
Shared memory is divided into 32 banks. Access is fastest when threads in a warp access different banks simultaneously. A bank conflict occurs when multiple threads try to access different addresses within the same bank, forcing the hardware to serialize the requests. When a warp issues a memory request, the hardware examines the address requested by each thread. If each thread accesses a different memory bank, the request can be serviced in a single transaction. This parallel access is the ideal, conflict-free scenario. Another conflict-free case is a broadcast, where all threads read the exact same address; the hardware reads the word once and broadcasts it to all threads, however bandwidth utilization is poor because only a small number of bytes are read. The problem arises when multiple threads attempt to access different addresses within the same memory bank. This is a bank conflict. The hardware cannot service these requests in parallel and must instead serialize them, processing them one after another. If all 32 threads hit different addresses in the same bank, a single memory operation effectively becomes 32 separate operations, and the shared memory bandwidth plummets by a factor of 32.

![Alt text](/assets/banks.png "Bank Accesses")

The figure (top) shows the most optimal parallel access pattern where each thread accesses one 32-bit word. The middle figure is also conflict free as each thread accesses a different bank. The bottom figure illustrates another irregular access pattern where several threads access the same bank. 

For an Ampere GPU like the A6000, each bank is 32-bits (4 bytes) wide. Successive 32-bit words are mapped to successive banks in a cyclical pattern. The mapping is a simple modulo operation:

`bank_index = (word_address) % 32`
where `word_address = byte_address / 4.`

This means bank 0 holds words 0, 32, 64, etc. Bank 1 holds words 1, 33, 65, etc. This layout is designed to make contiguous memory access fast.

![Alt text](/assets/bank_words.png "Bank Accesses")

Our row-major data layout combined with our column-major access pattern for the `K` and `V` matrices creates a worst-case scenario, where threads in a warp consistently hit the same banks.

### Padding
The solution is **padding**. By adding a small amount of padding to the end of each row in our shared memory arrays, we change the stride. This skews the layout, causing column elements that previously fell into the same bank to be distributed across different banks, eliminating the conflict.

In the figure, we see a layout analogous to our Kernel 4. The width of our data (5 elements) is a multiple of the number of banks (5). Notice how all elements in a given column (e.g., all the '0's) fall into the same memory bank (Bank 0). When our warp tries to read this column, all threads hit Bank 0 simultaneously, causing a massive, serialized conflict.

![Alt text](/assets/padding.png "Padding")

On the right, we've applied the fix. We've added a single column of padding (the light green squares) at the end of each row. This increases the stride of our array from 5 to 6. Now, look at what happens to the columns. The elements of the first column ('0') are no longer all in Bank 0. Because of the padding, they are now skewed across Bank 0, Bank 1, Bank 2, Bank 3, and Bank 4.

When the warp reads this column, each thread accesses a different bank. The conflict is gone, and the memory access can proceed in parallel at full speed. The padded memory's only role is to shift data elements so that data that originally resided in the same bank are distributed among banks. As a result, the total amount of useful shared memory each block requires increasing, decreasing the overall shared memory available to a thread block will decrease. However, in our case, we only add a singular extra element for each row, which only adds 32 * 4 bytes = 128 bytes of shared memory. 

### Vectorization
Our global-to-shared memory copies are coalesced but can be made even more efficient. Instead of loading one `float` at a time, we can use a 128-bit `float4` type to load four `float` values with a single instruction. Looking at the generated SASS (assembly), we can see the compiler replaces four 32-bit `LDG.E` instructions with a single, much more efficient 128-bit `LDG.E.128` instruction.

{% highlight c++ %}
__device__ void vectorized_load(...) {
    // A single instruction loads 4 floats (128 bits)
    float4 in4 = reinterpret_cast<const float4 *>(...)[0];
}
{% endhighlight %}


**SASS for standard load:**
{% highlight sass %}
LDG.E.CONSTANT R41, [R4.64+0x1100]
LDG.E.CONSTANT R43, [R4.64+0x1200]
LDG.E.CONSTANT R45, [R4.64+0x1300]
LDG.E.CONSTANT R47, [R4.64+0x1400]
{% endhighlight %}

**SASS for vectorized load:**
{% highlight sass %}
LDG.E.128.CONSTANT R44, [R22.64]
{% endhighlight %}

### More Tiling
We are no longer stalled, but we are underutilized. According to Nsight Compute, our kernel achieves an occupancy of only 12.5%.

Occupancy is the ratio of active warps on a Streaming Multiprocessor (SM) to the maximum number of warps that SM can support. While high occupancy doesn't guarantee high performance (as in kernel 3), extremely low occupancy is a red flag. It means the SM has very few independent warps to choose from to hide memory and instruction latencies. When one warp stalls waiting for data, the SM has a limited pool of other warps to switch to, leaving its execution units idle.

The root cause of our low occupancy is shared memory pressure. On my A6000, each SM has 100 KB of shared memory available, with a hardware limit of 48 KB per thread block. Our footprint is:

*   `Q_smem`: `Br * d` = 32 * 64 = 2048 floats
*   `K_smem`: `Bc * (d + 1)` = 32 * 65 = 2080 floats
*   `V_smem`: `Bc * d` = 32 * 64 = 2048 floats
*   `S_ij_smem`: `Br * (Bc + 1)` = 32 * 33 = 1056 floats

The total is 7,232 floats. At 4 bytes per float, this comes to 28,928 bytes, or roughly 28.25 KB. Including the CUDA runtime's overhead, each block consumes about 29.25 KB of shared memory. Given the 100 KB pool per SM, we can only launch 100 KB / 29.25 KB ≈ 3.4 blocks. Since blocks are indivisible, this means a maximum of 3 blocks can run concurrently on each SM. This is the direct cause of our 12.5% occupancy and the primary limiter of our performance. 

The problem is that our design requires the full d-dimension slices of Q, K, and V tiles to be resident in shared memory to perform the S_ij = QK^T computation. The solution is to break this dependency by tiling the computation along the head dimension, d. 

<!-- We introduce a new blocking parameter, Bk, and loop over the d dimension in D_TILES = d / Bk chunks. In this implementation, we set Bk=32, so we process the QK^T and PV matrix multiplications in two stages. So now K_smem and V_smem no longer need to hold the full d dimension, and K_smem size is reduce from Bc * (d + 1) to Bc * (Bk + 1) and V_smem is reducfed from Bc * d to Bc * Bk. -->

Now, the new shared memory is 5,184 floats, or 20,736 bytes (20.25 KB). With overhead, each block now consumes about 21.25 KB. This smaller footprint means we can now launch 100 KB / 21.25 KB ≈ 4.7, or 4 blocks per SM. This pushes our occupancy up to 16.7%, giving the SM scheduler more active warps to hide latency.

These optimizations combined boost performance to **19.0331ms** (21.66 TFLOP/s), finally surpassing the PyTorch reference implementation.
# Final Results and Reflections
Our final kernel surpasses the performance of PyTorch's default `scaled_dot_product_attention (spda)` backend, which is powered by a heavily engineered CUTLASS library.

![Alt text](/assets/benchmark_seq_len.png "Bank Accesses")

The chart above shows a consistent performance advantage for our hand-written kernel across a range of sequence lengths. For a sequence of 4096, our kernel achieves **20.10 TFLOP/s**, making it **1.18x faster** than the PyTorch baseline.

| SeqLen | PyTorch SDPA (ms) | Our Kernel (ms) | Speedup |
|:---|---:|---:|---:|
| 1024 | 1.43 | 1.28 | 1.12x |
| 2048 | 5.23 | 5.15 | 1.02x |
| 4096 | 21.15 | 20.40 | 1.04x |
| 8192 | 89.34 | 82.23 | 1.09x |
| 16384 | 383.17 | 332.63 | 1.15x |
| 32768 | 1611.22 | 1349.39 | 1.19x |

However, this kernel is tuned for a specific problem size and hardware. A production-grade library contains hundreds of kernel variations. At runtime, it dispatches the optimal one based on the input shapes, data types, and hardware capabilities. So this produces robust high performance across all scenarios, a feature our single kernel lacks.

## The Triton Pill

So is hand-writing CUDA the only path to peak performance? Not really. To put this effort into perspective, I also implemented the same algorithm using Triton, a Python-based language for writing GPU kernels.

Under the exact same conditions (N=4096, d=64, A6000), the Triton kernel executed in just **4.68 ms**.

-   **Our Hand-Tuned CUDA Kernel:** 19.03 ms
-   **Triton Kernel:** 4.68 ms

Under identical test conditions (N=4096, d=64, A6000), the Triton kernel executed in **4.68 ms**, a **4.38x speedup** over our most optimized CUDA kernel (20.51 ms).

This performance gap suggests that domain-specific compilers like Triton's can generate more optimal code than is practical to write manually in CUDA, where the compiler effectively abstracts tasks like instruction scheduling, pipeline synchronization, and register management, without sacrificing performance.

## Next

While this project is complete, the learning journey continues. The next steps are clear:

1.  **Half percision FA:** Re-implementing these concepts for `bfloat16` and using tensor cores on an A100 (still outdated ik) using matrix-multiply-accumulate (MMA) instructions, pipelining, etc.
2.  **Exploring CUTLASS and CuTe:**  Try to implement a library like implementation of FA-2 in CUTLASS and then use CuTe-DSL on FlashAttention-3 on an H100.

# Further Resources
- **University of Washington - CSE 599M:** [Lecture notes on FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) provide a great academic overview of the algorithm's mathematical foundations.
- **Siboehm's Blog:** [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) was a major inspiration for the iterative optimization approach in this post.
- **Minimal Flash Attention Implementation:** This [GitHub repository](https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu) provides a clean, understandable reference implementation in CUDA.
- **ELI5 Flash Attention:** A [Medium article by Aleksa Gordić](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) that explains the high-level concepts in a very accessible way.
- **Zhihu Article on Flash Attention:** A [detailed technical walkthrough](https://zhuanlan.zhihu.com/p/708867810) (in Chinese) that dives into the implementation details.
