# CPU vs. GPU Terrain Generation: Measuring When CUDA Beats Sequential C++

**CEPARCO Integrating Project: Project Proposal (Revised Milestone #1)**

**Group members:** Matthew Fabregas, Kane Ibanez, Bowen Li, Jat Teoxon

## Abstract

Minecraft worlds are generated from a seed. For each (x, z) column, the generator uses layered noise (Perlin noise) to decide the height Y of the ground. A column's height depends only on its own coordinates, the seed, and a small shared lookup table, so every column can be computed independently. This is data-level parallelism.

We will build a simplified terrain generator with no caves, structures or villages, only the height of each column. We will make two versions and run both on the same PC: a sequential C++ version that computes one column at a time, and a CUDA version that gives each column its own GPU thread. We will check that both give the same heights, then compare their run times. We will test different world sizes, noise octaves, and three terrain types (plains, hilly, and mountains with a mountain-range look) to find out when the GPU becomes faster than the CPU, and why.

## 1. Project Description

Minecraft generates terrain on the CPU, one 16 × 16 chunk (256 columns) at a time, and mods such as C2ME spread this work over several CPU cores. A chunk is a tiny job for a GPU, because launching a kernel and copying data has a fixed cost. So the GPU may only help when a lot of terrain is generated at once.

We do not copy Minecraft's real generator, and our terrain will not match a real Minecraft world with the same seed. We build a Minecraft-style heightmap generator and measure it. Our research questions are:

1. **RQ1:** At what world size (16 × 16 up to 4096 × 4096 columns) does the GPU become faster than the CPU?
2. **RQ2:** How does the speedup change with the number of noise octaves and the terrain type?
3. **RQ3:** How do memory type (global, constant, shared) and block size (64, 256, 512, 1024 threads) change GPU time?
4. **RQ4:** How much of the GPU's total time is spent copying data?

We also check that the terrain is smooth, meaning no sudden cliffs between neighboring columns.

### Inputs

The program has no image or file input. All inputs are numbers given by the user:

| Input | Values |
|---|---|
| Seed | 64-bit integer. Default `3762183612631`. A second seed, `42`, is used in correctness tests. |
| World size (W × H columns) | See the size tables below. |
| World origin (x0, z0) | (0, 0) by default. Tests also use (-5000, 3000) and (1000000, -1000000) for negative and far-away coordinates. |
| Terrain type | Plains, hilly, or mountains (see section 2). |
| Octaves | Default per terrain type. For RQ2 we also try 1, 2, 4, 6 and 8 octaves on the hilly type. |
| Block size | 16 × 16 (256 threads) by default. For RQ3 we also try 8 × 8, 32 × 16 and 32 × 32. |

**Benchmark sizes** (used for timing):

| Test | World size | Columns | Chunks (16 × 16) |
|---|---:|---:|---:|
| One chunk | 16 × 16 | 256 | 1 |
| Tiny | 64 × 64 | 4,096 | 16 |
| Small | 256 × 256 | 65,536 | 256 |
| Medium | 1024 × 1024 | 1,048,576 | 4,096 |
| Large | 4096 × 4096 | 16,777,216 | 65,536 |

**Edge-case sizes** (used for correctness and bounds checking). All benchmark sizes are multiples of 16, so they never leave a partly filled block. These sizes do:

| Size | Why |
|---|---|
| 1 × 1, 3 × 3 | Smaller than one block. A 1 × 1 world has no neighbors. |
| 2 × 100, 100 × 2 | Very thin: one side smaller than a block, the other very long. |
| 15 × 17, 17 × 5 | One side just under or over the block size; partly filled blocks. |
| 255 × 257 | Just under and just over a multiple of 16. |
| 1001 × 777 | Random non-multiples, not square. |
| 4097 × 4097 | One past the largest benchmark size. |

### Outputs

- A saved heightmap for each run, plus a grayscale picture of it
- A rotatable 3D terrain viewer (Python, using the saved heights shrunk to about 512 × 512), colored from blue (low) to red (high). It is not part of any timing.
- The run time of the CPU version and the GPU version (GPU kernel time, and total GPU time including copying data)
- A check that the CPU and GPU outputs match, and a smoothness report per terrain type
- The speedup, saved in a CSV file and shown in graphs for RQ1 to RQ4

## 2. What we will parallelize and how

| Step | CPU (sequential) | GPU (CUDA) |
|---|---|---|
| 1. Height of every column | Nested loop over all columns | One kernel, one thread per column |
| 2. Max, min and sum of heights | One loop over the height array | Reduction kernel (atomics, and a shared-memory version) |
| 3. Smoothness check | One loop comparing each column to its right and lower neighbor | One kernel, then the same reduction |

In step 1, each column's result does not depend on any other column, so they can all be done at the same time.

### Our plan for the GPU version

- **One thread per column.** In a 2D grid of blocks, each thread computes `x = blockIdx.x * blockDim.x + threadIdx.x` and `z = blockIdx.y * blockDim.y + threadIdx.y`. The grid has `ceil(W / bx) × ceil(H / by)` blocks. Each thread checks `if (x < W && z < H)` first, so threads in partly filled blocks do nothing (the edge-case sizes test this).
- **Same function on CPU and GPU.** The height function is written once with `__host__ __device__`, so both versions run the same math.
- **Permutation table.** Built once on the CPU from the seed (512 bytes) and copied to the GPU. We compare three places for it: global memory, constant memory and shared memory. Constant memory is fast when all threads read the same address, but noise lookups depend on the coordinates, so we treat it as something to test, not as a guaranteed speedup.
- **Block size.** We test 64, 256, 512 and 1024 threads per block (8 × 8, 16 × 16, 32 × 16, 32 × 32).
- **Max, min and sum.** First version: `atomicMax`, `atomicMin` and `atomicAdd` on one value. Second version: each block reduces in shared memory with `__syncthreads()`, then adds its result with one atomic. Our lecture notes that many threads using atomics on one location slows things down, so we expect the second version to be faster and will test it.
- **Data transfer.** We compare explicit copies (`cudaMalloc` and `cudaMemcpy`) with unified memory (`cudaMallocManaged`, with and without prefetch). Heights stay on the GPU for the reductions, and only the final numbers are copied back.
- We follow the CUDA thread, block and memory model described in the CUDA C++ Programming Guide (NVIDIA, n.d.) and our lecture.

### The terrain formulas

The CPU and GPU versions use the same formulas and numbers. Only the way they run is different: the CPU goes through the columns one at a time, and the GPU gives each column its own thread.

**A. Seed and permutation table.** The seed is a 64-bit number. We use it to start a pseudo-random generator, then use that to shuffle the numbers 0 to 255 into a lookup table (the permutation table). The same seed always gives the same table. The table is built once on the CPU and copied to the GPU, so both versions use the identical table. We repeat it to 512 entries so lookups don't need to wrap around.

```text
rng = mt19937_64(seed)
p = [0, 1, 2, ..., 255]
for i = 255 down to 1:  j = rng() mod (i + 1);  swap(p[i], p[j])
perm[i] = p[i mod 256] for i = 0..511
```

We use a Fisher-Yates shuffle. We start from the end of the list and swap each number with a randomly chosen number from the part of the list before it. The random numbers come from `mt19937_64`, a standard C++ generator that gives the same numbers for the same seed on any computer. This way, if other group members run the code, they get the exact same terrain, so our results can be compared fairly. 

**B. 2D Perlin noise (improved version, Perlin 2002).**

```text
fade(t)       = 6t^5 - 15t^4 + 10t^3
lerp(t, a, b) = a + t * (b - a)
grad(h, x, z), using h mod 8:
  0: x     1: -x     2: z     3: -z
  4: x+z   5: -x+z   6: x-z   7: -x-z

noise(x, z):
  xi = floor(x); zi = floor(z)
  X = xi AND 255; Z = zi AND 255
  xf = x - xi; zf = z - zi
  u = fade(xf); v = fade(zf)
  aa = perm[perm[X] + Z]       ba = perm[perm[X+1] + Z]
  ab = perm[perm[X] + Z + 1]   bb = perm[perm[X+1] + Z + 1]
  return lerp(v, lerp(u, grad(aa, xf, zf),   grad(ba, xf-1, zf)),
                 lerp(u, grad(ab, xf, zf-1), grad(bb, xf-1, zf-1)))
```

The result is a smooth value between about -1 and 1. Nearby inputs give nearby outputs, so neighboring columns get similar heights.

**C. Octaves and terrain height.** We add several noise layers (octaves). Each octave has double the frequency (lacunarity 2.0) and half the strength (persistence 0.5) of the previous one. Mountains use **ridged** noise: the value of each octave is flipped so that the lines where noise is near 0 become sharp ridges. With a low base frequency, this gives long mountain ranges instead of random spikes.

```text
fbm(xw, zw):
  sum = 0; amp = 1; freq = f0
  for o = 0 .. N-1:
    n = noise(xw*freq + 17.37*o, zw*freq + 31.91*o)
    if ridged: n = 1 - 2 * abs(n)
    sum += amp * n
    freq = freq * 2.0; amp = amp * 0.5
  return sum * invNorm        invNorm = 1 / (2 * (1 - 0.5^N))

xw = x0 + x;  zw = z0 + z
h  = clamp( floor( B + A * fbm(float(xw), float(zw)) ), -64, 319 )
```

The height range -64 to 319 is the build range of recent Minecraft versions, used as a reference. The three terrain types only differ in these numbers (B = base height, A = amplitude in blocks, f0 = base frequency, N = octaves):

| Type | Noise | B | A | f0 | N | Height range (B ± A) | Smoothness limit |
|---|---|---:|---:|---:|---:|---|---|
| Plains | normal | 64 | 8 | 0.004 | 2 | 56 to 72 | ≤ 2 blocks |
| Hilly | normal | 72 | 24 | 0.008 | 4 | 48 to 96 | ≤ 4 blocks |
| Mountains | ridged | 100 | 90 | 0.004 | 6 | 10 to 190 | ≤ 8 blocks |

Smoothness limit means the largest height difference between two neighboring columns. These limits are our targets. If a terrain type goes over its limit during testing, we adjust its numbers and report it.

**D. Precision, rounding and clamping.**

- All noise math uses 32-bit `float` on both CPU and GPU. The table is `uint8_t`, heights are `int`, and sums are 64-bit integers.
- Rounding is `floorf` (round down), then convert to `int`. Heights are clamped to [-64, 319], and we report how many columns were clamped (expected 0).
- For the grayscale picture only, a height h is scaled to 0 to 255 using `round(255 * (h - hmin) / (hmax - hmin))`. This is for display and is not part of the comparison.
- The CPU is compiled in Release mode and the GPU with `nvcc -O3 --fmad=false`. We turn off fast-math and fused multiply-add on both sides. These are shortcuts that can round numbers slightly differently on the CPU and GPU, and we want the same results on both.

**E. Statistics.** We compute `hmax`, `hmin`, `hsum` and the average height. For smoothness, each column gets `d = max(|h(x,z) - h(x+1,z)|, |h(x,z) - h(x,z+1)|)` using only neighbors that exist (a 1 × 1 world has d = 0), and we report `dmax` and the average d.

### What will not be parallelized

- Building the permutation table from the seed (once, 256 values)
- Allocating memory and copying data between the CPU and GPU
- Saving files, the 3D viewer, comparing CPU and GPU results, timing statistics, the CSV file and graphs

**If we have time:** a multi-core (OpenMP) CPU version as an extra reference, grass/dirt/stone layers per column, a simple "snowy" color for high ground, and a 3D-printed model of a small piece of the terrain. These are not part of the main scope or the milestones. Caves, structures and biome blending are out of scope.

## 3. Related studies

Green (2005) already ran improved Perlin noise on a GPU using shaders, and it matched the CPU version, so GPU noise is not new.

For Minecraft, the C2ME mod speeds up chunk generation by using more CPU cores (C2ME, n.d.).

Patel (2015) is a tutorial that uses the same octave and ridged-noise approach as ours, and notes that noise alone has no large-scale features like rivers.

### How ours is different

The existing GPU noise work targets shading and texturing in graphics APIs, and C2ME targets multi-core CPUs. Our project uses CUDA to generate a whole heightmap and focuses on measuring: the world size where the GPU wins, the effect of work per column, memory type and block size, and the cost of copying data. We report both kernel time and total time including transfers, and we use the same PC for the CPU and GPU. We do not claim a new terrain algorithm.

## 4. Our setup

| Item | Details |
|---|---|
| CPU | Intel Core i5-8400 (6 cores, 6 threads) |
| GPU | NVIDIA GeForce GTX 1660 Super (Turing, compute capability 7.5, 22 SMs, 1,408 CUDA cores, 6 GB) |
| RAM and OS | 16 GB, Windows 11 |
| Software | Visual Studio and CUDA Toolkit (versions to be added after installation), NVIDIA driver 617.42 |
| Languages | C++ (CPU version), CUDA C++ (GPU version), Python (viewer and graphs only) |

All results come from this one PC, so the CPU and GPU are compared on the same machine. Google Colab is not used for any reported result. If we ever test on another machine, its results go in a separate table.

## 5. How we will test and measure

- Both versions use the same seed, terrain type and world size, in the same session.
- We time the CPU with `std::chrono` and the GPU with CUDA events. We report kernel time and total time (copy to GPU, kernels, copy back).
- Each test has one warm-up run and then 10 timed runs. We report the average, minimum and standard deviation.
- Speedup = CPU average time / GPU average time. We always state whether it uses kernel time or total time.

**Correctness checks** (run on every edge-case size, both seeds, and all three origins):

| Check | Pass criterion |
|---|---|
| Heights, CPU vs GPU | Largest difference ≤ 1 block and at most 0.01% of columns different. We expect 0 and will report the real numbers. |
| Max, min, sum, smoothness (GPU vs CPU on the same heights) | Exactly equal (integers) |
| Two GPU runs, same input | Identical heights |
| Smoothness | `dmax` of each terrain type ≤ its limit in the table |

## 6. Schedule and mid-project milestone

| Week | What we will have done |
|---|---|
| Week 6 | Milestone #1: this proposal |
| Week 10 | Milestone #2 (mid-project demo): the sequential CPU version is finished following section 2, and its heightmap picture looks like terrain. The first CUDA kernel (global memory) runs and passes the correctness checks on all edge-case sizes. We have our first CPU vs. GPU timings for at least three world sizes, and a first version of the 3D viewer. |
| Week 13 | Milestone #3 (final submission and demo): all three terrain types and all five sizes, the constant and shared memory versions, the octave and block-size tests, reductions and the smoothness check, full correctness tests, results CSV and graphs, write-up of the parallel algorithms, and the finished GitHub repo |

## References

Concurrent Chunk Management Engine (C2ME). (n.d.). CurseForge. https://www.curseforge.com/minecraft/mc-mods/c2me

Green, S. (2005). Implementing improved Perlin noise. In M. Pharr (Ed.), GPU Gems 2 (pp. 409–416). Addison-Wesley. https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter26.html

NVIDIA Corporation. (n.d.). CUDA C++ programming guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Patel, A. J. (2015). Making maps with noise functions. Red Blob Games. https://www.redblobgames.com/maps/terrain-from-noise/

Perlin, K. (2002). Improving noise. ACM Transactions on Graphics, 21(3), 681–682. https://doi.org/10.1145/566654.566636

