# CUDA-Accelerated Image Processing Using Gaussian Blur and Sobel Edge Detection

**CEPARCO Integrating Project: Project Proposal (Milestone #1)**

**Group members:** Matthew Fabregas, Kane Ibanez, Bowen Li, Jat Teoxon

## Abstract

Our project is a program that finds the edges in a picture. It turns the image to grayscale, blurs it with a Gaussian blur to get rid of noise, and then uses a Sobel filter to mark where the edges are. Every pixel gets the same kind of math, which makes this a good fit for parallel computing. We will build two versions: a normal C++ version that runs on the CPU and handles one pixel at a time, and a CUDA version that runs on the GPU and handles many pixels at once. Both versions will run on the same images, and we will check that their outputs match. Then we will compare how long each one takes to see how much faster the GPU version is, and why.

## 1. What we are building

We are making an image processing program that detects edges using grayscale conversion, Gaussian blur, and Sobel edge detection. We will make two versions of it:

1. CPU version (C++): loops through the pixels one at a time. This is our baseline.
2. GPU version (CUDA): gives each pixel to its own GPU thread so thousands of pixels are processed together.

Then we compare the CPU and GPU versions for correctness and speed.

### Inputs

The input is a color image (JPG or PNG). Both versions use the same images, in four sizes:

| Test | Image size | Pixels |
|---|---:|---:|
| Small | 512 × 512 | 262,144 |
| Medium | 1024 × 1024 | 1,048,576 |
| Large | 1920 × 1080 | 2,073,600 |
| Extra large | 3840 × 2160 | 8,294,400 |

### Outputs

- A grayscale, a blurred, and an edge-detected version of each image
- The run time of the CPU version and the GPU version (GPU kernel time, and total GPU time including copying data)
- A check that the CPU and GPU outputs match
- The speedup, saved in a CSV file and shown in a graph

## 2. What we will parallelize and how

The image goes through three steps in order:

1. Grayscale: turn each color pixel into one brightness value.
2. Gaussian blur: replace each pixel with a weighted average of itself and its neighbors, which smooths out noise.
3. Sobel: look at how fast brightness changes around each pixel. A big change means an edge.

We will parallelize all three. In each step, a pixel's result does not depend on the other pixels' results, so they can all be done at the same time. This is data-level parallelism.

### Our plan for the GPU version

- One GPU thread handles one output pixel.
- Threads are grouped into 2D blocks, for example 16 × 16, and we launch enough blocks to cover the whole image.
- Each step is its own CUDA kernel. The image is copied to the GPU once, all three kernels run there, and the result is copied back once.
- Pixels at the border are missing some neighbors, so we will use the same border rule in the CPU and GPU versions. Otherwise the outputs will not match.
- If we have time after the basic version works, we will try to speed up the blur and Sobel kernels, for example with shared memory.

We are still learning CUDA, so some details of this plan may change as we build it.

### What will not be parallelized

- Loading and saving image files
- Allocating memory and copying data between the CPU and GPU
- Comparing the CPU and GPU outputs
- Timing, saving results, and making graphs

## 3. Related studies

GPU edge detection and image filtering have been studied before. Here are three studies close to our project.

Smailović (2019) compared CPU and GPU computing using the Sobel filter on images of different sizes, and found that the GPU speedup grows as the image gets bigger. Their CPU version was written in Python, which makes the CPU baseline slow.

Ibrahim et al. (2021) looked at the 2D convolution behind Gaussian blur and ran it on both a multi-core CPU and a GPU with CUDA.

Zuo et al. (2009) built a faster Sobel algorithm for CUDA GPUs using texture storage, multiple-point access, and symmetry computation. They reported a speedup of 122 times over a CPU version on 4096 × 4096 images.

### How ours is different

Each of these focuses on one filter, or on a heavily optimized kernel. We put grayscale, Gaussian blur, and Sobel together in one pipeline, compare a simple sequential version with a simple parallel version, and report both kernel time and total time including data transfers. Our CPU version is in C++ (compiled in Release mode), so the comparison is fairer. We are not trying to build the fastest possible program. We want to understand how a sequential algorithm becomes a parallel one and measure the difference.

## 4. Our setup

| Item | Details |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop GPU |
| CPU | [CPU model] |
| Operating system | Windows |
| Languages | C++ (CPU version), CUDA C++ (GPU version) |
| Tools | Visual Studio, NVIDIA CUDA Toolkit, NVIDIA driver |
| Code and submissions | GitHub |
| Graphs (optional) | Python with Matplotlib |

We will compile the CPU version in Release mode so the CPU is not slowed down by debug settings. No microcontroller or circuit is needed for this project.

## 5. How we will test and measure

- Both versions use the same algorithm and the same images.
- We compare CPU time with GPU kernel time, and also with the total GPU time including copying data to and from the GPU.
- Each test runs several times and we use the average. The GPU gets one warm-up run first.
- We time the GPU with CUDA events and the CPU with a standard C++ timer.
- To check correctness, we compare the CPU and GPU output pixel by pixel. A tiny difference from rounding is okay.
- Results go in a CSV file and into a graph of CPU vs. GPU time. We will find the actual speedup by testing it.

## 6. Who does what

| Member | Main job |
|---|---|
| Matthew Fabregas | Sequential CPU image processing |
| Kane Ibanez | CUDA parallel implementation |
| Bowen Li | Correctness testing and performance benchmarks |
| Jat Teoxon | GitHub, documentation, and presentation |

Everyone will help test, review each other's code, and explain the results.

## 7. Schedule and mid-project milestone

| Week | What we will have done |
|---|---|
| Week 6 | Milestone #1: this proposal and the 5–8 minute video pitch |
| Week 10 | Milestone #2 (mid-project demo): the CPU version works and is tested on our sample images. A first working GPU version of grayscale, blur, and Sobel runs, and its output matches the CPU output. We have our first CPU vs. GPU timings. |
| Week 13 | Milestone #3 (final submission and demo): improved GPU version, benchmarks for all four image sizes, speedup graph, correctness tests, write-up of the parallel algorithms, and the finished GitHub repo |

## 8. GitHub project structure

```text
CEPARCO-CUDA-Edge-Detection/
├── README.md
├── proposal/
│   ├── project_proposal.pdf
│   └── video_pitch_link.md
├── src/
│   ├── main.cpp
│   ├── cpu_edge.cpp
│   ├── cuda_edge.cu
│   └── image_utils.h
├── images/
│   ├── input/
│   └── output/
├── benchmarks/
│   ├── results.csv
│   └── speedup_graph.png
└── docs/
    └── final_report.pdf
```

## References

Ibrahim, N. M., ElFarag, A. A., & Kadry, R. (2021). Gaussian blur through parallel computing. In Proceedings of the International Conference on Image Processing and Vision Engineering (pp. 175–179). SCITEPRESS. https://doi.org/10.5220/0010513301750179

NVIDIA Corporation. (2026). CUDA C++ programming guide (Version 13.3). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Smailović, A. (2019). Time performance comparison between GPU and CPU computing by implementing an edge-detection algorithm using Sobel filter. Bulletin of the Transilvania University of Brașov, Series I: Engineering Sciences, 12(1), 49–55. https://doi.org/10.31926/but.ens.2019.12.61.7

Zuo, H., Zhang, Q., Xu, Y., & Zhao, R. (2009). Fast Sobel edge detection algorithm based on GPU. Opto-Electronic Engineering, 36(1), 8. https://m.researching.cn/articles/OJb93f548102bc66f6
