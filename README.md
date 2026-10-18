# CUDA-Accelerated Image Processing Using Gaussian Blur and Sobel Edge Detection

**CEPARCO Integrating Project: Project Proposal (Milestone #1)**

**Group members:** Matthew Fabregas, Kane Ibanez, Bowen Li, Jat Teoxon

## Abstract

This project develops an image processing program that detects edges in digital images using Gaussian blur and Sobel edge detection. Edge detection applies the same calculation to every pixel in an image, which makes it a good fit for parallel computing. The group will build two versions of the program: a sequential version in C++ that runs on the CPU, and a parallel version in NVIDIA CUDA that runs on the GPU. Both versions will process the same images, and their outputs will be compared to check that the results are correct. The group will then measure and compare their execution times to see how much faster the parallel version is. The project aims to show how data-level parallelism on a GPU can improve the performance of image processing tasks.

## 1. Project Description

The project is an image processing application that detects edges in digital images using grayscale conversion, Gaussian blur, and Sobel edge detection. The group will implement two versions of the program:

1. **Sequential CPU version (C++):** processes image pixels one at a time using ordinary loops.
2. **Parallel GPU version (CUDA):** processes image pixels using thousands of GPU threads under the SIMT model.

The group will compare the correctness, execution time, and performance of both versions.

### 1.1 Inputs

The program takes digital color images (JPG or PNG) as input. The same images are used for both the CPU and GPU versions, at four sizes:

| Test        | Image size  | Pixels    |
| ----------- | ----------- | --------- |
| Small       | 512 × 512   | 262,144   |
| Medium      | 1024 × 1024 | 1,048,576 |
| Large       | 1920 × 1080 | 2,073,600 |
| Extra large | 3840 × 2160 | 8,294,400 |

### 1.2 Proposed Process

The image goes through three steps, one after the other:

1. **Grayscale conversion:** each color pixel is converted to a single brightness value.
2. **Gaussian blur:** each pixel is replaced by a weighted average of itself and its neighbors, which reduces noise.
3. **Sobel edge detection:** each pixel's neighborhood is used to estimate how quickly brightness changes, and pixels with a strong change are marked as edges.

#### Processes to be parallelized (SIMT, GPU/CUDA)

All three steps will be parallelized. In each step, the result for one pixel does not depend on the results of other pixels, so many pixels can be processed at the same time. This is data-level parallelism.

#### Plan for parallelization

In the sequential version, nested loops visit each pixel one at a time. In the CUDA version, the work is divided as follows:

- Each GPU thread computes the output for one pixel.
- Threads are grouped into two-dimensional blocks (for example, 16 × 16 threads), and enough blocks are launched to cover the whole image.
- Each of the three steps is a separate GPU function (kernel). The intermediate results stay in GPU memory, so the image is copied to the GPU once and the final result is copied back once.
- Pixels at the image border, which have missing neighbors, are handled the same way in the CPU and GPU versions so that their outputs can be compared fairly.
- If time allows, the group will try to reduce slow global-memory reads in the blur and Sobel steps, for example by using shared memory. Because the group is still learning CUDA, the design details will be refined during development.

#### Processes that will not be parallelized

- Loading and saving image files
- Allocating memory and copying data between the CPU and GPU
- Comparing the CPU and GPU outputs
- Timing, recording results, and creating graphs

### 1.3 Outputs

- A grayscale, a blurred, and an edge-detected version of each input image
- Execution times of the CPU version and the GPU version (GPU kernel time, and total GPU time including data transfers)
- A correctness check comparing the CPU and GPU outputs
- Speedup results saved as a CSV file and shown in a performance graph

## 2. Project Objectives

By the end of the project, the group will:

- Implement grayscale conversion, Gaussian blur, and Sobel edge detection in C++.
- Develop equivalent parallel algorithms using NVIDIA CUDA.
- Compare CPU and GPU outputs to verify correctness.
- Measure CPU execution time and GPU execution time.
- Calculate speedup and analyze GPU performance.
- Explain how parallel processing improves computational efficiency.

## 3. Related Studies

Edge detection and image filtering on GPUs have been studied before. The following works are closely related to this project.

1. **Sobel edge detection on CPU and GPU.** The paper "GPU and CPU Computing by Implementing an Edge-Detection Algorithm Using Sobel Filter" compares CPU and GPU computing using the Sobel filter, and reports that the GPU speedup increases as the image size increases [1].
2. **Gaussian blur with CUDA.** "Gaussian Blur through Parallel Computing" presents the 2D convolution used in Gaussian blur and runs it on both a multi-core CPU and a GPU with CUDA [2].
3. **Optimized GPU Sobel.** Zuo et al. present a fast Sobel algorithm for CUDA GPUs that uses texture storage, multiple-point access, and symmetry computation, and report a 122 times speedup over a CPU implementation on 4096 × 4096 images [3].

**How this project differs.** These works each focus on a single filter, or on a highly optimized kernel. This project combines grayscale conversion, Gaussian blur, and Sobel edge detection into one pipeline, compares a straightforward sequential version with a straightforward parallel version, and reports both GPU kernel time and total time including data transfers. The goal is to understand how a sequential algorithm is converted into a parallel one and to measure the difference, not to build the fastest possible implementation.

## 4. Target Implementation Platform

| Item              | Details                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | NVIDIA RTX 4050 Laptop GPU                              |
| CPU               | [CPU model]                                             |
| Operating system  | Windows                                                 |
| Languages         | C++ (CPU version), CUDA C++ (GPU version)               |
| Tools             | Visual Studio (C++), NVIDIA CUDA Toolkit, NVIDIA driver |
| Source control    | GitHub                                                  |
| Graphs (optional) | Python with Matplotlib                                  |

The CPU version will be compiled in Release mode (optimizations on) so that the comparison with the GPU is fair. No microcontroller or physical circuit is required.

## 5. Performance Evaluation Plan

- The same algorithm and the same input images are used for the CPU and GPU versions.
- CPU processing time is compared with GPU kernel time, and also with total GPU time including data transfers.
- Each test is run several times and the average is reported. A warm-up run is done first on the GPU.
- GPU time is measured with CUDA events and CPU time with a standard C++ timer.
- Correctness is checked by comparing CPU and GPU output pixels, allowing a very small difference caused by rounding.
- Results are saved to a CSV file and plotted as a CPU vs. GPU execution time graph. The speedup will be determined experimentally.

## 6. Group Responsibilities

| Member           | Main responsibility                                 |
| ---------------- | --------------------------------------------------- |
| Matthew Fabregas | Sequential CPU image processing                     |
| Kane Ibanez      | CUDA parallel implementation                        |
| Bowen Li         | Correctness testing and performance benchmarks      |
| Jat Teoxon       | GitHub integration, documentation, and presentation |

All members will participate in testing, reviewing the code, and explaining the results.

## 7. Project Schedule and Mid-Project Milestones

| Week    | Milestone                                                                                                                                                                                                                                                                     |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Week 6  | **Milestone #1:** project proposal document and 5-8 minute video pitch                                                                                                                                                                                                        |
| Week 10 | **Milestone #2 (mid-project update demo):** the sequential CPU version works and is tested on the sample images; a first working GPU version of grayscale, Gaussian blur, and Sobel runs and its output matches the CPU output; first CPU vs. GPU timing results are recorded |
| Week 13 | **Milestone #3 (final submission and demo):** improved GPU version, complete benchmarks at all four image sizes, speedup graph, correctness testing, documentation of the parallel algorithms, and the complete GitHub repository                                             |

## 8. GitHub Project Structure

The following folder structure is planned for development. Files will be added as the project progresses.

```text
CEPARCO-CUDA-Edge-Detection/
  README.md
  proposal/
    project_proposal.pdf
    video_pitch_link.md
  src/
    main.cpp
    cpu_edge.cpp
    cuda_edge.cu
    image_utils.h
  images/
    input/
    output/
  benchmarks/
    results.csv
    speedup_graph.png
  docs/
    final_report.pdf
```

## References

[1] "GPU and CPU Computing by Implementing an Edge-Detection Algorithm Using Sobel Filter," Bulletin of the Transilvania University of Brasov, Series I. https://webbut.unitbv.ro/index.php/Series_I/article/download/1084/968/1897

[2] "Gaussian Blur through Parallel Computing," SCITEPRESS, 2021. https://www.scitepress.org/Papers/2021/105133/pdf/index.html

[3] H. Zuo, Q. Zhang, Y. Xu, and R. Zhao, "Fast Sobel Edge Detection Algorithm Based on GPU," Opto-Electronic Engineering, vol. 36, no. 1, p. 8, 2009. https://m.researching.cn/articles/OJb93f548102bc66f6

[4] NVIDIA, "CUDA C++ Programming Guide." https://docs.nvidia.com/cuda/cuda-c-programming-guide/
