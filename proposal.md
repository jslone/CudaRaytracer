---
layout: page
title: CUDA Raytracer - Proposal
---

# Proposal
## Summary
We are going to implement adaptive thread-block assignment for a real-time GPU raytracer and profile its performance. We will create a BVH accelerated raytracer and implement at least two methods for optimizing thread-block assignment at runtime: hill climbing on the dimensions of blocks and spatial reassignment of rays to threads during traces. We will then profile the performance of each combination of these two methods.

## Background
Raytracing is a hot topic in computer graphics capable of producing much higher fidelity images than traditional rendering methods. The basic idea is that we can determine how a scene looks by simulating how photons bounce through the scene. Generally, this is simplified slightly by backtracking from the camera to find which photons could possibly reach the camera.

While parallelizing raytracing is rather trivial (simply parallelize across the rays), doing so efficiently on a GPU is more challenging due to ray divergence. We hope to minimize the costs of ray divergence by performing dynamic optimization on thread-block assignment used during tracing. This includes performing basic hill climbing on our initial dimensions at the start of a trace as well as leveraging our spatial data structure to reallocate rays among CUDA threads in order to optimize memory access and SIMD coherency on each block.

The idea is that by ensuring our blocks consists of threads tracing rays through a single spatial region ensures high SIMD coherency and our blocks can efficiently use shared memory since each ray will be checking collisions with the same objects.

## The Challenge
- Efficiently updating rays in the spatial data structure after a trace step
- Minimizing discrepancy between the times each block takes to execute
- Minimizing the latency between trace steps

## Resources
- GHC Cluster NVIDIA GPUs
- Personal NVIDIA GPU
- Starter Code
    - References
        - PBRT code
        - 15-462 starter code
    - Existing Code to be Used
        - SDL: Used for cross platform drawing and input
        - GLM: Used for vector and matrix operations
        - TinyXML: Used for loading scenes created in xml from multiple objects
        - tinyobjloader: Used for loading 3d objects to be rendered

## Goals and Deliverables
- Functional real-time GPU raytracer
- Report on analysis of different optimizations

### Grade Proposal
Grade | Project Status
------|-----------------------------------------------------------------------------------
A+    | Feature complete raytracer (caustics/volumetric scattering/sub-surface scattering)
A     | Profiling and analysis
A-    | Dynamic ray assignment
B     | Optimized data structure
C     | Create functional (unoptimized) parallel raytracer with hill climbing

## Platform Choice
We chose CUDA as our platform of choice as its massive parallelism with relatively low latency makes
it the best candidate for real time ray tracing, but there are still many interesting challenges with
utilizing it effectively which we would like to explore. This makes the assignment more interesting
to us than a CPU implementation, as a GPU implementation could in theory be much faster if usage was
optimal.

## Schedule
- Week of April 10
    - Scene parsing
- Week of April 17
    - Functional (unoptimized) raytracer
    - Hill climbing
-Week of April 24
    - Optimized data structure
    - Dynamic ray assignment
- Week of May 1
    - Final debugging
- Week of May 8
    - Profile/Analysis 

## Team
Jacob Slone [jslone] @jslone

Paul Aluri [paluri] @paulaluri