---
layout: page
title: Project Preview
---

# Overview
A real-time GPU raytracer created using CUDA and adaptive thread-block allocation and assignment of rays.

# Cornell Box Renders

## Diffuse Scene
![Diffuse Render](images/diffuse.PNG "Diffuse Render")

## Specular Scene
![Specular Render](images/specular.PNG "Specular Render")

## Specular Scene with Spotlight
![Spotlight Render](images/spot.PNG "Spotlight Render")

## Scene Converging
![Converging Render](images/converging.PNG "Converging Render")

~~~
git clone git@github.com:jslone/CudaRaytracer.git
cd CudaRayTracer
mkdir build && cd build
cmake ../src
make
~~~

# Running

~~~
./build/bin/application
~~~

# Dependencies

### cmake

### CUDA Toolkit