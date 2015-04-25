---
layout: page
title: CUDA Raytracer - Checkpoint
---

# Proposal

## Scene Parsing
We experienced quite a few challenges in getting to our desired checkpoint.  We found a C++ library "Assimp" which allowed us to easily load all sorts of standard 3D scene files.  While this is going to be a fantastic tool for allowing us to quickly test different scenes and also be a nice feature to allow our raytracer to display virtually any scene we wish, there was also quite a bit of overhead in getting the data from the library into formats and data structures that we wished to use (and copy over to our NVIDIA graphics card).  We did finally accomplish this even though we had to deal with some of pains of the library (having objects linked to each other by string keys, for example).

## Data Structures
We moved the mesh data (vertices, edges, faces) from the Assimp library's format into a CUDA host vector so we could easily flush it onto the device.

## Simple Raytracer
We then plugged in code from a simply raytracer kernel that Jacob had built for Graphics just to get a running, functional raytracer that we could then optimize and analyze with our data structures and improved algorithms. 
 

## Team
Jacob Slone [jslone] @jslone

Paul Aluri [paluri] @paulaluri
