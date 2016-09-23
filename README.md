Tinsel
======

A lightweight CPU/GPU path tracer focusing on speed and simplicity. Tinsel was
originally designed for rendering out physics based animations where turn around
time is more important than generality. It is designed to be easy to set up
animation sequences.

Features
--------

- Unbiased uni-directional path tracer
- Disney's principled BRDF with importance sampling of diffuse and specular lobes
- CPU or GPU tracing and shading with a persistent CUDA threads model
- Interactive OpenGL progressive mode
- Explicit area light sampling
- Motion blur
- Gaussian reconstruction filter
- Instanced triangle mesh primitives with affine transformations
- AABB tree with SAH and splitting
- Simple scene description format
- Windows / macOS / Linux support

Example Scene
-------------

The scene description in Tinsel is very simple, and loosely based off Arnold's .ass format,
here is an example:

```
# This is a comment

material gold
{
	color 1.0 0.71 0.29
	roughness 0.2
	metallic 1.0	
}

material plaster
{
	color 0.94 0.94 0.94
	roughness 0.5
	specular 0.1
}

material light
{
	emission 5.0 5.0 5.0
}

primitive
{
	type plane
	plane 0 1 0 0
	material plaster
}

primitive
{
	type sphere
	radius 0.5
	material light

	position 0.0 10.0 0.0
	rotation 0.0 0.0 0.0 1.0
	scale 1.0
}

primitive
{
	type mesh
	mesh octopus.obj
	material gold

	position 0.0 0.0 0.0
	rotation 0.0 0.0 0.0 1.0
	scale 2.0
}

```

Command Line
------------

Image:

```
tinsel -spp 100 scene.tin output.pfm
```

Animation (must have ffmpeg in path):

```
tinsel -spp 100 frame%d.tin output.mp4
```

Interactive:

```
tinsel -interactive scene.tin
```

Todo List
---------

[ ] Mesh affine transformation support
[ ] SAH and BVH splitting
[ ] Clean up mesh allocations
[ ] Command line interface
[ ] Scene sky parameters
[ ] Scene include files
[ ] Scene camera definition
[ ] Tone mapping
[ ] Bloom filter
[ ] Output formats
[ ] FFmpeg encoding
[ ] Reconstruction filter

Supported Platforms
-------------------

Tinsel ships with makefiles and Visual Studio projects for OSX and Windows respectively. Although not explicitly supported it should be relatively simple to build for Linux.

License
-------

Tinsel is licensed under the ZLib license, see LICENSE.txt.

Author
------

Miles Macklin - http://mmacklin.com