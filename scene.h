#pragma once

#include "maths.h"
#include "mesh.h"
#include "bvh.h"
#include "skylight.h"

#include <vector>


struct Material
{
	Material() 
	{	
		color = SrgbToLinear(Color(0.82f, 0.67f, 0.16f));
		emission = Color(0.0f);
		metallic = 0.0;
		subsurface = 0.0f;
		specular = 0.5f;
		roughness = 0.5f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.5f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
	}

	Color emission;
	Color color;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
};

enum GeometryType
{
	eSphere,
	ePlane,
	eMesh
};

struct SphereGeometry
{
	float radius;
};

struct PlaneGeometry
{
	float plane[4];
};

struct MeshGeometry
{
	Mesh* mesh;
	BVH* bvh;
};

struct Primitive
{

	Primitive() : light(false) {}

	Mat44 transform;
	Mat44 lastTransform;
	
	Material material;
	GeometryType type;

	union
	{
		SphereGeometry sphere;
		PlaneGeometry plane;
		MeshGeometry mesh;
	};

	// if true then will participate in explicit light sampling
	bool light;
};


struct Light
{
	Light() : numSamples(0) {}
	
	// number of samples to take per path segment
	int numSamples;

	// shape to use for emission
	Primitive primitive;
};



struct Scene
{
	// contiguous buffer for the data
	typedef std::vector<Primitive> PrimitiveArray;
	typedef std::vector<Light> LightArray;

	PrimitiveArray primitives;
	
	Color sky;
	
	void AddPrimitive(const Primitive& p)
	{
		primitives.push_back(p);
	}	
};
