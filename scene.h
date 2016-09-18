#pragma once

#include "maths.h"
#include "mesh.h"
#include "bvh.h"
#include "skylight.h"

#include <vector>


struct Material
{
	Material() : reflectance(0.25), emission(0.0f), shininess(80.0f) {}

	Material(Color c, float s=80.0f) : reflectance(c), shininess(s) {}

	Color reflectance;
	Color emission;

	float shininess;

	/*
	Color baseColor .82 .67 .16
	float metallic 0 1 0
	float subsurface 0 1 0
	float specular 0 1 .5
	float roughness 0 1 .5
	float specularTint 0 1 0
	float anisotropic 0 1 0
	float sheen 0 1 0
	float sheenTint 0 1 .5
	float clearcoat 0 1 0
	float clearcoatGloss 0 1 1
	*/
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
	LightArray lights;
	
	float skyTheta;
	float skyPhi;
	float skyTurbidity;

	Scene() 
		: skyTheta(kPi/2.1f)
		, skyPhi(kPi/1.5f)
		, skyTurbidity(2.0f)
	{		
	}
	
	void AddPrimitive(const Primitive& p)
	{
		primitives.push_back(p);
	}

	void AddLight(const Light& l)
	{
		lights.push_back(l);
	}	

	void SetSkyParams(float theta, float phi, float turbidity)
	{
		skyTheta = theta;
		skyPhi = phi;
		skyTurbidity = turbidity;
	}
};
