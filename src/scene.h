#pragma once

#include "maths.h"
#include "mesh.h"
#include "bvh.h"
#include "skylight.h"
#include "probe.h"

#include <vector>

struct Camera
{
	Camera()
	{
		fov = DegToRad(45.0f);

		shutterStart = 0.0f;
		shutterEnd = 1.0f;
	}

	Vec3 position;
	Quat rotation;

	float fov;

	float shutterStart;
	float shutterEnd;

	// todo: lens options
};



struct Material
{
	Material() 
	{	
		color = SrgbToLinear(Color(0.82f, 0.67f, 0.16f));
		emission = Color(0.0f);
		absorption = Color(0.0);

		metallic = 0.0;
		subsurface = 0.0f;
		specular = 0.5f;
		roughness = 0.5f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.0f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
		transmission = 0.0f;
		
		// when eta is zero the index of refraction will be inferred from the specular component
		eta = 0.0f;	 
	}

	CUDA_CALLABLE inline float GetIndexOfRefraction() const
	{
		if (eta == 0.0f)
			return 2.0f/(1.0f-sqrtf(0.08*specular)) - 1.0f;
		else
			return eta;
	}

	Color emission;
	Color color;
	Color absorption;

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
	float transmission;	
	float eta;
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

	const Vec3* positions;
	const Vec3* normals;
	const int* indices;
	const BVHNode* nodes;
	const float* cdf;

	int numVertices;
	int numIndices;
	int numNodes;

	float area;

	unsigned long id;
};


struct Primitive
{
	Primitive() : lightSamples(0) {}

	// begin end transforms for the primitive
	Transform startTransform;	
	Transform endTransform;
	
	GeometryType type;

	union
	{
		SphereGeometry sphere;
		PlaneGeometry plane;
		MeshGeometry mesh;
	};

	Material material;

	// if > 0 then primitive will be explicitly sampled
	int lightSamples;
};

struct Sky
{
	Color horizon;
	Color zenith;

	Probe probe;

	CUDA_CALLABLE Color Eval(const Vec3& dir) const
	{
		if (probe.valid)
		{
			return ProbeEval(probe, ProbeDirToUV(dir));
		}
		else
		{
			return Lerp(horizon, zenith, sqrtf(Abs(dir.y)));
		}
	}

	// map
};

struct Scene
{
	// contiguous buffer for the data
	typedef std::vector<Primitive> PrimitiveArray;
	PrimitiveArray primitives;
	
	Sky sky;
	Camera camera;	

	void AddPrimitive(const Primitive& p)
	{
		primitives.push_back(p);
	}	
};


