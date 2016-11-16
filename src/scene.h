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


struct Texture
{
	Texture() : data(NULL), width(0), height(0), depth(0) {}

	float* data;

	int width;
	int height;
	int depth;
};


struct Material
{
	Material() 
	{	
		color = Vec3(0.82f, 0.67f, 0.16f);
		emission = Vec3(0.0f);
		absorption = Vec3(0.0);

		// when eta is zero the index of refraction will be inferred from the specular component
		eta = 0.0f;

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
		bump = 0.0f;
		bumpTile = 10.0f;

	}

	CUDA_CALLABLE inline float GetIndexOfRefraction() const
	{
		if (eta == 0.0f)
			return 2.0f/(1.0f-sqrtf(0.08*specular)) - 1.0f;
		else
			return eta;
	}

	Vec3 emission;
	Vec3 color;
	Vec3 absorption;

	float eta;
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

	Texture bumpMap;
	float bump;
	Vec3 bumpTile;
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
	Vec3 horizon;
	Vec3 zenith;

	Probe probe;

	CUDA_CALLABLE Vec3 Eval(const Vec3& dir) const
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

	BVH bvh;

	void AddPrimitive(const Primitive& p)
	{
		primitives.push_back(p);
	}

	void Build();
};


