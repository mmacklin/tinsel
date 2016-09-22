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
	const Vec3* positions;
	const Vec3* normals;
	const int* indices;
	const BVHNode* nodes;

	int numVertices;
	int numIndices;
	int numNodes;

	int id;
};


inline MeshGeometry GeometryFromMesh(const Mesh* mesh)
{
	MeshGeometry geo;
	geo.positions = &mesh->positions[0];
	geo.normals = &mesh->normals[0];
	geo.indices = &mesh->indices[0];
	geo.nodes = &mesh->bvh.nodes[0];
	
	geo.numNodes = mesh->bvh.numNodes;
	geo.numIndices = mesh->indices.size();
	geo.numVertices = mesh->positions.size();

	return geo;
}

struct Primitive
{
	Primitive() : light(0) {}

	Transform transform;
	Transform lastTransform;
	
	Material material;
	GeometryType type;

	union
	{
		SphereGeometry sphere;
		PlaneGeometry plane;
		MeshGeometry mesh;
	};

	// if > 0 then primitive will be explicitly sampled
	int light;
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
