#pragma once

#include "mesh.h"
#include "scene.h"

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

	
CUDA_CALLABLE inline Color ToneMap(const Color& c, float limit)
{
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;

	return c * 1.0f/(1.0f + luminance/limit);
}


struct CameraSampler
{
	CUDA_CALLABLE inline CameraSampler() {}

	CUDA_CALLABLE inline CameraSampler(
			const Mat44& cameraToWorld,
			float fov,
			float near,
			float far,
			int width,
			int height) : cameraToWorld(cameraToWorld)
	{
		Mat44 rasterToScreen( 2.0f / width, 0.0f, 0.0f, -1.0f,
								 0.0f, -2.0f / height, 0.0f, 1.0f,
								 0.0f,  0.0f, 1.0f, 1.0f,
								0.0f,  0.0f, 0.0f, 1.0f);

		float f = tanf(fov*0.5f);
		float aspect = float(width) / height;

		Mat44 screenToCamera(f*aspect, 0.0f, 0.0f, 0.0f,
										0.0f, f, 0.0f, 0.0f, 
										0.0f, 0.0f, -1.0f, 0.0f,
										0.0f, 0.0f, 0.0f, 1.0f);

		rasterToWorld = cameraToWorld*screenToCamera*rasterToScreen;
	}

	CUDA_CALLABLE inline void GenerateRay(float rasterX, float rasterY, Vec3& origin, Vec3& dir)
	{
		Vec3 p = TransformPoint(rasterToWorld, Vec3(rasterX, rasterY, 0.0f));

	    origin = Vec3(cameraToWorld.GetCol(3));
		dir = Normalize(p-origin);
	}

	Mat44 cameraToWorld;
	Mat44 rasterToWorld;
};

double GetSeconds();
