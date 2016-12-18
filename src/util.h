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
    geo.cdf = &mesh->cdf[0];
    geo.area = mesh->area;
    
    geo.numNodes = mesh->bvh.numNodes;
    geo.numIndices = mesh->indices.size();
    geo.numVertices = mesh->positions.size();

	geo.id = (unsigned long)mesh;

    return geo;
}
	
CUDA_CALLABLE inline Color ToneMap(const Color& c, float limit)
{
	/*
	// reinhard
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;
	return c * 1.0f/(1.0f + luminance/limit);
	*/

	
	// filmic
	Vec3 texColor = Vec3(c);
	Vec3 x = Max(Vec3(0.0f),texColor-Vec3(0.004f));
	Vec3 retColor = (x*(Vec3(6.2f)*x+Vec3(.5f)))/(x*(Vec3(6.2f)*x+Vec3(1.7f))+Vec3(0.06));

	return SrgbToLinear(Color(retColor, 0.0f));
	

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

inline void MakeRelativePath(const char* filePath, const char* fileRelativePath, char* fullPath)
{
	// get base path of file
	const char* lastSlash = NULL;

	if (!lastSlash)
		lastSlash = strrchr(filePath, '\\');
	if (!lastSlash)
		lastSlash = strrchr(filePath, '/');

	int baseLength = 0;

	if (lastSlash)
	{
		baseLength = (lastSlash-filePath)+1;

		// copy base path (including slash to relative path)
		memcpy(fullPath, filePath, baseLength);
	}

	// append mesh filename
	strcpy(fullPath + baseLength, fileRelativePath);
}

double GetSeconds();
