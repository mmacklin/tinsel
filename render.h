#pragma once

#include "maths.h"
#include "scene.h"

struct Camera
{

	Camera(Mat44 cameraToWorld,
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

	Mat44 cameraToWorld;
	Mat44 rasterToWorld;
};


enum RenderMode
{
	eNormals,
	eComplexity,
	ePathTrace
};

struct Renderer
{
	virtual ~Renderer() {}

	virtual void Render(Camera* c, Color* output, int width, int height, int samplesPerPixel, RenderMode mode) = 0;

};

Renderer* CreateCpuRenderer(const Scene* s);
Renderer* CreateGpuRenderer(const Scene* s);
