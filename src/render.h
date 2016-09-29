#pragma once

#include "maths.h"
#include "scene.h"
#include "intersection.h"

enum FilterType
{
	eFilterBox,
	eFilterGaussian
};

struct Filter
{
	CUDA_CALLABLE Filter(FilterType type=eFilterGaussian, float width=1.0f, float falloff=2.0f) : type(type), width(width), falloff(falloff)
	{
		if (type == eFilterGaussian)
			offset = expf(-falloff*width*width);
	}

	CUDA_CALLABLE float Eval(float x, float y) const
	{
		if (type == eFilterGaussian)
			return Gaussian(x)*Gaussian(y);
		else
			return 1.0f;
	}

	CUDA_CALLABLE float Gaussian(float x) const
	{
		return Max(0.0f, float(expf(-falloff*x*x)) - offset);
	}

	FilterType type;

	float width;
	float falloff;
	float offset;
};


enum RenderMode
{
	eNormals = 0,
	eComplexity = 1,
	ePathTrace =2
};


struct Options
{
	RenderMode mode;
	int width;
	int height;

	Filter filter;
	float exposure;
	float clamp;
	
	int maxDepth;
	int maxSamples;
};


struct Renderer
{
	virtual ~Renderer() {}

	virtual void Init(int width, int height) {}
	virtual void Render(const Camera& c, const Options& options, Color* output) = 0;

};

Renderer* CreateCpuRenderer(const Scene* s);
Renderer* CreateGpuRenderer(const Scene* s);
