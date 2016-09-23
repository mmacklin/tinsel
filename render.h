#pragma once

#include "maths.h"
#include "scene.h"
#include "intersection.h"

struct Camera
{
	CUDA_CALLABLE inline Camera() {}

	CUDA_CALLABLE inline Camera(Mat44 cameraToWorld,
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

CUDA_CALLABLE inline Color ToneMap(const Color& c)
{
	float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;

	return c * 1.0f/(1.0f + luminance);
}

CUDA_CALLABLE inline void GenerateRay(const Camera& camera, float rasterX, float rasterY, Vec3& origin, Vec3& dir)
{
	Vec3 p = TransformPoint(camera.rasterToWorld, Vec3(rasterX, rasterY, 0.0f));

    origin = Vec3(camera.cameraToWorld.GetCol(3));
	dir = Normalize(p-origin);
}

CUDA_CALLABLE inline void GenerateRayNoJitter(const Camera& camera, int rasterX, int rasterY, Vec3& origin, Vec3& dir)
{
	Vec3 p = TransformPoint(camera.rasterToWorld, Vec3(float(rasterX) + 0.5f, float(rasterY) + 0.5f, 0.0f));

    origin = Vec3(camera.cameraToWorld.GetCol(3));
	dir = Normalize(p-origin);
}

//-------------


CUDA_CALLABLE inline void Sample(const Primitive& p, float time, Vec3& pos, float& area, Random& rand)
{
	Transform transform = InterpolateTransform(p.startTransform, p.endTransform, time);

	switch (p.type)
	{
		case eSphere:
		{
			pos = TransformPoint(transform, UniformSampleSphere(rand)*p.sphere.radius);
			
			// todo: handle scaling in transform matrix
			area = 4.0f*kPi*p.sphere.radius*p.sphere.radius;  

			break;
		}
		case ePlane:
		{
			assert(0);
			return;
		}
		case eMesh:
		{
			assert(0);
			return;
		}
	}
}

struct Ray
{
	CUDA_CALLABLE inline Ray(const Vec3& o, const Vec3& d, float t) : origin(o), dir(d), time(t) {}

	Vec3 origin;
	Vec3 dir;

	float time;
};

CUDA_CALLABLE inline bool Intersect(const Primitive& p, const Ray& ray, float& outT, Vec3* outNormal)
{
	Transform transform = InterpolateTransform(p.startTransform, p.endTransform, ray.time);

	switch (p.type)
	{
		case eSphere:
		{
			bool hit = IntersectRaySphere(transform.p, p.sphere.radius, ray.origin, ray.dir, outT, outNormal);

			return hit;
		}
		case ePlane:
		{
			bool hit = IntersectRayPlane(ray.origin, ray.dir, (const Vec4&)p.plane, outT);
			
			if (hit && outNormal)
			{
				*outNormal = (const Vec3&)p.plane;
			}

			return hit;
		}
		case eMesh:
		{
			float t, u, v, w;
			int tri;

			// transform ray to mesh space			
			bool hit = IntersectRayMesh(p.mesh, InverseTransformPoint(transform, ray.origin), InverseTransformVector(transform, ray.dir), t, u, v, w, tri);
			//bool hit = IntersectRayMesh(p.mesh, ray.origin, ray.dir, t, u, v, w, tri);
			
			if (hit)
			{
				// interpolate vertex normals
				Vec3 n1 = p.mesh.normals[p.mesh.indices[tri*3+0]];
				Vec3 n2 = p.mesh.normals[p.mesh.indices[tri*3+1]];
				Vec3 n3 = p.mesh.normals[p.mesh.indices[tri*3+2]];

				outT = t;
				*outNormal = Normalize(TransformVector(transform, u*n1 + v*n2 + w*n3));
			}			

			return hit;
		}
	}

	return false;
}

enum FilterType
{
	eFilterBox,
	eFilterGaussian
};

struct Filter
{
	CUDA_CALLABLE Filter(FilterType type, float width, float falloff) : type(type), width(width), alpha(falloff)
	{
		if (type == eFilterGaussian)
			offset = expf(-alpha*width*width);
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
		return Max(0.0f, float(expf(-alpha*x*x)) - offset);
	}

	FilterType type;

	float width;
	float alpha;
	float offset;
};

enum RenderMode
{
	eNormals = 0,
	eComplexity = 1,
	ePathTrace =2
};

struct Renderer
{
	virtual ~Renderer() {}

	virtual void Init(int width, int height) {}
	virtual void Render(Camera* c, Color* output, int width, int height, int samplesPerPixel, Filter filter, RenderMode mode) = 0;

};

Renderer* CreateCpuRenderer(const Scene* s);
Renderer* CreateGpuRenderer(const Scene* s);
