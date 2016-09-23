#include "maths.h"
#include "render.h"
#include "disney.h"

#include <map>

struct GPUScene
{
	Primitive* primitives;
	int numPrimitives;

	Primitive* lights;
	int numLights;

	Color sky;
};


MeshGeometry CreateGPUMesh(const MeshGeometry& hostMesh)
{
	const int numVertices = hostMesh.numVertices;
	const int numIndices = hostMesh.numIndices;
	const int numNodes = hostMesh.numNodes;

	MeshGeometry gpuMesh;
	cudaMalloc(&gpuMesh.positions, sizeof(Vec3)*numVertices);
	cudaMemcpy((Vec3*)gpuMesh.positions, &hostMesh.positions[0], sizeof(Vec3)*numVertices, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuMesh.normals, sizeof(Vec3)*numVertices);
	cudaMemcpy((Vec3*)gpuMesh.normals, &hostMesh.normals[0], sizeof(Vec3)*numVertices, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuMesh.indices, sizeof(int)*numIndices);
	cudaMemcpy((int*)gpuMesh.indices, &hostMesh.indices[0], sizeof(int)*numIndices, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuMesh.nodes, sizeof(BVHNode)*numNodes);
	cudaMemcpy((BVHNode*)gpuMesh.nodes, &hostMesh.nodes[0], sizeof(BVHNode)*numNodes, cudaMemcpyHostToDevice);
	
	gpuMesh.numIndices = numIndices;
	gpuMesh.numVertices = numVertices;
	gpuMesh.numNodes = numNodes;

	return gpuMesh;

}

void DestroyGPUMesh(const MeshGeometry& m)
{

}


// trace a ray against the scene returning the closest intersection
__device__ bool Trace(const GPUScene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
	// disgard hits closer than this distance to avoid self intersection artifacts
	const float kEpsilon = 0.001f;

	float minT = REAL_MAX;
	const Primitive* closestPrimitive = NULL;
	Vec3 closestNormal(0.0f);

	for (int i=0; i < scene.numPrimitives; ++i)
	{
		const Primitive& primitive = scene.primitives[i];

		float t;
		Vec3 n;

		if (Intersect(primitive, ray, t, &n))
		{
			if (t < minT && t > kEpsilon)
			{
				minT = t;
				closestPrimitive = &primitive;
				closestNormal = n;
			}
		}
	}
	
	outT = minT;		
	outNormal = closestNormal;
	*outPrimitive = closestPrimitive;

	return closestPrimitive != NULL;
}



__device__ Color SampleLights(const GPUScene& scene, const Primitive& primitive, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& wo, float time, Random& rand)
{	
	Color sum(0.0f);

	for (int i=0; i < scene.numLights; ++i)
	{
		// assume all lights are area lights for now
		const Primitive& lightPrimitive = scene.lights[i];
				
		Color L(0.0f);

		const int numSamples = 2;

		for (int s=0; s < numSamples; ++s)
		{
			// sample light source
			Vec3 lightPos;
			float lightArea;

			Sample(lightPrimitive, time, lightPos, lightArea, rand);

			Vec3 wi = Normalize(lightPos-surfacePos);

			// ignore samples backfacing to the surface
			if (Dot(wi, surfaceNormal) < 0.0f)
				continue;
			
			// check visibility
			float t;
			Vec3 ln;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos, wi, time), t, ln, &hit))
			{
				// did we hit a light prim?
				if (hit->light)
				{
					const Color f = BRDFEval(primitive.material, surfacePos, surfaceNormal, wi, wo);

					// light pdf
					const float nl = Clamp(Dot(ln, -wi), 0.0f, 1.0f);
					
					if (nl > 0.0)
					{
						const float lightPdf = (t*t) / (nl*lightArea);
					
						L += f * hit->material.emission * Clamp(Dot(wi, surfaceNormal), 0.0f, 1.0f)  / lightPdf;
					}
				}
			}		
		}
	
		sum += L / float(numSamples);
	}

	return sum;
}


// reference, no light sampling, uniform hemisphere sampling
__device__ Color PathTrace(const GPUScene& scene, const Vec3& origin, const Vec3& dir, float time, Random& rand)
{	
    // path throughput
    Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Color totalRadiance(0.0f);

	Vec3 rayOrigin = origin;
	Vec3 rayDir = dir;
	float rayTime = time;

    float t = 0.0f;
    Vec3 n(rayDir);
    const Primitive* hit;

	const int maxDepth = 4;
//	const int maxSamples = 12;

	int pathCount = 1;
	int pathDepth = 0;

    for (int i=0; i < maxDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit) && pathDepth < maxDepth)
        {	
            // calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t;

			if (pathDepth == 0)
			{
				// first trace is our only chance to add contribution from directly visible light sources        
				totalRadiance += hit->material.emission;				
			}
			
			// integrate direct light over hemisphere
			totalRadiance += pathThroughput*SampleLights(scene, *hit, p, n, -rayDir, rayTime, rand);

			// integrate indirect light by sampling BRDF
			Mat33 localFrame(u, v, n);

			Vec3 outDir;
			float outPdf;
			BRDFSample(hit->material, p, localFrame, -rayDir, outDir, outPdf, rand);

            // reflectance
            Color f = BRDFEval(hit->material, p, n, -rayDir, outDir);

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, outDir), 0.0f, 1.0f) / outPdf;

            // update path direction
            rayDir = outDir;
            rayOrigin = p;

			pathDepth++;
        }
        else
        {
            // hit nothing, terminate loop
        	totalRadiance += scene.sky*pathThroughput;
			break;
			/*
			// reset path, this is the persistent threads model to keep generating new work
			GenerateRay(camera, x, y, rayOrigin, rayDir, rand);

			rayTime = rand.Randf();

			t = 0.0f;
			n = rayDir;

			pathThroughput = Color(1.0f, 1.0f, 1.0f);			
			pathCount++;
			pathDepth = 0;
			*/
        }
    }

    return totalRadiance/float(pathCount);
}

__device__ void AddSample(Color* output, int width, int height, float rasterX, float rasterY, float clamp, Filter filter, const Color& sample)
{
	switch (filter.type)
	{
		case eFilterBox:
		{
			int x = int(rasterX);
			int y = int(rasterY);

			output[(height-1-y)*width+x] += Color(sample.x, sample.y, sample.z, 1.0f);
			break;
		}
		case eFilterGaussian:
		{
			int startX = Max(0, int(rasterX - filter.width));
			int startY = Max(0, int(rasterY - filter.width));
			int endX = Min(int(rasterX + filter.width), width-1);
			int endY = Min(int(rasterY + filter.width), height-1);

			for (int x=startX; x <= endX; ++x)
			{
				for (int y=startY; y <= endY; ++y)
				{
					float w = filter.Eval(x-rasterX, y-rasterY);

					//output[(height-1-y)*width+x] += Color(Min(sample.x, clamp), Min(sample.y, clamp), Min(sample.z, clamp), 1.0f)*w;

					const int index = (height-1-y)*width+x;

					Color c =  Color(Min(sample.x, clamp), Min(sample.y, clamp), Min(sample.z, clamp), 1.0f)*w;

					atomicAdd(&output[index].x, c.x);
					atomicAdd(&output[index].y, c.y);
					atomicAdd(&output[index].z, c.z);
					atomicAdd(&output[index].w, c.w);
				}
			}
		
			break;
		}
	};
}

__global__ void RenderGpu(GPUScene scene, Camera camera, int width, int height, int maxDepth, int sampleIndex, RenderMode mode, int seed, Filter filter, Color* output)
{
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	const int i = tid%width;
	const int j = tid/width;

	if (i < width && j < height)
	{
		Vec3 origin;
		Vec3 dir;

		// initialize a per-thread PRNG
		Random rand(tid + seed);

		if (mode == eNormals)
		{
			GenerateRayNoJitter(camera, i, j, origin, dir);

			const Primitive* p;
			float t;
			Vec3 n;

			if (Trace(scene, Ray(origin, dir, 1.0f), t, n, &p))
			{
				n = n*0.5f+0.5f;
				output[(height-1-j)*width+i] = Color(n.x, n.y, n.z, 1.0f);
			}
			else
			{
				output[(height-1-j)*width+i] = Color(0.5f);
			}
		}
		else if (mode == ePathTrace)
		{
			float fx = i + rand.Randf(-0.5f, 0.5f) + 0.5f;
			float fy = j + rand.Randf(-0.5f, 0.5f) + 0.5f;

			Vec3 origin, dir;
			GenerateRay(camera, fx, fy, origin, dir);

#if 0
			// stratified motion blur sampling
			int strata = 16;
			float strataWidth = 1.0f/strata;
			float time = (float(sampleIndex%strata) + rand.Randf())*strataWidth;
#else
			float time = rand.Randf();
#endif

			//output[(height-1-j)*width+i] += PathTrace(*scene, origin, dir);
			Color sample = PathTrace(scene, origin, dir, time, rand);
			float maxValue = FLT_MAX;

			AddSample(output, width, height, fx, fy, maxValue, filter, sample);
		}
	}
}

struct GpuRenderer : public Renderer
{
	Color* output = NULL;
	
	GPUScene sceneGPU;
	
	Random seed;

	// map id to geometry struct
	std::map<int, MeshGeometry> gpuMeshes;

	GpuRenderer(const Scene* s)
	{
		// build GPU primitive and light lists
		std::vector<Primitive> primitives;		
		std::vector<Primitive> lights;

		for (int i=0; i < s->primitives.size(); ++i)
		{
			Primitive primitive = s->primitives[i];

			if (primitive.light)
			{
				lights.push_back(primitive);
			}

			// if mesh primitive then copy to the GPU
			if (primitive.type == eMesh)
			{
				// see if we have already uploaded the mesh to the GPU
				if (gpuMeshes.find(primitive.mesh.id) == gpuMeshes.end())
				{
					MeshGeometry geo = CreateGPUMesh(primitive.mesh);
					gpuMeshes[geo.id] = geo;

					// replace CPU mesh with GPU copy
					primitive.mesh = geo;
				}
			}	
			
			primitives.push_back(primitive);
		}

		// upload to the GPU
		sceneGPU.numPrimitives = primitives.size();
		sceneGPU.numLights = lights.size();

		if (sceneGPU.numLights > 0)
		{
			cudaMalloc(&sceneGPU.lights, sizeof(Primitive)*lights.size());
			cudaMemcpy(sceneGPU.lights, &lights[0], sizeof(Primitive)*lights.size(), cudaMemcpyHostToDevice);
		}

		if (sceneGPU.numPrimitives > 0)
		{
			cudaMalloc(&sceneGPU.primitives, sizeof(Primitive)*primitives.size());
			cudaMemcpy(sceneGPU.primitives, &primitives[0], sizeof(Primitive)*primitives.size(), cudaMemcpyHostToDevice);
		}

		// misc params
		sceneGPU.sky = s->sky;
	}

	virtual ~GpuRenderer()
	{
		cudaFree(output);
		cudaFree(sceneGPU.primitives);
		cudaFree(sceneGPU.lights);
	}
	
	void Init(int width, int height)
	{
		cudaFree(output);
		cudaMalloc(&output, sizeof(Color)*width*height);
		cudaMemset(output, 0, sizeof(Color)*width*height);
	}

	void Render(Camera* camera, Color* outputHost, int width, int height, int samplesPerPixel, Filter filter, RenderMode mode)
	{
		const int numThreads = width*height;
		const int kNumThreadsPerBlock = 256;
		const int kNumBlocks = (numThreads + kNumThreadsPerBlock - 1) / (kNumThreadsPerBlock);

		const int maxDepth = 40;
		
		static int sampleIndex = 0;

		for (int i=0; i < samplesPerPixel; ++i)
		{
			RenderGpu<<<kNumBlocks, kNumThreadsPerBlock>>>(sceneGPU, *camera, width, height, maxDepth, sampleIndex, mode, seed.Rand(), filter, output);

			++sampleIndex;
		}

		// copy back to output
		cudaMemcpy(outputHost, output, sizeof(Color)*numThreads, cudaMemcpyDeviceToHost);
	}
};


Renderer* CreateGpuRenderer(const Scene* s)
{
	return new GpuRenderer(s);
}
