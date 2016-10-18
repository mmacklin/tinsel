#include "maths.h"
#include "render.h"
#include "util.h"
#include "disney.h"

#include <map>

struct GPUScene
{
	Primitive* primitives;
	int numPrimitives;

	Primitive* lights;
	int numLights;

	Sky sky;
};

#define kBrdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.001f

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

Sky CreateGPUSky(const Sky& sky)
{
	Sky gpuSky = sky;

	// copy probe
	if (sky.probe.valid)
	{
		const int numPixels = sky.probe.width*sky.probe.height;

		// copy pixel data
		cudaMalloc(&gpuSky.probe.data, numPixels*sizeof(float)*4);
		cudaMemcpy(gpuSky.probe.data, sky.probe.data, numPixels*sizeof(float)*4, cudaMemcpyHostToDevice);

		// copy cdf tables
		cudaMalloc(&gpuSky.probe.cdfValuesX, numPixels*sizeof(float));
		cudaMemcpy(gpuSky.probe.cdfValuesX, sky.probe.cdfValuesX, numPixels*sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc(&gpuSky.probe.cdfValuesY, sky.probe.height*sizeof(float));
		cudaMemcpy(gpuSky.probe.cdfValuesY, sky.probe.cdfValuesY, sky.probe.height*sizeof(float), cudaMemcpyHostToDevice);

		// copy pdf tables
		cudaMalloc(&gpuSky.probe.pdfValuesX, numPixels*sizeof(float));
		cudaMemcpy(gpuSky.probe.pdfValuesX, sky.probe.pdfValuesX, numPixels*sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc(&gpuSky.probe.pdfValuesY, sky.probe.height*sizeof(float));
		cudaMemcpy(gpuSky.probe.pdfValuesY, sky.probe.pdfValuesY, sky.probe.height*sizeof(float), cudaMemcpyHostToDevice);

	}

	return gpuSky;
}

void DestroyGPUSky(const Sky& gpuSky)
{
	if (gpuSky.probe.valid)
	{
		cudaFree(gpuSky.probe.data);
	}
}


// trace a ray against the scene returning the closest intersection
__device__ bool Trace(const GPUScene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
	// disgard hits closer than this distance to avoid self intersection artifacts
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
			if (t < minT && t > 0.0f)
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


__device__ inline Color SampleLights(const GPUScene& scene, const Primitive& surfacePrimitive, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& wo, float time, Random& rand)
{	
	Color sum(0.0f);

	if (scene.sky.probe.valid)
	{
		for (int i=0; i < kProbeSamples; ++i)
		{

			Color skyColor;
			float skyPdf;
			Vec3 wi;

			ProbeSample(scene.sky.probe, wi, skyColor, skyPdf, rand);
			
			/*
			wi = UniformSampleSphere(rand);
			skyColor = ProbeEval(scene.sky.probe, ProbeDirToUV(wi));
			skyPdf = 0.5f*kInv2Pi;
			*/	
			
			
			if (Dot(wi, surfaceNormal) <= 0.0f)
				continue;

			// check if occluded
			float t;
			Vec3 n;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos, wi, time), t, n, &hit) == false)
			{
				float brdfPdf = BRDFPdf(surfacePrimitive.material, surfacePos, surfaceNormal, wo, wi);
				Color f = BRDFEval(surfacePrimitive.material, surfacePos, surfaceNormal, wo, wi);
				
				int N = kProbeSamples+kBrdfSamples;
				float cbrdf = kBrdfSamples/N;
				float csky = float(kProbeSamples)/N;
				float weight = csky*skyPdf/(cbrdf*brdfPdf + csky*skyPdf);

				Validate(weight);

				if (weight > 0.0f)
					sum += weight*skyColor*f*Abs(Dot(wi, surfaceNormal))/skyPdf;
			}
		}

		if (kProbeSamples > 0)
			sum /= float(kProbeSamples);
	}

	for (int i=0; i < scene.numLights; ++i)
	{
		// assume all lights are area lights for now
		const Primitive& lightPrimitive = scene.lights[i];

		Color L(0.0f);

		int numSamples = lightPrimitive.lightSamples;

		if (numSamples == 0)
			continue;

		for (int s=0; s < numSamples; ++s)
		{
			// sample light source
			Vec3 lightPos;
			Vec3 lightNormal;

			LightSample(lightPrimitive, time, lightPos, lightNormal, rand);
			
			Vec3 wi = lightPos-surfacePos;
			
			float dSq = LengthSq(wi);
			wi /= sqrtf(dSq);

			// light is behind surface
			if (Dot(wi, surfaceNormal) <= 0.0f)
				continue; 				

			// surface is behind light
			if (Dot(wi, lightNormal) >= 0.0f)
				continue;

			// check visibility
			float t;
			Vec3 n;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos, wi, time), t, n, &hit))			
			{
				float tSq = t*t;

				// if our next hit was further than distance to light then accept
				// sample, this works for portal sampling where you have a large light
				// that you sample through a small window
				const float kTolerance = 1.e-2f;

				if (fabsf(t - sqrtf(dSq)) <= kTolerance)
				{				
					const float nl = Dot(lightNormal, -wi);

					// light pdf with respect to area and convert to pdf with respect to solid angle
					float lightArea = LightArea(lightPrimitive);
					float lightPdf = ((1.0f/lightArea)*tSq)/nl;

					// brdf pdf for light's direction
					float brdfPdf = BRDFPdf(surfacePrimitive.material, surfacePos, surfaceNormal, wo, wi);
					Color f = BRDFEval(surfacePrimitive.material, surfacePos, surfaceNormal, wo, wi);

					// calculate relative weighting of the light and brdf sampling
					int N = lightPrimitive.lightSamples+kBrdfSamples;
					float cbrdf = kBrdfSamples/N;
					float clight = float(lightPrimitive.lightSamples)/N;
					float weight = clight*lightPdf/(cbrdf*brdfPdf + clight*lightPdf);
						
					L += weight*f*hit->material.emission*(Abs(Dot(wi, surfaceNormal))/Max(1.e-3f, lightPdf));
				}
			}
		}
	
		sum += L * (1.0f/numSamples);
	}

	return sum;
}


// reference, no light sampling, uniform hemisphere sampling
__device__ Color PathTrace(const GPUScene& scene, const Vec3& origin, const Vec3& dir, float time, int maxDepth, Random& rand)
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

	float brdfPdf = 1.0f;

    for (int i=0; i < maxDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit))
        {	
#if 1
			
			if (i == 0)
			{
				// first trace is our only chance to add contribution from directly visible light sources        
				totalRadiance += hit->material.emission;
			}			
			else if (kBrdfSamples > 0)
			{
				// area pdf that this dir was already included by the light sampling from previous step
				float lightArea = LightArea(*hit);

				if (lightArea > 0.0f)
				{
					// convert to pdf with respect to solid angle
					float lightPdf = ((1.0f/lightArea)*t*t)/Clamp(Dot(-rayDir, n), 1.e-3f, 1.0f);

					// calculate weight for brdf sampling
					int N = hit->lightSamples+kBrdfSamples;
					float cbrdf = kBrdfSamples/N;
					float clight = float(hit->lightSamples)/N;
					float weight = cbrdf*brdfPdf/(cbrdf*brdfPdf+ clight*lightPdf);
							
					Validate(weight);

					// pathThroughput already includes the brdf pdf
					totalRadiance += weight*pathThroughput*hit->material.emission;
				}
			}

            // calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t + n*kRayEpsilon;

			// integrate direct light over hemisphere
			totalRadiance += pathThroughput*SampleLights(scene, *hit, p, n, -rayDir, rayTime, rand);
#else
			
			// calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t + n*kRayEpsilon;

			totalRadiance += pathThroughput*hit->material.emission;

#endif

			// integrate indirect light by sampling BRDF
			Mat33 localFrame(u, v, n);

            Vec3 brdfDir = BRDFSample(hit->material, p, Mat33(u, v, n), -rayDir, rand);
			brdfPdf = BRDFPdf(hit->material, p, n, -rayDir, brdfDir);

			
            if (brdfPdf <= 0.0f)
            	break;

            if (Dot(brdfDir, n) <= 0.0f)
            	break;
				

			Validate(brdfPdf);


            // reflectance
            Color f = BRDFEval(hit->material, p, n, -rayDir, brdfDir);

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, brdfDir), 0.0f, 1.0f)/brdfPdf;

            // update path direction
            rayDir = brdfDir;
            rayOrigin = p;
        }
        else
        {
            // hit nothing, sample sky dome and terminate         
            float weight = 1.0f;

        	if (scene.sky.probe.valid && i > 0)
        	{ 
        		// probability that this dir was already sampled by probe sampling
        		float skyPdf = ProbePdf(scene.sky.probe, rayDir);
				 
				int N = kProbeSamples+kBrdfSamples;
				float cbrdf = kBrdfSamples/N;
				float csky = float(kProbeSamples)/N;
			
				weight = cbrdf*brdfPdf/(cbrdf*brdfPdf+ csky*skyPdf);

				Validate(brdfPdf);
				Validate(skyPdf);

			}

			Validate(weight);
		
       		totalRadiance += weight*scene.sky.Eval(rayDir)*pathThroughput; 
			break;
        }
    }

    return totalRadiance;
}

__device__ void AddSample(Color* output, int width, int height, float rasterX, float rasterY, float clamp, Filter filter, const Color& sample)
{
	switch (filter.type)
	{
		case eFilterBox:
		{
			int x = int(rasterX);
			int y = int(rasterY);

			output[y*width+x] += Color(sample.x, sample.y, sample.z, 1.0f);
			break;
		}
		case eFilterGaussian:
		{
			int startX = Max(0, int(rasterX - filter.width));
			int startY = Max(0, int(rasterY - filter.width));
			int endX = Min(int(rasterX + filter.width), width-1);
			int endY = Min(int(rasterY + filter.width), height-1);

			Color c =  ClampLength(sample, clamp);
			c.w = 1.0f;

			for (int x=startX; x <= endX; ++x)
			{
				for (int y=startY; y <= endY; ++y)
				{
					float w = filter.Eval(x-rasterX, y-rasterY);

					//output[(height-1-y)*width+x] += Color(Min(sample.x, clamp), Min(sample.y, clamp), Min(sample.z, clamp), 1.0f)*w;

					const int index = y*width+x;

					atomicAdd(&output[index].x, c.x*w);
					atomicAdd(&output[index].y, c.y*w);
					atomicAdd(&output[index].z, c.z*w);
					atomicAdd(&output[index].w, w);
				}
			}
		
			break;
		}
	};
}

__global__ void RenderGpu(GPUScene scene, Camera camera, CameraSampler sampler, Options options, int seed, Color* output)
{
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	const int i = tid%options.width;
	const int j = tid/options.width;

	if (i < options.width && j < options.height)
	{
		Vec3 origin;
		Vec3 dir;

		// initialize a per-thread PRNG
		Random rand(tid + seed);

		if (options.mode == eNormals)
		{
			sampler.GenerateRay(i, j, origin, dir);

			const Primitive* p;
			float t;
			Vec3 n;

			if (Trace(scene, Ray(origin, dir, 1.0f), t, n, &p))
			{
				n = n*0.5f+0.5f;
				output[j*options.width+i] = Color(n.x, n.y, n.z, 1.0f);
			}
			else
			{
				output[j*options.width+i] = Color(0.5f);
			}
		}
		else if (options.mode == ePathTrace)
		{
			const float time = rand.Randf(camera.shutterStart, camera.shutterEnd);
			const float fx = i + rand.Randf(-0.5f, 0.5f) + 0.5f;
			const float fy = j + rand.Randf(-0.5f, 0.5f) + 0.5f;

			Vec3 origin, dir;
			sampler.GenerateRay(fx, fy, origin, dir);

			//output[(height-1-j)*width+i] += PathTrace(*scene, origin, dir);
			Color sample = PathTrace(scene, origin, dir, time, options.maxDepth, rand);

			AddSample(output, options.width, options.height, fx, fy, options.clamp, options.filter, sample);
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

			if (primitive.lightSamples)
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

		// copy sky and probe texture
		sceneGPU.sky = CreateGPUSky(s->sky);
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

	void Render(const Camera& camera, const Options& options, Color* outputHost)
	{
		const int numThreads = options.width*options.height;
		const int kNumThreadsPerBlock = 256;
		const int kNumBlocks = (numThreads + kNumThreadsPerBlock - 1) / (kNumThreadsPerBlock);
	
		// create a sampler for the camera
		CameraSampler sampler(
			Transform(camera.position, camera.rotation),
			camera.fov, 
			0.001f,
			1.0f,
			options.width,
			options.height);

		RenderGpu<<<kNumBlocks, kNumThreadsPerBlock>>>(sceneGPU, camera, sampler, options, seed.Rand(), output);

		// copy back to output
		cudaMemcpy(outputHost, output, sizeof(Color)*numThreads, cudaMemcpyDeviceToHost);
	}
};


Renderer* CreateGpuRenderer(const Scene* s)
{
	return new GpuRenderer(s);
}
