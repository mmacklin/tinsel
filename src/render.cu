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

#define kBsdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.0001f

// create a texture object from memory and store it in a 64-bit pointer
void CreateIntTexture(int** deviceBuffer, const int* hostBuffer, int sizeInBytes)
{
	int* buffer;
	cudaMalloc(&buffer, sizeInBytes);
	cudaMemcpy(buffer, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);

#if USE_TEXTURES

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void*)buffer;
	resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = sizeInBytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// cast to pointer
	*deviceBuffer = (int*)tex;
#else

	*deviceBuffer = buffer;

#endif
}

// create a texture object from memory and store it in a 64-bit pointer
void CreateFloatTexture(float** deviceBuffer, const float* hostBuffer, int sizeInBytes)
{
	float* buffer;
	cudaMalloc(&buffer, sizeInBytes);
	cudaMemcpy(buffer, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);

#if USE_TEXTURES

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void*)buffer;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = sizeInBytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// cast to pointer
	*deviceBuffer = (float*)tex;

#else

	*deviceBuffer = buffer;

#endif
}

// create a texture object from memory and store it in a 64-bit pointer
void CreateVec4Texture(Vec4** deviceBuffer, const Vec4* hostBuffer, int sizeInBytes)
{
	Vec4* buffer;
	cudaMalloc(&buffer, sizeInBytes);
	cudaMemcpy(buffer, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);

#if USE_TEXTURES

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void*)buffer;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.desc.y = 32; // bits per channel
	resDesc.res.linear.desc.z = 32; // bits per channel
	resDesc.res.linear.desc.w = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = sizeInBytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// cast to pointer
	*deviceBuffer = (Vec4*)tex;

#else

	*deviceBuffer = buffer;

#endif

}


MeshGeometry CreateGPUMesh(const MeshGeometry& hostMesh)
{
	const int numVertices = hostMesh.numVertices;
	const int numIndices = hostMesh.numIndices;
	const int numNodes = hostMesh.numNodes;

	MeshGeometry gpuMesh;
	CreateFloatTexture((float**)&gpuMesh.positions, (float*)&hostMesh.positions[0], sizeof(Vec3)*numVertices);
	CreateFloatTexture((float**)&gpuMesh.normals, (float*)&hostMesh.normals[0], sizeof(Vec3)*numVertices);
	CreateIntTexture((int**)&gpuMesh.indices, (int*)&hostMesh.indices[0], sizeof(int)*numIndices);
	

	/*
	cudaMalloc((Vec3**)&gpuMesh.positions, sizeof(Vec3)*numVertices);
	cudaMemcpy((Vec3*)gpuMesh.positions, &hostMesh.positions[0], sizeof(Vec3)*numVertices, cudaMemcpyHostToDevice);

	cudaMalloc((Vec3**)&gpuMesh.normals, sizeof(Vec3)*numVertices);
	cudaMemcpy((Vec3*)gpuMesh.normals, &hostMesh.normals[0], sizeof(Vec3)*numVertices, cudaMemcpyHostToDevice);

	cudaMalloc((int**)&gpuMesh.indices, sizeof(int)*numIndices);
	cudaMemcpy((int*)gpuMesh.indices, &hostMesh.indices[0], sizeof(int)*numIndices, cudaMemcpyHostToDevice);
	*/
	//cudaMalloc((BVHNode**)&gpuMesh.nodes, sizeof(BVHNode)*numNodes);
	//cudaMemcpy((BVHNode*)gpuMesh.nodes, &hostMesh.nodes[0], sizeof(BVHNode)*numNodes, cudaMemcpyHostToDevice);

	CreateVec4Texture((Vec4**)&gpuMesh.nodes, (Vec4*)&hostMesh.nodes[0], sizeof(BVHNode)*numNodes);
	
	cudaMalloc((float**)&gpuMesh.cdf, sizeof(float)*numIndices/3);
	cudaMemcpy((float*)gpuMesh.cdf, &hostMesh.cdf[0], sizeof(float)*numIndices/3, cudaMemcpyHostToDevice);
	
	gpuMesh.numIndices = numIndices;
	gpuMesh.numVertices = numVertices;
	gpuMesh.numNodes = numNodes;
	gpuMesh.area = hostMesh.area;

	return gpuMesh;

}

void DestroyGPUMesh(const MeshGeometry& m)
{

}


Texture CreateGPUTexture(const Texture& tex)
{
	const int numTexels = tex.width*tex.height*tex.depth;
	
	Texture gpuTex = tex;

	cudaMalloc((void**)&gpuTex.data, sizeof(float)*numTexels);
	cudaMemcpy(gpuTex.data, tex.data, sizeof(float)*numTexels, cudaMemcpyHostToDevice);

	return gpuTex;
}

Sky CreateGPUSky(const Sky& sky)
{
	Sky gpuSky = sky;

	// copy probe
	if (sky.probe.valid)
	{
		const int numPixels = sky.probe.width*sky.probe.height;

		// copy pixel data
		CreateVec4Texture((Vec4**)&gpuSky.probe.data, sky.probe.data, numPixels*sizeof(float)*4);

		// copy cdf tables
		CreateFloatTexture((float**)&gpuSky.probe.cdfValuesX, sky.probe.cdfValuesX, numPixels*sizeof(float));
		CreateFloatTexture((float**)&gpuSky.probe.pdfValuesX, sky.probe.pdfValuesX, numPixels*sizeof(float));

		CreateFloatTexture((float**)&gpuSky.probe.cdfValuesY, sky.probe.cdfValuesY, sky.probe.height*sizeof(float));
		CreateFloatTexture((float**)&gpuSky.probe.pdfValuesY, sky.probe.pdfValuesY, sky.probe.height*sizeof(float));
	}

	return gpuSky;
}

void DestroyGPUSky(const Sky& gpuSky)
{
	if (gpuSky.probe.valid)
	{
		// todo
	}
}


// trace a ray against the scene returning the closest intersection
__device__ bool Trace(const GPUScene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
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
	outNormal = FaceForward(closestNormal, -ray.dir);
	*outPrimitive = closestPrimitive;

	return closestPrimitive != NULL;
}

__device__ inline float SampleTexture(const Texture& map, int i, int j, int k)
{
	int x = int(Abs(i))%map.width;
	int y = int(Abs(j))%map.height;
	int z = int(Abs(k))%map.depth;
	
	return map.data[z*map.width*map.height + y*map.width + x];
}


__device__ inline float LinearInterp(const Texture& map, const Vec3& pos) 
{
	int i = floorf(pos.x*map.width);
	int j = floorf(pos.y*map.height);
	int k = floorf(pos.z*map.depth);
		
	// trilinear interpolation
	float tx = pos.x*map.width-i;
	float ty = pos.y*map.height-j;
	float tz = pos.z*map.depth-k;
		
	float a = Lerp(SampleTexture(map, i, j, k), SampleTexture(map, i, j, k+1), tz);
	float b = Lerp(SampleTexture(map, i+1, j, k), SampleTexture(map, i+1, j, k+1), tz);
	float c = Lerp(SampleTexture(map, i, j+1, k), SampleTexture(map, i, j+1, k+1), tz);		
	float d = Lerp(SampleTexture(map, i+1, j+1, k), SampleTexture(map, i+1, j+1, k+1), tz);
		
	float e = Lerp(a, b, tx);
	float f = Lerp(c, d, tx);
		
	float g = Lerp(e, f, ty);
		
	return g;
}

__device__ inline Vec3 EvaluateBumpNormal(const Vec3& surfaceNormal, const Vec3& surfacePos, const Texture& bumpMap, const Vec3& bumpTile, float bumpStrength, Random& rand)
{
	Vec3 u, v;
	BasisFromVector(surfaceNormal, &u, &v);

	float eps = 0.01f;

	Vec3 dpdu = u + bumpStrength*surfaceNormal*(LinearInterp(bumpMap, bumpTile*(surfacePos)+u*eps) - LinearInterp(bumpMap, bumpTile*surfacePos))/eps;
	Vec3 dpdv = v + bumpStrength*surfaceNormal*(LinearInterp(bumpMap, bumpTile*(surfacePos)+v*eps) - LinearInterp(bumpMap, bumpTile*surfacePos))/eps;

	return SafeNormalize(Cross(dpdu, dpdv), surfaceNormal);
}



__device__ inline Color SampleLights(const GPUScene& scene, const Primitive& surfacePrimitive, float etaI, float etaO, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& shadingNormal, const Vec3& wo, float time, Random& rand)
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
			
			
			//if (Dot(wi, surfaceNormal) <= 0.0f)
//				continue;

			// check if occluded
			float t;
			Vec3 n;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos + FaceForward(surfaceNormal, wi)*kRayEpsilon, wi, time), t, n, &hit) == false)
			{
				float bsdfPdf = BSDFPdf(surfacePrimitive.material, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
				Color f = BSDFEval(surfacePrimitive.material, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
				
				if (bsdfPdf > 0.0f)
				{
					int N = kProbeSamples+kBsdfSamples;
					float cbsdf = kBsdfSamples/N;
					float csky = float(kProbeSamples)/N;
					float weight = csky*skyPdf/(cbsdf*bsdfPdf + csky*skyPdf);

					Validate(weight);

					if (weight > 0.0f)
						sum += weight*skyColor*f*Abs(Dot(wi, surfaceNormal))/skyPdf;
				}
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
			//if (Dot(wi, surfaceNormal) <= 0.0f)
				//continue; 				

			// surface is behind light
			if (Dot(wi, lightNormal) >= 0.0f)
				continue;

			// check visibility
			float t;
			Vec3 n;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos + FaceForward(surfaceNormal, wi)*kRayEpsilon, wi, time), t, n, &hit))			
			{
				float tSq = t*t;

				// if our next hit was further than distance to light then accept
				// sample, this works for portal sampling where you have a large light
				// that you sample through a small window
				const float kTolerance = 1.e-2f;

				if (fabsf(t - sqrtf(dSq)) <= kTolerance)
				{				
					const float nl = Abs(Dot(lightNormal, wi));

					// light pdf with respect to area and convert to pdf with respect to solid angle
					float lightArea = LightArea(lightPrimitive);
					float lightPdf = ((1.0f/lightArea)*tSq)/nl;

					// bsdf pdf for light's direction
					float bsdfPdf = BSDFPdf(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);
					Color f = BSDFEval(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);

					// this branch is only necessary to exclude specular paths from light sampling
					// todo: make BSDFEval alwasy return zero for pure specular paths and roll specular eval into BSDFSample()
					if (bsdfPdf > 0.0f)
					{
						// calculate relative weighting of the light and bsdf sampling
						int N = lightPrimitive.lightSamples+kBsdfSamples;
						float cbsdf = kBsdfSamples/N;
						float clight = float(lightPrimitive.lightSamples)/N;
						float weight = clight*lightPdf/(cbsdf*bsdfPdf + clight*lightPdf);
						
						L += weight*f*hit->material.emission*(Abs(Dot(wi, shadingNormal))/Max(1.e-3f, lightPdf));
					}
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
	float rayEta = 1.0f;
	Vec3 rayAbsorption = 0.0f;
	BSDFType rayType = eReflected;

    float t;
    Vec3 n, ns;
    const Primitive* hit;

	float bsdfPdf = 1.0f;

    for (int i=0; i < maxDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit))
        {	
			float outEta;
			Vec3 outAbsorption;

        	// index of refraction for transmission, 1.0 corresponds to air
			if (rayEta == 1.0f)
			{
        		outEta = hit->material.GetIndexOfRefraction();
				outAbsorption = Vec3(hit->material.absorption);
			}
			else
			{
				// returning to free space
				outEta = 1.0f;
				outAbsorption = 0.0f;
			}

			// update throughput based on absorption through the medium
			pathThroughput *= Color(Exp(-rayAbsorption*t), 1.0f);

#if 1
			
			if (i == 0)
			{
				// first trace is our only chance to add contribution from directly visible light sources        
				totalRadiance += hit->material.emission;
			}			
			else if (kBsdfSamples > 0)
			{
				// area pdf that this dir was already included by the light sampling from previous step
				float lightArea = LightArea(*hit);

				if (lightArea > 0.0f)
				{
					// convert to pdf with respect to solid angle
					float lightPdf = ((1.0f/lightArea)*t*t)/Abs(Dot(rayDir, n));

					// calculate weight for bsdf sampling
					int N = hit->lightSamples+kBsdfSamples;
					float cbsdf = kBsdfSamples/N;
					float clight = float(hit->lightSamples)/N;
					float weight = cbsdf*bsdfPdf/(cbsdf*bsdfPdf+ clight*lightPdf);
							
					Validate(weight);

					// specular paths have zero chance of being included by direct light sampling (zero pdf)
					if (rayType == eSpecular)
						weight = 1.0f;

					// pathThroughput already includes the bsdf pdf
					totalRadiance += weight*pathThroughput*hit->material.emission;
				}
			}

            // calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t;

/*
			if (hit->material.bump > 0.0f)
			{
				ns = FaceForward(EvaluateBumpNormal(n, p, hit->material.bumpMap, hit->material.bumpTile, hit->material.bump, rand), n);	
			}
			else
			{
				ns = n;
			}
*/

			// integrate direct light over hemisphere
			totalRadiance += pathThroughput*SampleLights(scene, *hit, rayEta, outEta, p, n, n, -rayDir, rayTime, rand);
#else
			
			// calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t + n*kRayEpsilon;

			totalRadiance += pathThroughput*hit->material.emission;

#endif

			// integrate indirect light by sampling BSDF
			Mat33 localFrame(u, v, n);

			Vec3 bsdfDir;
			BSDFType bsdfType;

			BSDFSample(hit->material, rayEta, outEta, p, Mat33(u,v,n), -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);
			
            if (bsdfPdf <= 0.0f)
            	break;

			Validate(bsdfPdf);

            // reflectance
            Color f = BSDFEval(hit->material, rayEta, outEta, p, n, -rayDir, bsdfDir);

            // update ray medium if we are transmitting through the material
            if (Dot(bsdfDir, n) <= 0.0f)
			{
            	rayEta = outEta;
				rayAbsorption = outAbsorption;
			}

            // update throughput with primitive reflectance
            pathThroughput *= f * Abs(Dot(n, bsdfDir))/bsdfPdf;

            // update ray direction and type
            rayType = bsdfType;
			rayDir = bsdfDir;            
			rayOrigin = p + FaceForward(n, bsdfDir)*kRayEpsilon;
			
        }
        else
        {
            // hit nothing, sample sky dome and terminate         
            float weight = 1.0f;

        	if (scene.sky.probe.valid && i > 0 && rayType == eReflected)
        	{ 
        		// probability that this dir was already sampled by probe sampling
        		float skyPdf = ProbePdf(scene.sky.probe, rayDir);
				 
				int N = kProbeSamples+kBsdfSamples;
				float cbsdf = kBsdfSamples/N;
				float csky = float(kProbeSamples)/N;
			
				weight = cbsdf*bsdfPdf/(cbsdf*bsdfPdf+ csky*skyPdf);

				Validate(bsdfPdf);
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
	const int tx = blockIdx.x*blockDim.x;
	const int ty = blockIdx.y*blockDim.y;

	const int i = tx + threadIdx.x;
	const int j = ty + threadIdx.y;

	if (i < options.width && j < options.height)
	{
		Vec3 origin;
		Vec3 dir;

		// initialize a per-thread PRNG
		Random rand(i + j*options.width + seed);

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

			if (primitive.material.bump > 0.0f)
			{
				primitive.material.bumpMap = CreateGPUTexture(primitive.material.bumpMap);
			}
			
			// create explicit list of light primitives
			if (primitive.lightSamples)
			{
				lights.push_back(primitive);
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
		// create a sampler for the camera
		CameraSampler sampler(
			Transform(camera.position, camera.rotation),
			camera.fov, 
			0.001f,
			1.0f,
			options.width,
			options.height);


		// assign threads in non-square tiles to match warp width
		const int blockWidth = 32;
		const int blockHeight = 8;

		const int gridWidth = (options.width + blockWidth - 1)/blockWidth;
		const int gridHeight = (options.height + blockHeight - 1)/blockHeight;

		dim3 blockDim(blockWidth, blockHeight);
		dim3 gridDim(gridWidth, gridHeight);

		RenderGpu<<<gridDim, blockDim>>>(sceneGPU, camera, sampler, options, seed.Rand(), output);

		// copy back to output
		cudaMemcpy(outputHost, output, sizeof(Color)*options.width*options.height, cudaMemcpyDeviceToHost);
	}
};


Renderer* CreateGpuRenderer(const Scene* s)
{
	return new GpuRenderer(s);
}
