#include "maths.h"
#include "render.h"
#include "util.h"
#include "disney.h"
#include "bvh.h"

#include <map>

struct GPUScene
{
	Primitive* primitives;
	int numPrimitives;

	Primitive* lights;
	int numLights;

	Sky sky;

	BVH bvh;
};

#define kBsdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.0001f

#define LAUNCH_BOUNDS __launch_bounds__(256, 4)

__device__ inline int getGlobalIndex()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

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

#if USE_TEXTURES
	
	// expand positions out to vec4
	std::vector<Vec4> positions;
	std::vector<Vec4> normals;

	for (int i=0; i < numVertices; ++i)
	{
		positions.push_back(Vec4(hostMesh.positions[i], 1.0f));
		normals.push_back(Vec4(hostMesh.normals[i], 0.0f));
	}

	CreateVec4Texture((Vec4**)&gpuMesh.positions, (Vec4*)&positions[0], sizeof(Vec4)*numVertices);
	CreateVec4Texture((Vec4**)&gpuMesh.normals, (Vec4*)&normals[0], sizeof(Vec4)*numVertices);

#else
	CreateFloatTexture((float**)&gpuMesh.positions, (float*)&hostMesh.positions[0], sizeof(Vec3)*numVertices);
	CreateFloatTexture((float**)&gpuMesh.normals, (float*)&hostMesh.normals[0], sizeof(Vec3)*numVertices);

#endif

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


#if 1


inline __device__ bool Trace(const GPUScene& scene, const Vec3& rayOrigin, const Vec3& rayDir, float rayTime, float& outT, Vec3& outNormal, const Primitive** RESTRICT outPrimitive)
{
	int stack[64];
	stack[0] = 0;

	unsigned int count = 1;

	Vec3 dir, rcpDir;
	Vec3 origin;
	
	rcpDir.x = 1.0f/rayDir.x;
	rcpDir.y = 1.0f/rayDir.y;
	rcpDir.z = 1.0f/rayDir.z;
	origin = rayOrigin;
	dir = rayDir;

	const BVHNode* RESTRICT root = scene.bvh.nodes;

	MeshGeometry mesh;
	int primitiveIndex = -1;

	float closestT = FLT_MAX;
	//float closestU;
	float closestV;
	float closestW;

	Vec3 closestNormal;
	int closestPrimitive = -1;
	int closestTri;

	while(count)
	{
		const int nodeIndex = stack[--count];

		if (nodeIndex < 0)
		{
			// reset to scene bvh dir and address
			rcpDir.x = 1.0f/rayDir.x;
			rcpDir.y = 1.0f/rayDir.y;
			rcpDir.z = 1.0f/rayDir.z;
			origin = rayOrigin;
			dir = rayDir;
			root = scene.bvh.nodes;
			primitiveIndex = -1;

			continue;
		}

		BVHNode node = fetchNode(root, nodeIndex);

		int leftIndex = node.leftIndex;
		int rightIndex = node.rightIndex;

		if (node.leaf)
		{
			if (primitiveIndex < 0)
			{
				const Primitive& p = scene.primitives[leftIndex];

				Transform transform = InterpolateTransform(p.startTransform, p.endTransform, rayTime);

				switch (p.type)
				{
					case eSphere:
					{
						float minT, maxT;
						Vec3 n;

						bool hit = IntersectRaySphere(transform.p, p.sphere.radius*transform.s, origin, dir, minT, maxT, &n);

						if (hit && minT < closestT)
						{
							closestT = minT;
							closestNormal = n;
							closestPrimitive = leftIndex;
						}
						break;
					}
					case ePlane:
					{
						float t;
						bool hit = IntersectRayPlane(origin, dir, (const Vec4&)p.plane, t);
			
						if (hit && t < closestT)
						{
							closestT = t;
							closestNormal = (const Vec3&)p.plane;							
							closestPrimitive = leftIndex;
						}

						break;
					}
					case eMesh:
					{
						// push a back-tracking marker in the stack
						stack[count++] = -1;

						// push root of the mesh bvh
						stack[count++] = 0;

						// transform ray to primitive local space
						origin = InverseTransformPoint(transform, rayOrigin);					
						dir = InverseTransformVector(transform, rayDir);

						rcpDir.x = 1.0f/dir.x;
						rcpDir.y = 1.0f/dir.y;
						rcpDir.z = 1.0f/dir.z;				
				
						// set bvh and mesh sources
						root = p.mesh.nodes;
						mesh = p.mesh;

						primitiveIndex = leftIndex;
						break;
					}
				};
			}
			else
			{
				// mesh mode
				int i0 = fetchInt(mesh.indices, leftIndex*3+0);
				int i1 = fetchInt(mesh.indices, leftIndex*3+1);
				int i2 = fetchInt(mesh.indices, leftIndex*3+2);

				const Vec3 a = fetchVec3(mesh.positions, i0);
				const Vec3 b = fetchVec3(mesh.positions, i1);
				const Vec3 c = fetchVec3(mesh.positions, i2);

				float t, u, v, w;
				float sign;
				Vec3 n;
				//if (IntersectRayTri(rayOrigin, rayDir, a, b, c, t, u, v, w, &n))
				if (IntersectRayTriTwoSided(origin, dir, a, b, c, t, u, v, w, sign, &n))
				{
					if (t > 0.0f && t < closestT)
					{
						closestT = t;
						//closestU = u;
						closestV = v;
						closestW = w;

						closestTri = leftIndex;
						closestNormal = n*sign;						
						closestPrimitive = primitiveIndex;
					}
				}
			}
		}
		else
		{
			// check children
			BVHNode left = fetchNode(root, leftIndex);
			BVHNode right = fetchNode(root, rightIndex);

			float tLeft;
			bool hitLeft = IntersectRayAABBFast(origin, rcpDir, left.bounds.lower, left.bounds.upper, tLeft);// && tLeft < closestT;

			float tRight;
			bool hitRight = IntersectRayAABBFast(origin, rcpDir, right.bounds.lower, right.bounds.upper, tRight);// && tRight < closestT;

			// traverse closest first
			if (hitLeft && hitRight && (tLeft < tRight))
			{
				//Swap(leftIndex, rightIndex);
			}

			if (hitLeft)
				stack[count++] = leftIndex;

			if (hitRight)
				stack[count++] = rightIndex;			
		}
	}

	
	if (closestPrimitive >= 0)
	{
		const Primitive& p = scene.primitives[closestPrimitive];

		if (p.type == eMesh)
		{
			Transform transform = InterpolateTransform(p.startTransform, p.endTransform, rayTime);

			// interpolate vertex normals
			int i0 = fetchInt(p.mesh.indices, closestTri*3+0);
			int i1 = fetchInt(p.mesh.indices, closestTri*3+1);
			int i2 = fetchInt(p.mesh.indices, closestTri*3+2);

			const Vec3 n1 = fetchVec3(p.mesh.normals, i0);
			const Vec3 n2 = fetchVec3(p.mesh.normals, i1);
			const Vec3 n3 = fetchVec3(p.mesh.normals, i2);

			Vec3 smoothNormal = (1.0f-closestV-closestW)*n1 + closestV*n2 + closestW*n3;

			// ensure smooth normal lies on the same side of the geometric normal
			if (Dot(smoothNormal, closestNormal) < 0.0f)
				smoothNormal *= -1.0f;

			closestNormal = SafeNormalize(TransformVector(transform, smoothNormal), closestNormal);
		}

		outT = closestT;
		outNormal = FaceForward(closestNormal, -rayDir);
		
		if (outPrimitive)
			*outPrimitive = &p;

		return true;
	}
	else
	{
		// no hit
		return false;
	}
}

#else

// trace a ray against the scene returning the closest intersection
inline __device__ bool Trace(const GPUScene& scene, const Vec3& rayOrigin, const Vec3& rayDir, float rayTime, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{

#if 0

	struct Callback
	{
		float minT;
		Vec3 closestNormal;
		const Primitive* closestPrimitive;

		const Ray& ray;
		const GPUScene& scene;

		CUDA_CALLABLE inline Callback(const GPUScene& s, const Ray& r) : minT(REAL_MAX), closestPrimitive(NULL), ray(r), scene(s)
		{

		}
		
		CUDA_CALLABLE inline void operator()(int index)
		{
			float t;
			Vec3 n, ns;

			const Primitive& primitive = scene.primitives[index];

			if (PrimitiveIntersect(primitive, ray, t, &n))
			{
				if (t < minT && t > 0.0f)
				{
					minT = t;
					closestPrimitive = &primitive;
					closestNormal = n;
				}
			}			
		}
	};

	Callback callback(scene, ray);
	QueryBVH(callback, scene.bvh.nodes, ray.origin, ray.dir);

	outT = callback.minT;		
	outNormal = FaceForward(callback.closestNormal, -ray.dir);
	
	if (outPrimitive)
		*outPrimitive = callback.closestPrimitive;

	return callback.closestPrimitive != NULL;
	
#else

	float minT = REAL_MAX;
	const Primitive* closestPrimitive = NULL;
	Vec3 closestNormal(0.0f);

	for (int i=0; i < scene.numPrimitives; ++i)
	{
		const Primitive& primitive = scene.primitives[i];

		float t;
		Vec3 n;

		if (PrimitiveIntersect(primitive, Ray(rayOrigin, rayDir, rayTime), t, &n))
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
	outNormal = FaceForward(closestNormal, -rayDir);
	
	if (outPrimitive)
		*outPrimitive = closestPrimitive;

	return closestPrimitive != NULL;

#endif

}

#endif


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



__device__ inline Vec3 SampleLights(const GPUScene& scene, const Primitive& surfacePrimitive, float etaI, float etaO, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& shadingNormal, const Vec3& wo, float time, Random& rand)
{	
	Vec3 sum(0.0f);
	
	if (scene.sky.probe.valid)
	{
		for (int i=0; i < kProbeSamples; ++i)
		{

			Vec3 skyColor;
			float skyPdf;
			Vec3 wi;

			ProbeSample(scene.sky.probe, wi, skyColor, skyPdf, rand);
			
			
			//wi = UniformSampleSphere(rand);
			//skyColor = ProbeEval(scene.sky.probe, ProbeDirToUV(wi));
			//skyPdf = 0.5f*kInv2Pi;
						
			
			//if (Dot(wi, surfaceNormal) <= 0.0f)
//				continue;

			// check if occluded
			float t;
			Vec3 n;
			
			if (Trace(scene, surfacePos + FaceForward(surfaceNormal, wi)*kRayEpsilon, wi, time, t, n, NULL) == false)
			{
				float bsdfPdf = BSDFPdf(surfacePrimitive.material, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
				Vec3 f = BSDFEval(surfacePrimitive.material, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
				
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

		Vec3 L(0.0f);

		int numSamples = lightPrimitive.lightSamples;

		if (numSamples == 0)
			continue;

		for (int s=0; s < numSamples; ++s)
		{
			// sample light source
			Vec3 lightPos;
			Vec3 lightNormal;

			PrimitiveSample(lightPrimitive, time, lightPos, lightNormal, rand);
			
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
			if (Trace(scene, surfacePos + FaceForward(surfaceNormal, wi)*kRayEpsilon, wi, time, t, n, NULL))			
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
					float lightArea = PrimitiveArea(lightPrimitive);
					float lightPdf = ((1.0f/lightArea)*tSq)/nl;

					// bsdf pdf for light's direction
					float bsdfPdf = BSDFPdf(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);
					Vec3 f = BSDFEval(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);

					// this branch is only necessary to exclude specular paths from light sampling (always have zero brdf)
					// todo: make BSDFEval alwasy return zero for pure specular paths and roll specular eval into BSDFSample()
					if (bsdfPdf > 0.0f)
					{
						// calculate relative weighting of the light and bsdf sampling
						int N = lightPrimitive.lightSamples+kBsdfSamples;
						float cbsdf = kBsdfSamples/N;
						float clight = float(lightPrimitive.lightSamples)/N;
						float weight = clight*lightPdf/(cbsdf*bsdfPdf + clight*lightPdf);
						
						L += weight*f*lightPrimitive.material.emission*(Abs(Dot(wi, shadingNormal))/Max(1.e-3f, lightPdf));
					}
				}
			}
		}
	
		sum += L * (1.0f/numSamples);
	}

	return sum;
}


struct Tile
{
	int x;
	int y;
	int width;
	int height;
};

enum PathMode
{
	ePathGenerate,
	ePathAdvance,
	ePathProbeSample,
	ePathLightSample,
	ePathBsdfSample,
	ePathTerminate,
	ePathDisabled,
};


struct PathState
{		
	Vec3* __restrict__ rayOrigin;
	Vec3* __restrict__ rayDir;
	float* __restrict__ rayTime;

	Vec3* __restrict__ pos;
	Vec3* __restrict__ normal;

	int* __restrict__ depth;

	Vec3* __restrict__ pathThroughput;
	Vec3* __restrict__ absorption;
	const Primitive** __restrict__ primitive;

	Vec3* __restrict__ totalRadiance;

	float* __restrict__ etaI;
	float* __restrict__ etaO;

	PathMode* __restrict__ mode;

	// pdf from last brdf sampling
	float* __restrict__ bsdfPdf;
	BSDFType* __restrict__ bsdfType;

	// sample coordinate
	float* __restrict__ rasterX;
	float* __restrict__ rasterY;

	Random* __restrict__ rand;
};

template <typename T>
void Alloc(T** ptr, int num)
{
	cudaMalloc(ptr, sizeof(T)*num);
	cudaMemset(*ptr, 0, sizeof(T)*num);
}

PathState AllocatePaths(int num)
{
	PathState state;

	Alloc(&state.rayOrigin, num);
	Alloc(&state.rayDir, num);
	Alloc(&state.rayTime, num);

	Alloc(&state.pos, num);
	Alloc(&state.normal, num);

	Alloc(&state.depth, num);

	Alloc(&state.pathThroughput, num);
	Alloc(&state.absorption, num);
	Alloc(&state.primitive, num);
	Alloc(&state.totalRadiance, num);

	Alloc(&state.etaI, num);
	Alloc(&state.etaO, num);

	Alloc(&state.mode, num);

	Alloc(&state.bsdfPdf, num);
	Alloc(&state.bsdfType, num);

	Alloc(&state.rasterX, num);
	Alloc(&state.rasterY, num);

	Alloc(&state.rand, num);

	return state;
}

void FreePaths(PathState state)
{
	// todo:
}




LAUNCH_BOUNDS
__global__ void TerminatePaths(Color* output, Options options, PathState paths, int numPaths)
{
	const int i = getGlobalIndex();

	{
		if (paths.mode[i] != ePathDisabled)
		{
			float rasterX = paths.rasterX[i];
			float rasterY = paths.rasterY[i];

			Vec3 sample = paths.totalRadiance[i];

			// sample = paths[i].normal*0.5f + 0.5f;

			int width = options.width;
			int height = options.height;

			Filter filter = options.filter;

			switch (filter.type)
			{
				case eFilterBox:
				{
					int x = Clamp(int(rasterX), 0, width-1);
					int y = Clamp(int(rasterY), 0, height-1);

					output[y*width+x] += Color(sample.x, sample.y, sample.z, 1.0f);
					break;
				}
				case eFilterGaussian:
				{
					int startX = Max(0, int(rasterX - filter.width));
					int startY = Max(0, int(rasterY - filter.width));
					int endX = Min(int(rasterX + filter.width), width-1);
					int endY = Min(int(rasterY + filter.width), height-1);

					Vec3 c =  ClampLength(sample, options.clamp);

					for (int x=startX; x <= endX; ++x)
					{
						for (int y=startY; y <= endY; ++y)
						{
							float w = filter.Eval(x-rasterX, y-rasterY);

							//output[(height-1-y)*width+x] += Vec3(Min(sample.x, clamp), Min(sample.y, clamp), Min(sample.z, clamp), 1.0f)*w;

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

		paths.mode[i] = ePathGenerate;
	}
}

LAUNCH_BOUNDS
__global__ void SampleLights(GPUScene scene, PathState paths, int numPaths)
{
	const int i = getGlobalIndex();

	{
		if (paths.mode[i] == ePathLightSample)
		{
        	// calculate a basis for this hit point
        	const Primitive* hit = paths.primitive[i];        	
        	
        	float etaI = paths.etaI[i];
        	float etaO = paths.etaO[i];

			const Vec3 rayDir = paths.rayDir[i];
            float rayTime = paths.rayTime[i];

            const Vec3 p = paths.pos[i];
            const Vec3 n = paths.normal[i];

			// integrate direct light over hemisphere
			paths.totalRadiance[i] += paths.pathThroughput[i]*SampleLights(scene, *hit, etaI, etaO, p, n, n, -rayDir, rayTime, paths.rand[i]);			

			paths.mode[i] = ePathBsdfSample;		
		}
	}
}

LAUNCH_BOUNDS
__global__ void SampleBsdfs(PathState paths, int numPaths)
{
	const int i = getGlobalIndex();

	{
		if (paths.mode[i] == ePathBsdfSample)
		{	
			const Vec3 p = paths.pos[i];
			const Vec3 n = paths.normal[i];

			const Vec3 rayDir = paths.rayDir[i];

			const Primitive* hit = paths.primitive[i];

			Random& rand = paths.rand[i];

			float etaI = paths.etaI[i];
			float etaO = paths.etaO[i];

			// integrate indirect light by sampling BRDF
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

			Vec3 bsdfDir;
			BSDFType bsdfType;
			float bsdfPdf;

			BSDFSample(hit->material, etaI, etaO, p, u, v, n, -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);

            if (bsdfPdf <= 0.0f)
           	{
           		paths.mode[i] = ePathTerminate;
           	}
           	else
           	{
	            // reflectance
	            Vec3 f = BSDFEval(hit->material, etaI, etaO, p, n, -rayDir, bsdfDir);

	            // update ray medium if we are transmitting through the material
	            if (Dot(bsdfDir, n) <= 0.0f)
	            {
	            	paths.etaI[i] = etaO;
	            	paths.bsdfType[i] = eTransmitted;
					
	            	if (etaI != 1.0f)
	            	{
	            		// entering a medium, update the aborption (assume zero in air)
						paths.absorption[i] = hit->material.absorption;
					}
	            }
	            else
	            {
	            	paths.bsdfType[i] = eReflected;
	            }

	            // update throughput with primitive reflectance
	            paths.pathThroughput[i] *= f * Abs(Dot(n, bsdfDir))/bsdfPdf;
	            paths.bsdfPdf[i] = bsdfPdf;
	            paths.bsdfType[i] = bsdfType;
	            paths.rayDir[i] = bsdfDir;
	            paths.rayOrigin[i] = p + FaceForward(n, bsdfDir)*kRayEpsilon;
	            paths.mode[i] = ePathAdvance;

	        }
        }
    }
}

LAUNCH_BOUNDS
__global__ void SampleProbes(PathState paths, int numPaths)
{

}

LAUNCH_BOUNDS
__global__ void AdvancePaths(GPUScene scene, PathState paths, int numPaths)
{
	const int i = getGlobalIndex();

	{
		if (paths.mode[i] == ePathAdvance)
		{
			Vec3 rayOrigin = paths.rayOrigin[i];
			Vec3 rayDir = paths.rayDir[i];
			float rayTime = paths.rayTime[i];
			float etaI = paths.etaI[i];

			Vec3 pathThroughput = paths.pathThroughput[i];

			Vec3 n;
			float t;
			const Primitive* hit;

	        // find closest hit
	        if (Trace(scene, rayOrigin, rayDir, rayTime, t, n, &hit))
	        {	
				float etaO;

	        	// index of refraction for transmission, 1.0 corresponds to air
				if (etaI == 1.0f)
				{
	        		etaO = hit->material.GetIndexOfRefraction();
				}
				else
				{
					// returning to free space
					etaO = 1.0f;
				}

				pathThroughput *= Exp(-paths.absorption[i]*t);

				if (paths.depth[i] == 0)
				{
					// first trace is our only chance to add contribution from directly visible light sources        
					paths.totalRadiance[i] += hit->material.emission;
				}			
				else if (kBsdfSamples > 0)
				{
					// area pdf that this dir was already included by the light sampling from previous step
					float lightArea = PrimitiveArea(*hit);

					if (lightArea > 0.0f)
					{
						// convert to pdf with respect to solid angle
						float lightPdf = ((1.0f/lightArea)*t*t)/Clamp(Dot(-rayDir, n), 1.e-3f, 1.0f);

						// calculate weight for bsdf sampling
						int N = hit->lightSamples+kBsdfSamples;
						float cbsdf = kBsdfSamples/N;
						float clight = float(hit->lightSamples)/N;
						float weight = cbsdf*paths.bsdfPdf[i]/(cbsdf*paths.bsdfPdf[i] + clight*lightPdf);
						
						// specular paths have zero chance of being included by direct light sampling (zero pdf)
						if (paths.bsdfType[i] == eSpecular)
							weight = 1.0f;

						// pathThroughput already includes the bsdf pdf
						paths.totalRadiance[i] += weight*pathThroughput*hit->material.emission;
					}
				}

				// terminate ray if we hit a light source
				if (hit->lightSamples)
				{
					paths.mode[i] = ePathTerminate;
				}
				else
				{
					// update throughput based on absorption through the medium
					paths.pos[i] = rayOrigin + rayDir*t;
					paths.normal[i] = n;
					paths.primitive[i] = hit;
					paths.etaO[i] = etaO;
					paths.pathThroughput[i] = pathThroughput;
					paths.depth[i] += 1;

					paths.mode[i] = ePathLightSample;
				}
			}
			else
			{
				// todo: sky 

				// no hit, terminate path
				paths.mode[i] = ePathTerminate;
			}
		}
	}
}

LAUNCH_BOUNDS
__global__ void GeneratePaths(Camera camera, CameraSampler sampler, Tile tile, int seed, PathState paths, int numPaths)
{
	const int tx = blockIdx.x*blockDim.x;
	const int ty = blockIdx.y*blockDim.y;

	const int x = tx + threadIdx.x + tile.x;
	const int y = ty + threadIdx.y + tile.y;

	const int i = getGlobalIndex();

	{
		if (paths.mode[i] == ePathGenerate || paths.mode[i] == ePathDisabled || paths.mode[i] == ePathTerminate)
		{
			// if we're inside the tile
			if (threadIdx.x < tile.width && threadIdx.y < tile.height)
			{
				Random rand(i + tile.y*tile.width + tile.x + seed);

				// offset
				//float x, y, t;
				//StratifiedSample2D(i, tile.width, tile.height, rand, x, y);

				float t;
				StratifiedSample1D(i, 64, rand, t);

				// shutter time
				float time = Lerp(camera.shutterStart, camera.shutterEnd, t);
				
				//float px = tile.x + x*tile.width;
				//float py = tile.y + y*tile.height;
				float px = x + rand.Randf(-0.5f, 0.5f);
				float py = y + rand.Randf(-0.5f, 0.5f);

				Vec3 origin, dir;
				sampler.GenerateRay(px, py, origin, dir);

				// advance paths
				paths.depth[i] = 0;
				paths.rayOrigin[i] = origin;
				paths.rayDir[i] = dir;
				paths.rayTime[i] = time;
				paths.mode[i] = ePathAdvance;
				paths.rand[i] = rand;
				paths.totalRadiance[i] = 0.0f;
				paths.pathThroughput[i] = 1.0f;
				paths.etaI[i] = 1.0f;
				paths.bsdfType[i] = eReflected;
				paths.bsdfPdf[i] = 1.0f;
				paths.rasterX[i] = px;
				paths.rasterY[i] = py;

			}
			else
			{
				paths.mode[i] = ePathDisabled;
			}
		}
	}
}

//LAUNCH_BOUNDS
__global__ void VisualizeNormals(GPUScene scene, PathState paths, int numPaths)
{
	const int i = getGlobalIndex();

	if (i < numPaths)
	{
		Vec3 rayOrigin = paths.rayOrigin[i];
		Vec3 rayDir = paths.rayDir[i];

		Vec3 n;
		float t;
		
	    // find closest hit
	    if (Trace(scene, rayOrigin, rayDir, 0.0f, t, n, NULL))
	    {	
			paths.totalRadiance[i] = n;
		}

		paths.mode[i] = ePathTerminate;
	}
}


struct GpuWaveFrontRenderer : public Renderer
{
	Color* output = NULL;
	
	GPUScene sceneGPU;
	
	Random rand;

	int tileWidth;
	int tileHeight;

	PathState paths;

	// map id to geometry struct
	std::map<int, MeshGeometry> gpuMeshes;

	GpuWaveFrontRenderer(const Scene* s)
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

		// convert scene BVH
		CreateVec4Texture((Vec4**)&(sceneGPU.bvh.nodes), (Vec4*)s->bvh.nodes, sizeof(BVHNode)*s->bvh.numNodes);
		sceneGPU.bvh.numNodes = s->bvh.numNodes;

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

		tileWidth = 1024;
		tileHeight = 1024;

		const int numPaths = tileWidth*tileHeight;

		// allocate paths
		//cudaMalloc(&paths, sizeof(PathState)*numPaths);
		//cudaMemset(paths, 0, sizeof(PathState)*numPaths);

		paths = AllocatePaths(numPaths);
	}

	virtual ~GpuWaveFrontRenderer()
	{
		cudaFree(output);
		cudaFree(sceneGPU.primitives);
		cudaFree(sceneGPU.lights);
		
		FreePaths(paths);
	}
	
	void Init(int width, int height)
	{
		cudaFree(output);
		cudaMalloc(&output, sizeof(Color)*width*height);
		cudaMemset(output, 0, sizeof(Color)*width*height);
	}

	void Render(const Camera& camera, const Options& options, Color* outputHost)
	{
		std::vector<Tile> tiles;

		const int tilesx = (options.width + tileWidth - 1)/tileWidth;
		const int tilesy = (options.height + tileHeight - 1)/tileHeight;

		for (int y=0; y < tilesy; ++y)
		{
			for (int x=0; x < tilesx; ++x)
			{
				Tile tile;
				tile.x = x*tileWidth;
				tile.y = y*tileHeight;

				tile.width = Min(tileWidth, options.width-tile.x);
				tile.height = Min(tileHeight, options.height-tile.y);

				tiles.push_back(tile);
			}
		}

		const int numPaths = tileWidth*tileHeight;

		// create a sampler for the camera
		CameraSampler sampler(
			Transform(camera.position, camera.rotation),
			camera.fov, 
			0.001f,
			1.0f,
			options.width,
			options.height);

		
		for (int tileIndex=0; tileIndex < tiles.size(); ++tileIndex)
		{
			Tile tile = tiles[tileIndex];

			// a tile consists of many thread blocks
			const int blockWidth = 16;
			const int blockHeight = 16;

			const int gridWidth = (tile.width + blockWidth - 1)/blockWidth;
			const int gridHeight = (tile.height + blockHeight - 1)/blockHeight;

			dim3 blockDim(blockWidth, blockHeight);
			dim3 gridDim(gridWidth, gridHeight);

			/*
			const int kNumThreadsPerBlock = 256;
			const int kNumBlocks = (numPaths + kNumThreadsPerBlock - 1)/kNumThreadsPerBlock;
			
			dim3 gridDim(kNumBlocks);
			dim3 blockDim(kNumThreadsPerBlock);
			*/

			GeneratePaths<<<gridDim, blockDim>>>(camera, sampler, tile, rand.Rand(), paths, numPaths);
	
			if (options.mode == eNormals)
			{
				VisualizeNormals<<<gridDim, blockDim>>>(sceneGPU, paths, numPaths);
			}
			else
			{
				for (int i=0; i < options.maxDepth; ++i)
				{
					AdvancePaths<<<gridDim, blockDim>>>(sceneGPU, paths, numPaths);
					SampleLights<<<gridDim, blockDim>>>(sceneGPU, paths, numPaths);
					//SampleProbes();
					SampleBsdfs<<<gridDim, blockDim>>>(paths, numPaths);
				}
			}
			

			TerminatePaths<<<gridDim, blockDim>>>(output, options, paths, numPaths);
		}

		// copy back to output
		cudaMemcpy(outputHost, output, sizeof(Color)*options.width*options.height, cudaMemcpyDeviceToHost);
	}
};


Renderer* CreateGpuWavefrontRenderer(const Scene* s)
{
	return new GpuWaveFrontRenderer(s);
}
