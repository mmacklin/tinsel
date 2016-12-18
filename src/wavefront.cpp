#include "render.h"
#include "intersection.h"
#include "util.h"
#include "sampler.h"
#include "disney.h"
//#include "lambert.h"


#define kBsdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.0001f


namespace
{

// trace a ray against the scene returning the closest intersection
inline bool Trace(const Scene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{

#if 1

	struct Callback
	{
		float minT;
		Vec3 closestNormal;
		const Primitive* closestPrimitive;

		Ray ray;
		const Scene& scene;

		Callback(const Scene& s, const Ray& r) : minT(REAL_MAX), closestPrimitive(NULL), ray(r), scene(s)
		{

		}
		
		void operator()(int index)
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
	*outPrimitive = callback.closestPrimitive;

	return callback.closestPrimitive != NULL;
	
#else


	// disgard hits closer than this distance to avoid self intersection artifacts
	float minT = REAL_MAX;
	const Primitive* closestPrimitive = NULL;
	Vec3 closestNormal(0.0f);

	for (Scene::PrimitiveArray::const_iterator iter=scene.primitives.begin(), end=scene.primitives.end(); iter != end; ++iter)
	{
		float t;
		Vec3 n, ns;

		const Primitive& primitive = *iter;

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
	
	outT = minT;		
	outNormal = FaceForward(closestNormal, -ray.dir);
	*outPrimitive = closestPrimitive;

	return closestPrimitive != NULL;

#endif

	
}


inline Vec3 SampleLights(const Scene& scene, const Primitive& surfacePrimitive, float etaI, float etaO, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& shadingNormal, const Vec3& wo, float time, Random& rand)
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

	for (int i=0; i < scene.primitives.size(); ++i)
	{
		// assume all lights are area lights for now
		const Primitive& lightPrimitive = scene.primitives[i];

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
					float lightArea = PrimitiveArea(lightPrimitive);
					float lightPdf = ((1.0f/lightArea)*tSq)/nl;

					// bsdf pdf for light's direction
					float bsdfPdf = BSDFPdf(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);
					Vec3 f = BSDFEval(surfacePrimitive.material, etaI, etaO, surfacePos, shadingNormal, wo, wi);

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
	Vec3 rayOrigin;
	Vec3 rayDir;
	float rayTime;

	Vec3 pos;
	Vec3 normal;

	int depth;

	Vec3 pathThroughput;
	Vec3 absorption;
	const Primitive* primitive;

	Vec3 totalRadiance;

	float etaI;
	float etaO;

	PathMode mode;

	// pdf from last brdf sampling
	float bsdfPdf;
	BSDFType bsdfType;

	// sample coordinate
	float rasterX;
	float rasterY;

	Random rand;
};



void TerminatePaths(Color* output, Options options, PathState* paths, int numPaths)
{
	for (int i=0; i < numPaths; ++i)
	{
		if (paths[i].mode != ePathDisabled)
		{
			float rasterX = paths[i].rasterX;
			float rasterY = paths[i].rasterY;

			Vec3 sample = paths[i].totalRadiance;

			// sample = paths[i].normal*0.5f + 0.5f;

			int width = options.width;
			int height = options.height;

			switch (options.filter.type)		
			{
				case eFilterBox:
				{		
					int startX = Max(0, int(rasterX - options.filter.width));
					int startY = Max(0, int(rasterY - options.filter.width));
					int endX = Min(int(rasterX + options.filter.width), width-1);
					int endY = Min(int(rasterY + options.filter.width), height-1);

					Vec3 c =  ClampLength(sample, options.clamp);

					for (int x=startX; x <= endX; ++x)
					{
						for (int y=startY; y <= endY; ++y)
						{
							output[y*width+x] += Color(c, 1.0f);
						}
					}

					break;
				}
				case eFilterGaussian:
				{
					int startX = Max(0, int(rasterX - options.filter.width));
					int startY = Max(0, int(rasterY - options.filter.width));
					int endX = Min(int(rasterX + options.filter.width), width-1);
					int endY = Min(int(rasterY + options.filter.width), height-1);

					Vec3 c =  ClampLength(sample, options.clamp);

					for (int x=startX; x <= endX; ++x)
					{
						for (int y=startY; y <= endY; ++y)
						{
							float w = options.filter.Eval(x-rasterX, y-rasterY);

							output[y*width+x] += Color(c*w, w);
						}
					}
					break;
				}
			};
		}

		paths[i].mode = ePathGenerate;
	}
}

void SampleLights(const Scene& scene, PathState* paths, int numPaths)
{
	for (int i=0; i < numPaths; ++i)
	{
		if (paths[i].mode == ePathLightSample)
		{
        	// calculate a basis for this hit point
        	const Primitive* hit = paths[i].primitive;        	
        	
        	float etaI = paths[i].etaI;
        	float etaO = paths[i].etaO;

			const Vec3 rayDir = paths[i].rayDir;
            float rayTime = paths[i].rayTime;

            const Vec3 p = paths[i].pos;
            const Vec3 n = paths[i].normal;

			// integrate direct light over hemisphere
			paths[i].totalRadiance += paths[i].pathThroughput*SampleLights(scene, *hit, etaI, etaO, p, n, n, -rayDir, rayTime, paths[i].rand);			

			paths[i].mode = ePathBsdfSample;		
		}
	}
}

void SampleBsdfs(PathState* paths, int numPaths)
{
	for (int i=0; i < numPaths; ++i)
	{
		if (paths[i].mode == ePathBsdfSample)
		{	
			const Vec3 p = paths[i].pos;
			const Vec3 n = paths[i].normal;

			const Vec3 rayDir = paths[i].rayDir;

			const Primitive* hit = paths[i].primitive;

			Random& rand = paths[i].rand;

			float etaI = paths[i].etaI;
			float etaO = paths[i].etaO;

			// integrate indirect light by sampling BRDF
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

			Vec3 bsdfDir;
			BSDFType bsdfType;
			float bsdfPdf;

			BSDFSample(hit->material, etaI, etaO, p, u, v, n, -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);

            if (bsdfPdf <= 0.0f)
           	{
           		paths[i].mode = ePathTerminate;
           	}
           	else
           	{
	            // reflectance
	            Vec3 f = BSDFEval(hit->material, etaI, etaO, p, n, -rayDir, bsdfDir);

	            // update ray medium if we are transmitting through the material
	            if (Dot(bsdfDir, n) <= 0.0f)
	            {
	            	paths[i].etaI = etaO;
	            	paths[i].bsdfType = eTransmitted;
					
	            	if (etaI != 1.0f)
	            	{
	            		// entering a medium, update the aborption (assume zero in air)
						paths[i].absorption = hit->material.absorption;
					}
	            }
	            else
	            {
	            	paths[i].bsdfType = eReflected;
	            }

	            // update throughput with primitive reflectance
	            paths[i].pathThroughput *= f * Abs(Dot(n, bsdfDir))/bsdfPdf;
	            paths[i].bsdfPdf = bsdfPdf;
	            paths[i].bsdfType = bsdfType;
	            paths[i].rayDir = bsdfDir;
	            paths[i].rayOrigin = p + FaceForward(n, bsdfDir)*kRayEpsilon;
	            paths[i].mode = ePathAdvance;

	        }
        }
    }
}

void SampleProbes(PathState* paths, int numPaths)
{

}

void AdvancePaths(const Scene& scene, PathState* paths, int numPaths)
{
	for (int i=0; i < numPaths; ++i)
	{
		if (paths[i].mode == ePathAdvance)
		{
			Vec3 rayOrigin = paths[i].rayOrigin;
			Vec3 rayDir = paths[i].rayDir;
			float rayTime = paths[i].rayTime;
			float etaI = paths[i].etaI;

			Vec3 pathThroughput = paths[i].pathThroughput;

			Vec3 n;
			float t;
			const Primitive* hit;

	        // find closest hit
	        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit))
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

				pathThroughput *= Exp(-paths[i].absorption*t);

				if (paths[i].depth == 0)
				{
					// first trace is our only chance to add contribution from directly visible light sources        
					paths[i].totalRadiance += hit->material.emission;
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
						float weight = cbsdf*paths[i].bsdfPdf/(cbsdf*paths[i].bsdfPdf + clight*lightPdf);
						
						// specular paths have zero chance of being included by direct light sampling (zero pdf)
						if (paths[i].bsdfType == eSpecular)
							weight = 1.0f;

						// pathThroughput already includes the bsdf pdf
						paths[i].totalRadiance += weight*pathThroughput*hit->material.emission;
					}
				}

				// terminate ray if we hit a light source
				if (hit->lightSamples)
				{
					paths[i].mode = ePathTerminate;
				}
				else
				{
					// update throughput based on absorption through the medium
					paths[i].pos = rayOrigin + rayDir*t;
					paths[i].normal = n;
					paths[i].primitive = hit;
					paths[i].etaO = etaO;
					paths[i].pathThroughput = pathThroughput;
					paths[i].depth += 1;

					paths[i].mode = ePathLightSample;
				}
			}
			else
			{
				// todo: sky 

				// no hit, terminate path
				paths[i].mode = ePathTerminate;
			}
		}
	}
}

void GeneratePaths(Camera camera, CameraSampler sampler, Tile tile, int seed, PathState* paths, int numPaths)
{
	for (int i=0; i < numPaths; ++i)
	{
		if (paths[i].mode == ePathGenerate || paths[i].mode == ePathDisabled || paths[i].mode == ePathTerminate)
		{
			// if we're inside the tile
			if (i < tile.width*tile.height)
			{
				Random rand(i + tile.y*tile.width + tile.x + seed);

				// offset
				float x, y, t;
				StratifiedSample2D(i, tile.width, tile.height, rand, x, y);
				StratifiedSample1D(i, 64, rand, t);

				// shutter time
				float time = Lerp(camera.shutterStart, camera.shutterEnd, t);
				
				float px = tile.x + x*tile.width;
				float py = tile.y + y*tile.height;

				Vec3 origin, dir;
				sampler.GenerateRay(px, py, origin, dir);

				// advance paths
				paths[i].depth = 0;
				paths[i].rayOrigin = origin;
				paths[i].rayDir = dir;
				paths[i].rayTime = time;
				paths[i].mode = ePathAdvance;
				paths[i].rand = rand;
				paths[i].totalRadiance = 0.0f;
				paths[i].pathThroughput = 1.0f;
				paths[i].etaI = 1.0f;
				paths[i].bsdfType = eReflected;
				paths[i].bsdfPdf = 1.0f;
				paths[i].rasterX = px;
				paths[i].rasterY = py;

			}
			else
			{
				paths[i].mode = ePathDisabled;
			}
		}
	}
}

} // anonymous namespace

struct CpuWaveFrontRenderer : public Renderer
{
	int tileWidth;
	int tileHeight;

	PathState* paths;

	const Scene* scene;

	Random rand;

	CpuWaveFrontRenderer(const Scene* s) : scene(s) 
	{
		tileWidth = 32;
		tileHeight = 32;

		const int numPaths = tileWidth*tileHeight;

		// allocate paths
		paths = new PathState[numPaths];

		for (int i=0; i < numPaths; ++i)
			paths[i].mode = ePathGenerate;
	}

	virtual ~CpuWaveFrontRenderer()
	{
		delete[] paths;
	}


	void Render(const Camera& camera, const Options& options, Color* output)
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
			GeneratePaths(camera, sampler, tiles[tileIndex], rand.Rand(), paths, numPaths);
	
			for (int i=0; i < options.maxDepth; ++i)
			{
				AdvancePaths(*scene, paths, numPaths);
				SampleLights(*scene, paths, numPaths);
				//SampleProbes();
				SampleBsdfs(paths, numPaths);
			}
			

			TerminatePaths(output, options, paths, numPaths);
		}
	}
};


Renderer* CreateCpuWavefrontRenderer(const Scene* s)
{
	return new CpuWaveFrontRenderer(s);
}
