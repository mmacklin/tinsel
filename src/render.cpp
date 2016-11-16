#include "render.h"
#include "intersection.h"
#include "util.h"
#include "sampler.h"
#include "disney.h"
//#include "lambert.h"


#define kBsdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.0001f



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

// reference, no light sampling, uniform hemisphere sampling
Vec3 PathTrace(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir, float time, int maxDepth, Random& rand)
{	
    // path throughput
    Vec3 pathThroughput(1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Vec3 totalRadiance(0.0f, 0.0f, 0.0f);

    Vec3 rayOrigin = startOrigin;
    Vec3 rayDir = startDir;
	float rayTime = time;
	float rayEta = 1.0f;
	Vec3 rayAbsorption = 0.0f;
	BSDFType rayType = eReflected;

    float t = 0.0f;
    Vec3 n;
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
			pathThroughput *= Exp(-rayAbsorption*t);

#if 1
			if (i == 0)
			{
				// first trace is our only chance to add contribution from directly visible light sources        
				totalRadiance += hit->material.emission;
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
					float weight = cbsdf*bsdfPdf/(cbsdf*bsdfPdf+ clight*lightPdf);
					
					// specular paths have zero chance of being included by direct light sampling (zero pdf)
					if (rayType == eSpecular)
						weight = 1.0f;

					// pathThroughput already includes the bsdf pdf
					//totalRadiance += weight*pathThroughput*hit->material.emission;
					totalRadiance += weight*pathThroughput*hit->material.emission;
				}
			}

        	// calculate a basis for this hit point
            const Vec3 p = rayOrigin + rayDir*t;

			// integrate direct light over hemisphere
			totalRadiance += pathThroughput*SampleLights(scene, *hit, rayEta, outEta, p, n, n, -rayDir, rayTime, rand);			

			//totalRadiance = Color(u*0.5f + 0.5f, 1.0f);
#else

        	// calculate a basis for this hit point


            const Vec3 p = rayOrigin + rayDir*t;

			totalRadiance += pathThroughput*hit->material.emission;

#endif

			// terminate ray if we hit a light source
			if (hit->lightSamples)
				break;


			// integrate indirect light by sampling BRDF
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

			Vec3 bsdfDir;
			BSDFType bsdfType;
			BSDFSample(hit->material, rayEta, outEta, p, u, v, n, -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);

            if (bsdfPdf <= 0.0f)
            	break;

            // reflectance
            Vec3 f = BSDFEval(hit->material, rayEta, outEta, p, n, -rayDir, bsdfDir);

            // update ray medium if we are transmitting through the material
            if (Dot(bsdfDir, n) <= 0.0f)
            {
            	rayEta = outEta;
            	rayType = eTransmitted;
				rayAbsorption = outAbsorption;
            }
            else
            {
            	rayType = eReflected;
            }

            // update throughput with primitive reflectance
            pathThroughput *= f * Abs(Dot(n, bsdfDir))/bsdfPdf;

            // update path direction
			rayType = bsdfType;
            rayDir = bsdfDir;
            rayOrigin = p + FaceForward(n, bsdfDir)*kRayEpsilon;
        }
        else
        {
            // hit nothing, sample sky dome and terminate         
            float weight = 1.0f;

        	if (scene.sky.probe.valid && i > 0 && rayType != eSpecular)
        	{ 
        		// probability that this dir was already sampled by probe sampling
        		float skyPdf = ProbePdf(scene.sky.probe, rayDir);

				int N = kProbeSamples+kBsdfSamples;
				float cbsdf = kBsdfSamples/N;
				float csky = float(kProbeSamples)/N;
				
				weight = cbsdf*bsdfPdf/(cbsdf*bsdfPdf+ csky*skyPdf);
			}
		
       		totalRadiance += weight*scene.sky.Eval(rayDir)*pathThroughput; 
			break;
        }
    }

    return totalRadiance;
}

struct CpuRenderer : public Renderer
{
	CpuRenderer(const Scene* s) : scene(s) 
	{

	}

	const Scene* scene;

	Random rand;

	void AddSample(Color* output, int width, int height, float rasterX, float rasterY, float clamp, const Filter& filter, const Vec3& sample)
	{
		switch (filter.type)		
		{
			case eFilterBox:
			{		
				int startX = Max(0, int(rasterX - filter.width));
				int startY = Max(0, int(rasterY - filter.width));
				int endX = Min(int(rasterX + filter.width), width-1);
				int endY = Min(int(rasterY + filter.width), height-1);

				Vec3 c =  ClampLength(sample, clamp);

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
				int startX = Max(0, int(rasterX - filter.width));
				int startY = Max(0, int(rasterY - filter.width));
				int endX = Min(int(rasterX + filter.width), width-1);
				int endY = Min(int(rasterY + filter.width), height-1);

				Vec3 c =  ClampLength(sample, clamp);

				for (int x=startX; x <= endX; ++x)
				{
					for (int y=startY; y <= endY; ++y)
					{
						float w = filter.Eval(x-rasterX, y-rasterY);

						output[y*width+x] += Color(c*w, w);
					}
				}
				break;
			}
		};
	}

	void Render(const Camera& camera, const Options& options, Color* output)
	{
		// create a sampler for the camera
		CameraSampler sampler(
			Transform(camera.position, camera.rotation),
			camera.fov, 
			0.001f,
			1.0f,
			options.width,
			options.height);

		Random decorrelation;

		//for (int k=0; k < options.numSamples; ++k)
		{
			for (int j=0; j < options.height; ++j)
			{
				for (int i=0; i < options.width; ++i)
				{
					Vec3 origin;
					Vec3 dir;

					// generate a ray         
					switch (options.mode)
					{
						case ePathTrace:
						{							
							float x, y, t;

							Sample2D(rand, x, y);
							Sample1D(rand, t);

							float time = Lerp(camera.shutterStart, camera.shutterEnd, t);
							
							x += i;
							y += j;

							sampler.GenerateRay(x, y, origin, dir);

							Vec3 sample = PathTrace(*scene, origin, dir, time, options.maxDepth, rand);

							Validate(sample);

							AddSample(output, options.width, options.height, x, y, options.clamp, options.filter, sample);

							break;
						}
						case eNormals:
						{
							const float x = i;// + 0.5f;
							const float y = j;// + 0.5f;

							sampler.GenerateRay(x, y, origin, dir);

							const Primitive* p;
							float t;
							Vec3 n;

							if (Trace(*scene, Ray(origin, dir, 1.0f), t, n, &p))
							{
								n = n*0.5f+0.5f;
								output[j*options.width+i] = Color(n.x, n.y, n.z, 1.0f);
							}
							else
							{
								output[j*options.width+i] = Color(0.0f);
							}
							break;
						}
						case eComplexity:
						{
							break;
						}		
					}
				}
			}
		}
	}
};


Renderer* CreateCpuRenderer(const Scene* s)
{
	return new CpuRenderer(s);
}
