#include "render.h"
#include "intersection.h"
#include "util.h"
#include "disney.h"
//#include "lambert.h"


#define kBrdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.001f

// trace a ray against the scene returning the closest intersection
inline bool Trace(const Scene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
	// disgard hits closer than this distance to avoid self intersection artifacts
	float minT = REAL_MAX;
	const Primitive* closestPrimitive = NULL;
	Vec3 closestNormal(0.0f);

	for (Scene::PrimitiveArray::const_iterator iter=scene.primitives.begin(), end=scene.primitives.end(); iter != end; ++iter)
	{
		float t;
		Vec3 n;

		const Primitive& primitive = *iter;

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



inline Color SampleLights(const Scene& scene, const Primitive& surfacePrimitive, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& wo, float time, Random& rand)
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

				if (weight > 0.0f)
					sum += weight*skyColor*f*Abs(Dot(wi, surfaceNormal))/skyPdf;
			}
		}

		if (kProbeSamples > 0)
			sum /= float(kProbeSamples);
	}

	for (int i=0; i < int(scene.primitives.size()); ++i)
	{
		// assume all lights are area lights for now
		const Primitive& lightPrimitive = scene.primitives[i];

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
Color PathTrace(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir, float time, int maxDepth, Random& rand)
{	
    // path throughput
    Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Color totalRadiance(0.0f);

    Vec3 rayOrigin = startOrigin;
    Vec3 rayDir = startDir;
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

			//totalRadiance = Color(u*0.5f + 0.5f, 1.0f);
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
			}
		
       		totalRadiance += weight*scene.sky.Eval(rayDir)*pathThroughput; 
			break;
        }
    }

    return totalRadiance;
}

struct CpuRenderer : public Renderer
{
	CpuRenderer(const Scene* s) : scene(s) {}

	const Scene* scene;
	Random rand;

	void AddSample(Color* output, int width, int height, float rasterX, float rasterY, float clamp, const Filter& filter, const Color& sample)
	{
		switch (filter.type)		
		{
			case eFilterBox:
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
						float w = 1.0f;

						output[y*width+x] += c*w;
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

				Color c =  ClampLength(sample, clamp);
				c.w = 1.0f;

				for (int x=startX; x <= endX; ++x)
				{
					for (int y=startY; y <= endY; ++y)
					{
						float w = filter.Eval(x-rasterX, y-rasterY);

						output[y*width+x] += c*w;
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
							const float time = rand.Randf(camera.shutterStart, camera.shutterEnd);
							const float x = i + rand.Randf(-0.5f, 0.5f) + 0.5f;
							const float y = j + rand.Randf(-0.5f, 0.5f) + 0.5f;

							sampler.GenerateRay(x, y, origin, dir);

							Color sample = PathTrace(*scene, origin, dir, time, options.maxDepth, rand);

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
