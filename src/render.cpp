#include "render.h"
#include "intersection.h"
#include "util.h"
#include "disney.h"
//#include "lambert.h"


// trace a ray against the scene returning the closest intersection
inline bool Trace(const Scene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
	// disgard hits closer than this distance to avoid self intersection artifacts
	const float kEpsilon = 0.00001f;

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



inline Color SampleLights(const Scene& scene, const Primitive& primitive, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& wo, float time, Random& rand)
{	
	Color sum(0.0f);

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
			float lightArea;

			Sample(lightPrimitive, time, lightPos, lightNormal, lightArea, rand);

			Vec3 wi = Normalize(lightPos-surfacePos);

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
				// todo: assumes no self-intersection, need to check we hit the actual sample point, not just the shape
				// otherwise we would double count the contribution from some areas of the shape (only possible if non-convex) 
				if (hit == &lightPrimitive)
				{
					const Color f = BRDFEval(primitive.material, surfacePos, surfaceNormal, wi, wo);

					// light pdf					
					const float lightPdf = 1.0f/lightArea;
				
					const float nl = Dot(lightNormal, -wi);

					L += f * hit->material.emission * (Clamp(Dot(wi, surfaceNormal), 0.0f, 1.0f)*nl/Max(1.e-3f, t*t*lightPdf));
				}
			}
		}
	
		sum += L * (1.0f/numSamples);
	}

	return sum;
}


// reference, no light sampling, uniform hemisphere sampling
Color PathTrace(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir, Random& rand)
{	
    // path throughput
    Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Color totalRadiance(0.0f);

    Vec3 rayOrigin = startOrigin;
    Vec3 rayDir = startDir;
	float rayTime = rand.Randf();

    float t = 0.0f;
    Vec3 n(rayDir);
    const Primitive* hit;

    const int kMaxPathDepth = 4;

    for (int i=0; i < kMaxPathDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit))
        {	
            // calculate a basis for this hit pointq
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t;

#if 1
    		// if we hit a light then terminate and return emission
			// first trace is our only chance to add contribution from directly visible light sources
            if (i == 0)
            {
				totalRadiance += hit->material.emission;
            }
            else
			{
				totalRadiance += 0.25*pathThroughput*hit->material.emission;
			}

	    	// integral of Le over hemisphere
        	totalRadiance += 0.75*pathThroughput*SampleLights(scene, *hit, p, n, -rayDir, rayTime, rand);
#else

            totalRadiance += pathThroughput*hit->material.emission;
#endif

            // update position and path direct  ion
            //const Vec3 outDir = Mat33(u, v, n)*UniformSampleHemisphere(rand);
            Vec3 outDir;
            float outPdf;
            BRDFSample(hit->material, p, Mat33(u, v, n), -rayDir, outDir, outPdf, rand);

            // reflectance
            Color f = BRDFEval(hit->material, p, n, -rayDir, outDir);

            if (outPdf == 0.0f)
            	break;

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, outDir), 0.0f, 1.0f) / outPdf;

            // update path direction
            rayDir = outDir;
            rayOrigin = p;
        }
        else
        {
            // hit nothing, terminate loop
        	//totalRadiance += pathThroughput*scene.sky;
        	totalRadiance += pathThroughput*scene.sky.Eval(rayDir);
            break;
        }
    }

    return totalRadiance;
}

void Validate(const Color& c)
{
	assert(isfinite(c.x) && isfinite(c.y) && isfinite(c.z));
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
				int x = Min(width-1, int(rasterX));
				int y = Min(height-1, int(rasterY));

				output[(height-1-y)*width+x] += Color(sample.x, sample.y, sample.z, 1.0f);
				/*
				Color old = output[(height-1-y)*width+x];
				float oldCount = old.w;
				float newCount = oldCount + 1.0f;

				output[(height-1-y)*width+x].x += (sample.x - old.x) / newCount;
				output[(height-1-y)*width+x].y += (sample.y - old.y) / newCount;
				output[(height-1-y)*width+x].z += (sample.z - old.z) / newCount;
				output[(height-1-y)*width+x].w = newCount;
				*/

				return;
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
						
						output[(height-1-y)*width+x] += Color(Min(sample.x, clamp), Min(sample.y, clamp), Min(sample.z, clamp), 1.0f)*w;						
					}
				}
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

		for (int k=0; k < options.numSamples; ++k)
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
							const float x = i + rand.Randf(-0.5f, 0.5f) + 0.5f;
							const float y = j + rand.Randf(-0.5f, 0.5f) + 0.5f;

							sampler.GenerateRay(x, y, origin, dir);

							//origin = Vec3(0.0f, 1.0f, 5.0f);
							//dir = Normalize(-origin);

							Color sample = PathTrace(*scene, origin, dir, rand);

							Validate(sample);

							AddSample(output, options.width, options.height, x, y, options.clamp, options.filter, sample);

							break;
						}
						case eNormals:
						{
							const float x = i + 0.5f;
							const float y = j + 0.5f;

							sampler.GenerateRay(x, y, origin, dir);

							const Primitive* p;
							float t;
							Vec3 n;

							if (Trace(*scene, Ray(origin, dir, 1.0f), t, n, &p))
							{
								n = n*0.5f+0.5f;
								output[(options.height-1-j)*options.width+i] = Color(n.x, n.y, n.z, 1.0f);
							}
							else
							{
								output[(options.height-1-j)*options.width+i] = Color(0.0f);
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
