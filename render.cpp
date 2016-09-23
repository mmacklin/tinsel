#include "render.h"
#include "intersection.h"
#include "disney.h"


// trace a ray against the scene returning the closest intersection
inline bool Trace(const Scene& scene, const Ray& ray, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
{
	// disgard hits closer than this distance to avoid self intersection artifacts
	const float kEpsilon = 0.001f;

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
		
		// skip non-emitting primitives, todo: make this an explicit
		if (!lightPrimitive.light)
			continue;

		Color L(0.0f);

		const int numSamples = 1;

		for (int s=0; s < numSamples; ++s)
		{
			// sample light source
			Vec3 lightPos;
			float lightArea;

			Sample(lightPrimitive, time, lightPos, lightArea, rand);

			Vec3 wi = Normalize(lightPos-surfacePos);

			if (Dot(wi, surfaceNormal) < 0.0f)
				continue;

			// check visibility
			float t;
			Vec3 ln;
			const Primitive* hit;
			if (Trace(scene, Ray(surfacePos, wi, time), t, ln, &hit))
			{
				// did we hit a light prim (doesn't have to be the one we're sampling, useful for portals which don't themselves emit)
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
            // calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t;

    		// if we hit a light then terminate and return emission
			// first trace is our only chance to add contribution from directly visible light sources
            if (i == 0)
			{
				totalRadiance += hit->material.emission;
			}

    	    // integral of Le over hemisphere
            totalRadiance += pathThroughput*SampleLights(scene, *hit, p, n, -rayDir, rayTime, rand);

            // update position and path direction
            //const Vec3 outDir = Mat33(u, v, n)*UniformSampleHemisphere(rand);
            Vec3 outDir;
            float outPdf;
            BRDFSample(hit->material, p, Mat33(u, v, n), -rayDir, outDir, outPdf, rand);
			

            // reflectance
            Color f = BRDFEval(hit->material, p, n, -rayDir, outDir);

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, outDir), 0.0f, 1.0f) / outPdf;

            // update path direction
            rayDir = outDir;
            rayOrigin = p;
        }
        else
        {
            // hit nothing, terminate loop
        	totalRadiance += pathThroughput*scene.sky;
            break;
        }
    }

    return totalRadiance;
}

// reference, no light sampling, uniform hemisphere sampling
Color ForwardTraceUniform(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir, Random& rand)
{	
    // path throughput
    Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Color totalRadiance(0.0f);

    Vec3 rayOrigin = startOrigin;
    Vec3 rayDir = startDir;
	float rayTime = 1.0f;

    float t = 0.0f;
    Vec3 n(rayDir);
    const Primitive* hit;

    const int kMaxPathDepth = 4;

    for (int i=0; i < kMaxPathDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit))
        {	

            // calculate a basis for this hit point
            Vec3 u, v;
            BasisFromVector(n, &u, &v);

            const Vec3 p = rayOrigin + rayDir*t;

	        totalRadiance += hit->material.emission * pathThroughput;

            // update position and path direction
            const Vec3 outDir = Mat33(u, v, n)*UniformSampleHemisphere(rand);

            // reflectance
            //Color f = BlinnBRDF(n, -rayDir, outDir, hit->material.reflectance, hit->material.shininess);
            Color f = BRDFEval(hit->material, p, n, -rayDir, outDir);

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, outDir), 0.0f, 1.0f) / kInv2Pi;

            // update path direction
            rayDir = outDir;
            rayOrigin = p;
        }
        else
        {
            // hit nothing, terminate loop
            break;
        }
    }

    return totalRadiance;
}



struct CpuRenderer : public Renderer
{
	CpuRenderer(const Scene* s) : scene(s), clamp(FLT_MAX){}

	const Scene* scene;
	Random rand;

	float clamp;

	void AddSample(Color* output, int width, int height, float rasterX, float rasterY, const Filter& filter, const Color& sample)
	{
		switch (filter.type)		
		{
			case eFilterBox:
			{		
				int x = int(rasterX);
				int y = int(rasterY);

				output[(height-1-y)*width+x] = sample;
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

	void Render(Camera* camera, Color* output, int width, int height, int samplesPerPixel, Filter filter, RenderMode mode)
	{
		for (int k=0; k < samplesPerPixel; ++k)
		{
			for (int j=0; j < height; ++j)
			{
				for (int i=0; i < width; ++i)
				{
					Vec3 origin;
					Vec3 dir;

					// generate a ray         
					switch (mode)
					{
						case ePathTrace:
						{							
							const float x = i + rand.Randf(-0.5f, 0.5f) + 0.5f;
							const float y = j + rand.Randf(-0.5f, 0.5f) + 0.5f;

							GenerateRay(*camera, x, y, origin, dir);

							Color sample = PathTrace(*scene, origin, dir, rand);

							AddSample(output, width, height, x, y, filter, sample);

							break;
						}
						case eNormals:
						{
							GenerateRayNoJitter(*camera, i, j, origin, dir);

							const Primitive* p;
							float t;
							Vec3 n;

							if (Trace(*scene, Ray(origin, dir, 1.0f), t, n, &p))
							{
								n = n*0.5f+0.5f;
								output[(height-1-j)*width+i] = Color(n.x, n.y, n.z, 1.0f);
							}
							else
							{
								output[(height-1-j)*width+i] = Color(0.0f);
							}
							break;
						}
						case eComplexity:
						{
	                		/*
							job.camera.GenerateRayNoJitter(i, j, origin, dir);

							const Primitive* p;
							job.scene->Trace(origin, dir, t, n, &p);

							// visualise traversal
							job.output[(g_height-1-j)*g_width+i] = Color(AABBTree::GetTraceDepth() / 100.0f);
							*/
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
