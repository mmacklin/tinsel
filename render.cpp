#include "render.h"
#include "intersection.h"


inline void GenerateRay(Camera& camera, int rasterX, int rasterY, Vec3& origin, Vec3& dir, Random& rand)
{
	float xoff = rand.Randf(-0.5f, 0.5f);
	float yoff = rand.Randf(-0.5f, 0.5f);

	Vec3 p = TransformPoint(camera.rasterToWorld, Vec3(float(rasterX) + 0.5f + xoff, float(rasterY) + 0.5f + yoff, 0.0f));

    origin = Vec3(camera.cameraToWorld.GetCol(3));
	dir = Normalize(p-origin);
}

inline void GenerateRayNoJitter(Camera& camera, int rasterX, int rasterY, Vec3& origin, Vec3& dir)
{
	Vec3 p = TransformPoint(camera.rasterToWorld, Vec3(float(rasterX) + 0.5f, float(rasterY) + 0.5f, 0.0f));

    origin = Vec3(camera.cameraToWorld.GetCol(3));
	dir = Normalize(p-origin);
}

//-------------

// cosTheta should be the angle between the wi and wh
inline Color Schlick(const Color& c, float cosTheta)
{
	return c + (Color(1.0f, 1.0f, 1.0f)-c)*powf(1.0f-cosTheta, 5.0f);
}

inline Color BlinnBRDF(const Vec3& n, const Vec3& wi, const Vec3& wo, const Color& reflectance, float exponent)
{
	// calculate half-angle
	Vec3 wh = Normalize(wi+wo);

	float NdotWh = Abs(Dot(wh, n));
	float NdotWo = Abs(Dot(wo, n));
	float NdotWi = Abs(Dot(wi, n));
	float WodotWh = Abs(Dot(wo, wh));

	//if (Dot(wo, wi) < 0.0f)
		//return Colour::kBlack;

	Color f = Schlick(reflectance, WodotWh);

	// geometric term
	float g = Min(1.0f, Min((2.0f * NdotWh * NdotWo / WodotWh),
					     	(2.0f * NdotWh *NdotWi / WodotWh)));

	float d = (exponent + 2.0f) * kInv2Pi * powf(Abs(Dot(wh, n)), exponent);

	return f  * d * g / (4.0f * NdotWi * NdotWo + 1.0e-4f);
}


Color LambertBRDF(const Vec3& n, const Vec3& wi, const Vec3& wo, const Color& reflectance)
{
	return reflectance*kInvPi;
}


float LambertPdf(const Vec3& wo, const Vec3& wi)
{
	float pdf = wi.z * kInvPi;

	if (pdf == 0.0f)
		pdf = REAL_MAX;	

	return pdf;
}

void LambertSample(const Vec3& n, const Vec3& woWorld, Vec3& wiWorld, float& pdf, Random& rand)
{
	/*
	// generate a sample on the hemisphere weighted by cosTheta term
	Vec3 wiLocal = CosineSampleHemisphere(rand);

	pdf = Pdf(Vec3(0.0), wiLocal);

	wiWorld = m_localToWorld * wiLocal;
	*/
}

inline Color BRDF(const Material& mat, const Vec3& p, const Vec3& n, const Vec3& wi, const Vec3& wo)
{
	return LambertBRDF(n, wi, wo, mat.reflectance);
}

inline void Sample(const Primitive& p, Vec3& pos, float& area, Random& rand)
{
	switch (p.type)
	{
		case eSphere:
		{
			pos = TransformPoint(p.transform, UniformSampleSphere(rand)*p.sphere.radius);
			
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

inline bool Intersect(const Primitive& p, const Vec3& rayOrigin, const Vec3& rayDir, float& t, Vec3* normal)
{
	switch (p.type)
	{
		case eSphere:
		{
			bool hit = IntersectRaySphere(Vec3(p.transform.GetCol(3)), p.sphere.radius, rayOrigin, rayDir, t, normal);
			return hit;
		}
		case ePlane:
		{
			bool hit = IntersectRayPlane(rayOrigin, rayDir, (const Vec4&)p.plane, t);
			if (hit && normal)
				*normal = (const Vec3&)p.plane;

			return hit;
		}
		case eMesh:
		{
			return false;
		}
	}
}


// trace a ray against the scene returning the closest intersection
inline bool Trace(const Scene& scene, const Vec3& rayOrigin, const Vec3& rayDir, float& outT, Vec3& outNormal, const Primitive** outPrimitive)
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

		if (Intersect(primitive, rayOrigin, rayDir, t, &n))
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



inline Color SampleLights(const Scene& scene, const Primitive& primitive, const Vec3& surfacePos, const Vec3& surfaceNormal, const Vec3& wo, Random& rand)
{	
	Color sum(0.0f);

	for (int i=0; i < scene.primitives.size(); ++i)
	{
		// assume all lights are area lights for now
		const Primitive& lightPrimitive = scene.primitives[i];
		
		// skip non-emitting primitives, todo: make this an explicit
		if (!lightPrimitive.light)
			continue;

		Color L(0.0f);

		const int numSamples = 10;

		for (int s=0; s < numSamples; ++s)
		{
			// sample light source
			Vec3 lightPos;
			float lightArea;

			Sample(lightPrimitive, lightPos, lightArea, rand);

			Vec3 wi = Normalize(lightPos-surfacePos);

			// check visibility
			float t;
			Vec3 ln;
			const Primitive* hit;
			if (Trace(scene, surfacePos, wi, t, ln, &hit))
			{
				// did we hit the light prim?
				if (hit == &lightPrimitive)
				{
					const Color f = BRDF(primitive.material, surfacePos, surfaceNormal, wi, wo);

					// light pdf
					const float nl = Clamp(Dot(ln, -wi), 0.0f, 1.0f);
					
					if (nl > 0.0)
					{
						const float lightPdf = (t*t) / (nl*lightArea);
					
						L += f * lightPrimitive.material.emission * Clamp(Dot(wi, surfaceNormal), 0.0f, 1.0f)  / lightPdf;
					}
				}
			}		
		}
	
		sum += L / numSamples;
	}

	return sum;
}




/*
Color PathTrace(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir)
{	
	// path throughput
	Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
	// accumulated radiance along the path
	Color totalRadiance(0.0f);

	Vec3 rayOrigin = startOrigin;
	Vec3 rayDir = startDir;

	float t = 0.0f;
	Vec3 n;
	const Primitive* hit;

	const int kMaxPathDepth = 4;

	for (int i=0; i < kMaxPathDepth; ++i)
	{
		// find closest hit
		if (Trace(scene, rayOrigin, rayDir, t, n, &hit))
		{			
			// update position and path direction
			Vec3 p = rayOrigin + rayDir*t;

			// first trace is our only chance to add contribution from directly visible light sources
            if (i == 0)
			{
				totalRadiance += hit->material.emission;
			}

			const BRDF* brdf = hit->material->GetBRDF(p, n);

			// sample light sources
			totalRadiance += pathThroughput * scene.SampleLights(p, n, -rayDir, brdf);
			
			// generate new path direction by sampling BRDF
			float pdf = 1.f;
			Vec3 wi;
			brdf->Sample(-rayDir, wi, pdf);
		
			// evaluate brdf
			Color f = brdf->F(-rayDir, wi);

			// update path throughput 
			pathThroughput *= f * Abs(Dot(n, wi)) / pdf;

			// update path start and direction
			rayOrigin = p;
			rayDir = wi;

			//delete brdf;
		}
            
		else
		{
			// hit nothing, evaluate background li and end loop
			totalRadiance += pathThroughput * scene.SampleSky(rayDir);						
			
			break;
		}
	}

	return totalRadiance;
}
*/

// reference, no light sampling, uniform hemisphere sampling
Color ForwardTraceExplicit(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir, Random& rand)
{	
    // path throughput
    Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
    // accumulated radiance
    Color totalRadiance(0.0f);

    Vec3 rayOrigin = startOrigin;
    Vec3 rayDir = startDir;

    float t = 0.0f;
    Vec3 n(rayDir);
    const Primitive* hit;

    const int kMaxPathDepth = 8;

    for (int i=0; i < kMaxPathDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, rayOrigin, rayDir, t, n, &hit))
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
            totalRadiance += SampleLights(scene, *hit, p, n, -rayDir, rand);

            // update position and path direction
            const Vec3 outDir = Mat33(u, v, n)*UniformSampleHemisphere(rand);

            // reflectance
            Color f = BRDF(hit->material, p, n, -rayDir, outDir);

            // update throughput with primitive reflectance
            pathThroughput *= f * Clamp(Dot(n, outDir), 0.0f, 1.0f) / kInv2Pi;

            // update path direction
            rayDir = outDir;
            rayOrigin = p;
        }
        else
        {
            // hit nothing, terminate loop
        	totalRadiance += pathThroughput*Vec4(0.02f, 0.2, 0.4f);
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

    float t = 0.0f;
    Vec3 n(rayDir);
    const Primitive* hit;

    const int kMaxPathDepth = 8;

    for (int i=0; i < kMaxPathDepth; ++i)
    {
        // find closest hit
        if (Trace(scene, rayOrigin, rayDir, t, n, &hit))
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
            Color f = BRDF(hit->material, p, n, -rayDir, outDir);

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


/*
// reference, no light sampling but does cosine weighted sampling
Color ForwardTraceImportance(const Scene& scene, const Vec3& startOrigin, const Vec3& startDir)
{	
	// path throughput
	Color pathThroughput(1.0f, 1.0f, 1.0f, 1.0f);
	// accumulated radiance
	Color totalRadiance(0.0f);

	Vec3 rayOrigin = startOrigin;
	Vec3 rayDir = startDir;

	float t = 0.0f;
	Vec3 n(rayDir);
	const Primitive* hit;

	const int kMaxPathDepth = 8;

	for (int i=0; i < kMaxPathDepth; ++i)
	{
		// find closest hit
		if (Trace(scene, rayOrigin, rayDir, t, n, &hit))
		{			
				// update position and path direction
			Vec3 p = rayOrigin + rayDir*t;

			totalRadiance += pathThroughput * hit->material.emission;

			const BRDF* brdf = hit->material->GetBRDF(p, n);

				// generate new path direction by sampling BRDF
			float pdf = 1.f;
			Vec3 wi;
			brdf->Sample(-rayDir, wi, pdf);
		
			// evaluate brdf
			Color f = brdf->F(-rayDir, wi);

			// update path throughput 
			pathThroughput *= f * Abs(Dot(n, wi)) / pdf;

			// update path start and direction
			rayOrigin = p;
			rayDir = wi;

			delete brdf;
		}
		else
		{
			// hit nothing, terminate loop
			break;
		}
	}

	return totalRadiance;
}

*/
/*

Color Whitted(const Scene& s, const Vec3& rayOrigin, const Vec3& rayDir)
{
	// TODO:

	return Color();
}


*/

Color Debug(const Scene& scene, const Vec3& rayOrigin, const Vec3& rayDir)
{
	// find closest hit
	float t;
	Vec3 n;
	const Primitive* p;
	if (Trace(scene, rayOrigin, rayDir, t, n, &p))
	{
		return Color(0.5f*n.x+0.5f, 0.5f*n.y+0.5f, 0.5f*n.z+0.5f, 1.0);
	}

	return Color(0.0f);
}


struct CpuRenderer : public Renderer
{
	CpuRenderer(const Scene* s) : scene(s) {}

	const Scene* scene;
	Random rand;

	void Render(Camera* camera, Color* output, int width, int height, int samplesPerPixel, RenderMode mode)
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
						GenerateRay(*camera, i, j, origin, dir, rand);

				        //output[(height-1-j)*width+i] += PathTrace(*scene, origin, dir);
				        output[(height-1-j)*width+i] += ForwardTraceExplicit(*scene, origin, dir, rand);
	                    break;
	                }
	                case eNormals:
	                {
						GenerateRayNoJitter(*camera, i, j, origin, dir);

	                    const Primitive* p;
	                    float t;
	                    Vec3 n;

	                    if (Trace(*scene, origin, dir, t, n, &p))
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
};


Renderer* CreateCpuRenderer(const Scene* s)
{
	return new CpuRenderer(s);
}
