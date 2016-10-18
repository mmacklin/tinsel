#pragma once

#include "maths.h"

// cosTheta should be the angle between the wi and wh
CUDA_CALLABLE inline Color Schlick(const Color& c, float cosTheta)
{
	return c + (Color(1.0f, 1.0f, 1.0f)-c)*powf(1.0f-cosTheta, 5.0f);
}

CUDA_CALLABLE inline Vec3 SphericalDirection(float sinTheta, float cosTheta, float phi)
{
	return Vec3(sinTheta * cosf(phi),
              	   sinTheta * sinf(phi),
			  	   cosTheta);
}

CUDA_CALLABLE inline float Exponent(const Material& mat)
{
	return 10.0f/mat.roughness;
}

CUDA_CALLABLE inline float BRDFPdf(const Material& mat, const Vec3& P, const Vec3& n, const Vec3& V, const Vec3& L)
{
	Vec3 H = Normalize(V + L);
	float cosTheta = Abs(Dot(H, n));

	// Compute PDF for wi from Blinn distribution
	float pdf = ((Exponent(mat) + 1.f) * powf(cosTheta, Exponent(mat))) / (2.f * kPi * 4.f *  Abs(Dot(V, H)));
	return pdf;
}

CUDA_CALLABLE inline Vec3 BRDFSample(const Material& mat, const Vec3& P, const Mat33& frame, const Vec3& V, Random& rand)
{
	Vec3 wi;

	float u1 = rand.Randf();
	float u2 = rand.Randf();

	float costheta = powf(u1, 1.f / (Exponent(mat)+1.f));
	float sintheta = sqrtf(Max(0.f, 1.f - costheta*costheta));
	float phi = u2 * 2.f * kPi;
	Vec3 H = frame*SphericalDirection(sintheta, costheta, phi);
	
	if (Dot(V, H) < 0.f)
		H *= -1.0f;

	// Compute incident direction by reflecting about $\wh$
	wi = -V + 2.f * Dot(V, H) * H;

	return wi;
}

CUDA_CALLABLE inline Color BRDFEval(const Material& mat, const Vec3& P, const Vec3& n, const Vec3& V, const Vec3& L)
{
	// calculate half-angle
	Vec3 wh = Normalize(V+L);

	float NdotWh = Abs(Dot(wh, n));
	float NdotWo = Abs(Dot(V, n));
	float NdotWi = Abs(Dot(L, n));
	float WodotWh = Abs(Dot(V, wh));

	//if (Dot(wo, wi) < 0.0f)
		//return Color::kBlack;

	Color f = Schlick(mat.color, WodotWh);

	// geometric term
	float g = Min(1.0f, Min((2.0f * NdotWh * NdotWo / WodotWh),
					     	(2.0f * NdotWh *NdotWi / WodotWh)));

	float d = (Exponent(mat) + 2.0f) * kInv2Pi * powf(Abs(Dot(wh, n)), Exponent(mat));

	return f  * d * g / (4.0f * NdotWi * NdotWo + 1.e-4f);
}




