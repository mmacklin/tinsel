/*
# Copyright Disney Enterprises, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# and the following modification to it: Section 6 Trademarks.
# deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the
# trade names, trademarks, service marks, or product names of the
# Licensor and its affiliates, except as required for reproducing
# the content of the NOTICE file.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Adapted to C++ by Miles Macklin 2016

*/

#include "maths.h"
#include "pfm.h"

//#include "blinn.h"
#if 1

#define DISABLE_IMPORTANCE 0

CUDA_CALLABLE inline float sqr(float x) { return x*x; }

CUDA_CALLABLE inline float SchlickFresnel(float u)
{
    float m = Clamp(1-u, 0.0f, 1.0f);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

CUDA_CALLABLE inline float GTR1(float NDotH, float a)
{
    if (a >= 1) return 1/kPi;
    float a2 = a*a;
    float t = 1 + (a2-1)*NDotH*NDotH;
    return (a2-1) / (kPi*logf(a2)*t);
}

CUDA_CALLABLE inline float GTR2(float NDotH, float a)
{
    float a2 = a*a;
    float t = 1.0f + (a2-1.0f)*NDotH*NDotH;
    return a2 / (kPi * t*t);
}

CUDA_CALLABLE inline float GTR2_aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
{
    return 1 / ( kPi * ax*ay * sqr( sqr(HDotX/ax) + sqr(HDotY/ay) + NDotH*NDotH ));
}

CUDA_CALLABLE inline float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NDotv*NDotv;
    return 1/(NDotv + sqrtf(a + b - a*b));
}

CUDA_CALLABLE inline float BRDFPdf(const Material& mat, const Vec3& P, const Vec3& n, const Vec3& V, const Vec3& L)
{

#if DISABLE_IMPORTANCE

	return kInv2Pi;

#else

    const float a = Max(0.001f, mat.roughness);

	const Vec3 half = SafeNormalize(L+V);

	const float cosThetaHalf = Abs(Dot(half, n));
    const float pdfHalf = GTR2(cosThetaHalf, a)*cosThetaHalf;

    // calculate pdf for each method given outgoing light vector
    float pdfSpec = pdfHalf*0.25f/Max(1.e-4f, Abs(Dot (L, half)));
    assert(isfinite(pdfSpec));

    float pdfDiff = Abs(Dot(L, n))*kInvPi;
    assert(isfinite(pdfDiff));

    // weight pdfs equally
    return Lerp(pdfSpec, pdfDiff, 0.5f);
	
#endif

}

// generate an importance sampled brdf direction
CUDA_CALLABLE inline Vec3 BRDFSample(const Material& mat, const Vec3& P, const Mat33& frame, const Vec3& V, Random& rand)
{

#if DISABLE_IMPORTANCE
	
	return frame*UniformSampleHemisphere(rand);

#else

    Vec3 light;

    const float select = rand.Randf();

    float r1, r2;
    Sample2D(rand, r1, r2);

    if (select < 0.5f)
    {
        // sample diffuse
        light = frame*CosineSampleHemisphere(r1, r2);
    }
    else
    {
	    const float a = Max(0.001f, mat.roughness);

        const float phiHalf = r1*k2Pi;
        
        const float cosThetaHalf = sqrtf((1.0f-r2)/(1.0f + (sqr(a)-1.0f)*r2));      
        const float sinThetaHalf = sqrtf(Max(0.0f, 1.0f-sqr(cosThetaHalf)));
        const float sinPhiHalf = sinf(phiHalf);
        const float cosPhiHalf = cosf(phiHalf);

		Validate(cosThetaHalf);
		Validate(sinThetaHalf);
		Validate(sinPhiHalf);
		Validate(cosPhiHalf);

        Vec3 half = frame*Vec3(sinThetaHalf*cosPhiHalf, sinThetaHalf*sinPhiHalf, cosThetaHalf);
        
        // ensure half angle in same hemisphere as incoming light vector
        if (Dot(half, V) <= 0.0f)
            half *= -1.0f;

        light = 2.0f*Dot(V, half)*half - V;
    }

	return light;

#endif
}


CUDA_CALLABLE inline Color BRDFEval(const Material& mat, const Vec3& P, const Vec3& N, const Vec3& V, const Vec3& L)
{
    float NDotL = Dot(N,L);
    float NDotV = Dot(N,V);
    if (NDotL <= 0 || NDotV <= 0) return Color(0);

    Vec3 H = Normalize(L+V);
    float NDotH = Dot(N,H);
    float LDotH = Dot(L,H);

    Vec3 Cdlin = Vec3(mat.color);
    float Cdlum = .3*Cdlin[0] + .6*Cdlin[1]  + .1*Cdlin[2]; // luminance approx.

    Vec3 Ctint = Cdlum > 0.0f ? Cdlin/Cdlum : Vec3(1); // normalize lum. to isolate hue+sat
    Vec3 Cspec0 = Lerp(mat.specular*.08*Lerp(Vec3(1), Ctint, mat.specularTint), Cdlin, mat.metallic);
    Vec3 Csheen = Lerp(Vec3(1), Ctint, mat.sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
    float Fd90 = 0.5 + 2.0f * LDotH*LDotH * mat.roughness;
    float Fd = Lerp(1.0f, Fd90, FL) * Lerp(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LDotH*LDotH*mat.roughness;
    float Fss = Lerp(1.0f, Fss90, FL) * Lerp(1.0f, Fss90, FV);
    float ss = 1.25 * (Fss * (1.0f / (NDotL + NDotV) - .5) + .5);

    // specular
    //float aspect = sqrt(1-mat.anisotrokPic*.9);
    //float ax = Max(.001f, sqr(mat.roughness)/aspect);
    //float ay = Max(.001f, sqr(mat.roughness)*aspect);
    //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);
    float a = Max(0.001f, mat.roughness);
    float Ds = GTR2(NDotH, a);
    float FH = SchlickFresnel(LDotH);
    Vec3 Fs = Lerp(Cspec0, Vec3(1), FH);
    float roughg = sqr(mat.roughness*.5+.5);
    float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

    // sheen
    Vec3 Fsheen = FH * mat.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NDotH, Lerp(.1,.001, mat.clearcoatGloss));
    float Fr = Lerp(.04f, 1.0f, FH);
    float Gr = smithG_GGX(NDotL, .25) * smithG_GGX(NDotV, .25);

    Vec3 out = ((1/kPi) * Lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
        * (1-mat.metallic)
        + Gs*Fs*Ds + .25*mat.clearcoat*Gr*Fr*Dr;

    return Vec4(out, 0.0f);
}
#endif

inline void BRDFTest(Material mat, Mat33 frame, float woTheta, const char* filename)
{
    /* example code to visualize a BRDF, its PDF and sampling

    Material mat;
    mat.color = Color(0.95, 0.9, 0.9);
    mat.specular = 1.0;
    mat.roughness = 0.025;
    mat.metallic = 0.0;

    Vec3 n = Normalize(Vec3(1.0f, 0.0f, 0.0f));
    Vec3 u, v;
    BasisFromVector(n, &u, &v);
    
    BRDFTest(mat, Mat33(u, v, n), kPi/2.05f, "brdftest.pfm");
    */

    int width = 512;
    int height = 256;

    PfmImage image;
    image.width = width;
    image.height = height;
    image.depth = 1;

    image.data = new float[width*height*3];

    Vec3* pixels = (Vec3*)image.data;

    Vec3 wo = frame*Vec3(0.0f, -sinf(woTheta), cosf(woTheta));

    Random rand;

    for (int j=0; j < height; ++j)
    {
        for (int i=0; i < width; ++i)
        {
            float u = float(i)/width;
            float v = float(j)/height;

            Vec3 wi = ProbeUVToDir(Vec2(u,v));

            Color f = BRDFEval(mat, Vec3(0.0f), frame.GetCol(2), wo, wi); 
            float pdf = BRDFPdf(mat, Vec3(0.0f), frame.GetCol(2), wo, wi);

          //  f.x = u;
            //f.y = v;
            //f.z = 1.0;
        //    printf("%f %f %f\n", f.x, f.y, f.z);

            pixels[j*width + i] = Vec3(f.x, pdf, 0.5f);
        }
    }

    int numSamples = 1000;

    for (int i=0; i < numSamples; ++i)
    {
        Vec3 wi = BRDFSample(mat, Vec3(0.0f), frame, wo, rand);
            
        Vec2 uv = ProbeDirToUV(wi);

        int px = Clamp(int(uv.x*width), 0, width-1);
        int py = Clamp(int(uv.y*height), 0, height-1);

        pixels[py*width + px] = Vec3(1.0f, 0.0f, 0.0f);
    }

    PfmSave(filename, image);
}

