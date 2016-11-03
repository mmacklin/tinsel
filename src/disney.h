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

CUDA_CALLABLE inline bool Refract(const Vec3 &wi, const Vec3 &n, float eta, Vec3& wt) {
    
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    float cosThetaI = Dot(n, wi);
    float sin2ThetaI = Max(0.0f, float(1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;

    float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vec3(n);
    return true;
}

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
    return 1 / ( kPi * ax*ay * Sqr( Sqr(HDotX/ax) + Sqr(HDotY/ay) + NDotH*NDotH ));
}

CUDA_CALLABLE inline float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NDotv*NDotv;
    return 1/(NDotv + sqrtf(a + b - a*b));
}


CUDA_CALLABLE inline float Fr(float VDotN, float etaI, float etaT)
{       
    float SinThetaT2 = Sqr(etaI/etaT)*(1.0f-VDotN*VDotN);
    
    // total internal reflection
    if (SinThetaT2 > 1.0f)
        return 1.0f;

    float LDotN = sqrtf(1.0f-SinThetaT2);

    // todo: reformulate to remove this division
    float eta = etaT/etaI;

    float r1 = (VDotN - eta*LDotN)/(VDotN + eta*LDotN);
    float r2 = (LDotN - eta*VDotN)/(LDotN + eta*VDotN);

    return 0.5f*(Sqr(r1) + Sqr(r2));
}


CUDA_CALLABLE inline float BRDFPdf(const Material& mat, float etaI, float etaO, const Vec3& P, const Vec3& n, const Vec3& V, const Vec3& L)
{
    if (Dot(L, n) <= 0.0f)
    {
        // transmission
        return 0.0f;
    }
    else
    {

#if DISABLE_IMPORTANCE

        return kInv2Pi;

#else
        float F = Fr(Dot(n,V), etaI, etaO);
        float PTrans = (1.0f-F)*mat.transmission;

        const float a = Max(0.001f, mat.roughness);

    	const Vec3 half = SafeNormalize(L+V);

    	const float cosThetaHalf = Abs(Dot(half, n));
        const float pdfHalf = GTR2(cosThetaHalf, a)*cosThetaHalf;

        // calculate pdf for each method given outgoing light vector
        float pdfSpec = pdfHalf*0.25f/Max(1.e-6f, Abs(Dot (L, half)));
        assert(isfinite(pdfSpec));

        float pdfDiff = Abs(Dot(L, n))*kInvPi;
        assert(isfinite(pdfDiff));

        // weight pdfs equally
        return Lerp(pdfDiff, pdfSpec, 0.5f)*(1.0f-PTrans);
#endif

    }
}


// generate an importance sampled brdf direction
CUDA_CALLABLE inline void BRDFSample(const Material& mat, float etaI, float etaO, const Vec3& P, const Mat33& frame, const Vec3& V, Vec3& light, float& pdf, Random& rand)
{
    float F = Fr(Dot(frame.GetCol(2),V), etaI, etaO);
    float PTrans = (1.0f-F)*mat.transmission;

    if (rand.Randf() < PTrans)
    {
        // sample transmission
        Vec3 n = frame.GetCol(2);

        float eta = etaI/etaO;
        //Vec3 h = Normalize(V+light);

        if (Refract(V, n, eta, light))
        {   
            pdf = PTrans;
            return;
        }
        else
        {
            assert(0);

            // shouldn't get here as the Fresnel based sampling 
            // will ensure refraction always succeeds
            pdf = 0.0f;
            return;
        }
    }
    else
    {
        // sample reflection
        float r1, r2;
        Sample2D(rand, r1, r2);

        const float select = rand.Randf();

        if (select < 0.5f)
        {
            // sample diffuse
            light = frame*CosineSampleHemisphere(r1, r2);
        }
        else
        {
            // sample specular
    	    const float a = Max(0.001f, mat.roughness);

            const float phiHalf = r1*k2Pi;
            
            const float cosThetaHalf = sqrtf((1.0f-r2)/(1.0f + (Sqr(a)-1.0f)*r2));      
            const float sinThetaHalf = sqrtf(Max(0.0f, 1.0f-Sqr(cosThetaHalf)));
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

        pdf = BRDFPdf(mat, etaI, etaO, P, frame.GetCol(2), V, light);
    }
}




CUDA_CALLABLE inline Color BRDFEval(const Material& mat, float etaI, float etaO, const Vec3& P, const Vec3& N, const Vec3& V, const Vec3& L)
{
    float NDotL = Dot(N,L);
    float NDotV = Dot(N,V);
    
    Vec3 H = Normalize(L+V);

    float NDotH = Dot(N,H);
    float LDotH = Dot(L,H);
        
    if (NDotL <= 0)
    {
        // transmission Fresnel
        float F = Fr(NDotV, etaI, etaO);

        Color T = mat.transmission*(1.0f-F)/Abs(NDotL);
        return T;
    }
    else
    {
        Vec3 Cdlin = Vec3(mat.color);
        float Cdlum = .3*Cdlin[0] + .6*Cdlin[1]  + .1*Cdlin[2]; // luminance approx.

        Vec3 Ctint = Cdlum > 0.0f ? Cdlin/Cdlum : Vec3(1); // normalize lum. to isolate hue+sat
        Vec3 Cspec0 = Lerp(mat.specular*.08*Lerp(Vec3(1), Ctint, mat.specularTint), Cdlin, mat.metallic);
       // Vec3 Csheen = Lerp(Vec3(1), Ctint, mat.sheenTint);

        // specular
        float a = Max(0.001f, mat.roughness);
        float Ds = GTR2(NDotH, a);

        // Fresnel term with the microfacet normal
        float FH = Fr(LDotH, etaI, etaO);

        Vec3 Fs = Lerp(Cspec0, Vec3(1), FH);
        float roughg = a;
        float Gs = smithG_GGX(NDotV, roughg)*smithG_GGX(NDotL, roughg);

/*
        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and mix in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
        float Fd90 = 0.5 + 2.0f * LDotH*LDotH * mat.roughness;
        float Fd = Lerp(1.0f, Fd90, FL) * (1.0f-FH);//Lerp(1.0f, Fd90, FV);

        // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
        // 1.25 scale is used to (roughly) preserve albedo
        // Fss90 used to "flatten" retroreflection based on roughness
        float Fss90 = LDotH*LDotH*mat.roughness;
        float Fss = Lerp(1.0f, Fss90, FL) * (1.0f-FH);//Lerp(1.0f, Fss90, FV);
        float ss = 1.25 * (Fss * (1.0f / (NDotL + NDotV) - .5) + .5);

        // sheen
        Vec3 Fsheen = FH * mat.sheen * Csheen;

        // clearcoat (ior = 1.5 -> F0 = 0.04)
        float Dr = GTR1(NDotH, Lerp(.1,.001, mat.clearcoatGloss));
        float Fr = Lerp(.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, .25) * smithG_GGX(NDotV, .25);
       


        Vec3 out = ((1/kPi) * Lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
            * (1-mat.metallic)*(1.0f-mat.transmission)
            + Gs*Fs*Ds + .25*mat.clearcoat*Gr*Fr*Dr;
 */
    
        Vec3 out = kInvPi*Cdlin*(Vec3(1.0f)-Fs)*(1.0-mat.metallic)*(1.0f-mat.transmission) + Gs*Fs*Ds;            

        return Vec4(out, 0.0f);
    }
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

            Color f = BRDFEval(mat, 1.0f, 1.0f, Vec3(0.0f), frame.GetCol(2), wo, wi); 
            float pdf = BRDFPdf(mat, 1.0f, 1.0f, Vec3(0.0f), frame.GetCol(2), wo, wi);

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
        Vec3 wi;
        float pdf;

        BRDFSample(mat, 1.0f, 1.0f, Vec3(0.0f), frame, wo, wi, pdf, rand);
            
        Vec2 uv = ProbeDirToUV(wi);

        int px = Clamp(int(uv.x*width), 0, width-1);
        int py = Clamp(int(uv.y*height), 0, height-1);

        pixels[py*width + px] = Vec3(1.0f, 0.0f, 0.0f);
    }

    PfmSave(filename, image);
}

