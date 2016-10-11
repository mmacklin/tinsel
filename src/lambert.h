#pragma once


// generate an importance sampled brdf direction
CUDA_CALLABLE inline void BRDFSample(const Material& mat, const Vec3& P, const Mat33& frame, const Vec3& V, Vec3& outDir, float& outPdf, Random& rand)
{	
	outDir = frame*UniformSampleHemisphere(rand);
	outPdf = kInv2Pi;
}


CUDA_CALLABLE inline Color BRDFEval(const Material& mat, const Vec3& P, const Vec3& N, const Vec3& V, const Vec3& L)
{
	return mat.color*kInvPi;
}