#pragma once

#include "maths.h"

// sample [0,1] with x strata
inline void StratifiedSample1D(int c, int dx, Random& rand, float& r1)
{
	// map c onto stratum
	int x = c%dx;

	r1 = (float(x) + rand.Randf())/dx;
}


// sample [0,1]*[0,1] with x,y strata
inline void StratifiedSample2D(int c, int dx, int dy, Random& rand, float& r1, float& r2)
{	
	// map c onto stratum
	int x = c%dx;
	int y = (c/dx)%dy;

	r1 = (float(x) + rand.Randf())/float(dx);
	r2 = (float(y) + rand.Randf())/float(dy);
}

// sample [0,1] with x strata
inline void UniformSample1D(int c, int dx, float& r1)
{
	// map c onto stratum
	int x = c%dx;

	r1 = float(x)/dx;
}


// sample [0,1]*[0,1] with x,y strata
inline void UniformSample2D(int c, int dx, int dy, float& r1, float& r2)
{	
	// map c onto stratum
	int x = c%dx;
	int y = (c/dx)%dy;

	r1 = (float(x))/float(dx);
	r2 = (float(y))/float(dy);
}

inline float ToiroidalDistanceSq(const float* v1, const float* v2, int dim)
{
	float dSq = 0.0f;

	// ensure each coordinate is in the lower half of the domain
	for (int i=0; i < dim; ++i)
	{
		float delta = Abs(v1[i] - v2[i]);

		if (delta > 0.5f)
			delta = 1.0f-delta;

		dSq += Sqr(delta);
	}

	return dSq;
}


inline float DistanceSq(const float* v1, const float* v2, int dim)
{

	float dSq = 0.0f;

	for (int i=0; i < dim; ++i)
	{
		float delta = v1[i] - v2[i];
		dSq += Sqr(delta);
	}

	return dSq;
}

template <int dim>
void BestCandidateSampling(int n, float* samples)
{
	const int kCandidates = 100000;

	Random rand;

	for (int i=0; i < n; ++i)
	{
		float bestCandidate[dim];
		float bestCandidateDistSq = 0;

		for (int c=0; c < kCandidates; ++c)
		{
			// generate candidate sample
			float candidate[dim];
			for (int d=0; d < dim; ++d)
			{
				candidate[d] = rand.Randf();
			}

			float minCandidateDistSq = FLT_MAX;

			// find the minimum distance to any existing sample
			for (int s=0; s < i; ++s)
			{
				//const float dSq = ToiroidalDistanceSq(candidate, &samples[s*dim], dim);
				const float dSq = ToiroidalDistanceSq(candidate, &samples[s*dim], dim);

				if (dSq < minCandidateDistSq)
					minCandidateDistSq = dSq;
			}

			// if our minimum distance is greater than the best candidate then update it
			if (minCandidateDistSq > bestCandidateDistSq)
			{
				for (int d=0; d < dim; ++d)
				{
					bestCandidate[d] = candidate[d];
				}

				bestCandidateDistSq = minCandidateDistSq;
			}
		}

		// add best candidate to sample set
		for (int d=0; d < dim; ++d)
		{
			samples[i*dim + d] = bestCandidate[d];
		}
	}
}

template <int dim>
void ProjectiveBlueNoiseSampling(int n, float* samples)
{
	const int kCandidates = 10000;
	const float kReduction = 0.999f;

	float radius = 1.0f;
	float radiusSq = 1.0f;

	float projectedRadius = 1.0f/n;
	float projectedRadiusSq = Sqr(projectedRadius);

	Random rand;

	for (int i=0; i < n; ++i)
	{
		bool valid = false;

		while (!valid)
		{
			for (int c=0; c < kCandidates; ++c)
			{
				// generate candidate sample
				float candidate[dim];
				for (int d=0; d < dim; ++d)
				{
					candidate[d] = rand.Randf();
				}
		
				float dSq = FLT_MAX;

				// find the minimum distance to any existing sample
				for (int s=0; s < i; ++s)
				{
					 dSq = ToiroidalDistanceSq(candidate, &samples[s*dim], dim);

					 // test projection on each axis
					 for (int d=0; d < dim; ++d)
					 {
					 	if (ToiroidalDistanceSq(&candidate[d], &samples[s*dim+d], 1) < projectedRadiusSq)
					 	{
					 		dSq = 0.0f;
					 		break;
					 	}
					 }
					
					if (dSq < radiusSq)
						break;
				}

				// if we found a valid candidate add best candidate to sample set
				if (dSq > radiusSq)
				{
					for (int d=0; d < dim; ++d)
					{
						samples[i*dim + d] = candidate[d];
					}

					printf("Added %d\n", i);
					fflush(stdout);

					valid = true;
					break;
				}
			}

			if (!valid)
			{
				// couldn't geneate a candidate, reduce test size
				radius *= kReduction;
				radiusSq = Sqr(radius);

				projectedRadius *= 0.999f;
				projectedRadiusSq = Sqr(projectedRadius);


				printf("Reducing radius %f\n", radius);
				fflush(stdout);
			}
		}
	}
}

inline void ToiroidalShiftSample(float* shifted, const float* sample, int dim, Random& rand)
{
	for (int i=0; i < dim; ++i)
	{
		float r = sample[i] + rand.Randf();
		if (r > 1.0f)
			r -= 1.0f;

		shifted[i] = r;
	}
}

extern const int sampleDim;
extern int sampleIndex;
extern int dimIndex;
extern float sampleOffsetX;
extern float sampleOffsetY;
extern float sample[];

#define USE_RANDOM 1
#define USE_BLUE_NOISE 0

inline void Sample1D(Random& rand, float& u1)
{
#if USE_RANDOM

	u1 = rand.Randf(0.0f, 1.0f);

#elif USE_BLUE_NOISE
	u1 = sample[dimIndex%sampleDim];
	dimIndex += 1;

#endif

}

inline void Sample2D(Random& rand, float& u1, float& u2)
{
#if USE_RANDOM

	u1 = rand.Randf(0.0f, 1.0f);
	u2 = rand.Randf(0.0f, 1.0f);

#elif USE_STRATIFIED

	float u1, u2;
	StratifiedSample2D(sampleCount, 8, 8, rand, u1, u2);

#elif USE_BLUE_NOISE

	u1 = sample[(dimIndex+0)%sampleDim];
	u2 = sample[(dimIndex+1)%sampleDim];

	dimIndex += 2;

#elif USE_JITTERED_UNIFORM

	// blue noise jittered uniform
	UniformSample2D(sampleCount, 8, 8, u1, u2);

	// jitter
	u1 += sampleOffsetX;
	u2 += sampleOffsetY;

	// toiroidal wrap
	if (u1 > 1.0f)
		u1 -= 1.0f;

	if (u2 > 1.0f)
		u2 -= 1.0f;

#endif

}



