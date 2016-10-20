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

	r1 = (float(x) + rand.Randf())/dx;
	r2 = (float(y) + rand.Randf())/dy;
}
