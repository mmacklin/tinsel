#include "nlm.h"
#include "maths.h"

void AverageFilter(const Color* in, Color* out, int width, int height, int radius)
{
	for (int y=0; y < height; ++y)
	{
		for (int x=0; x < width; ++x)
		{
			int xlower = std::max(0, x-radius);
			int xupper = std::min(width-1, x+radius);

			int ylower = std::max(0, y-radius);
			int yupper = std::min(height-1, y+radius);

			int count = 0;
			Color sum;

			for (int fx=xlower; fx <= xupper; ++fx)
			{
				for (int fy=ylower; fy <= yupper; ++fy)
				{
					sum += in[fy*width + fx];
					count += 1;
				}
			}

			out[y*width + x] = sum*(1.0f/count);
		}
	}
}

void NonLocalMeansFilter(const Color* in, Color* out, int width, int height, float falloff, int radius)
{
	Color* means = new Color[width*height];

	AverageFilter(in, means, width, height, radius);

	float invRadiusSq = falloff;//200.0f;//1.0f/(0.5f*0.5f);

	for (int y=0; y < height; ++y)
	{
		for (int x=0; x < width; ++x)
		{
			int xlower = std::max(0, x-radius);
			int xupper = std::min(width-1, x+radius);

			int ylower = std::max(0, y-radius);
			int yupper = std::min(height-1, y+radius);

			float totalWeight = 0.0f;
			Color sum;

			// 
			Color mean = means[y*width + x];

			for (int fx=xlower; fx <= xupper; ++fx)
			{
				for (int fy=ylower; fy <= yupper; ++fy)
				{
					float weight = expf(-invRadiusSq*LengthSq(mean - means[fy*width+fx]));

					sum += in[fy*width + fx]*weight;
					totalWeight += weight;
				}
			}

			out[y*width + x] = sum*(1.0f/totalWeight);
		}
	}

	delete[] means;
}




