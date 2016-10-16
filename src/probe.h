#pragma once

#include "maths.h"
#include "pfm.h"

struct Probe
{
	int width;
	int height;

	Color* data;

	// world space offset to effectively warp the skybox
	Vec3 offset;

	Probe() : valid(false) {}

	// cdf

		struct Entry
		{
			float weight;
			int index;

			inline bool operator < (const Entry& e) const { return weight < e.weight; }
		};

	inline void BuildCDF()
	{
		pdfValuesX = new float[width*height];
		pdfValuesY = new float[height];

		cdfValuesX = new float[width*height];
		cdfValuesY = new float[height];

		float totalWeightY = 0.0f;

		for (int j=0; j < height; ++j)
		{
			float totalWeightX = 0.0f;

			for (int i=0; i < width; ++i)
			{
				float weight = Luminance(data[j*width + i]);

				totalWeightX += weight;
				
				pdfValuesX[j*width + i] = weight;
				cdfValuesX[j*width + i] = totalWeightX;
			}			

			float invTotalWeightX = 1.0f/totalWeightX;

			// convert to pdf and cdf
			for (int i=0; i < width; ++i)
			{
				pdfValuesX[j*width + i] *= invTotalWeightX;
				cdfValuesX[j*width + i] *= invTotalWeightX;
			}

			// total weight y 
			totalWeightY += totalWeightX;

			pdfValuesY[j] = totalWeightX;
			cdfValuesY[j] = totalWeightY;
		}

		// convert Y to cdf
		for (int j=0; j < height; ++j)
		{
			cdfValuesY[j] /= float(totalWeightY);
			pdfValuesY[j] /= float(totalWeightY);
		}

		valid = true;
	}

	bool valid;
	
	float* pdfValuesX;
	float* cdfValuesX;

	float* pdfValuesY;
	float* cdfValuesY;
};


inline Color SampleProbeSphere(const Probe& image, const Vec3& dir)
{
	// convert world space dir to probe space
	float c = kInvPi * acosf(dir.z)/sqrt(dir.x*dir.x + dir.y*dir.y);
	
	int px = (0.5f + 0.5f*(dir.x*c))*image.width;
	int py = (0.5f + 0.5f*(-dir.y*c))*image.height;

	px = Min(Max(0, px), image.width-1);
	py = Min(Max(0, py), image.height-1);
	
	return image.data[py*image.width+px];
}

inline Vec2 ProbeDirToUV(const Vec3& dir)
{
	float theta = acosf(dir.y);
    float phi = atan2(dir.z, dir.x);
    float u = (kPi + phi)*kInvPi*0.5f;
    float v = theta*kInvPi;	

    return Vec2(u, v);
}

inline Vec3 ProbeUVToDir(const Vec2& uv)
{
	float theta = uv.y * kPi;
    float phi = uv.x * 2.0f * kPi;

	float x = -sinf(theta) * cosf(phi);
    float y = cosf(theta);
    float z = -sinf(theta) * sinf(phi);

    return Vec3(x, y, z);
}


inline Color ProbeEval(const Probe& image, const Vec2& uv)
{
	int px = Clamp(int(uv.x*image.width), 0, image.width-1);
	int py = Clamp(int(uv.y*image.height), 0, image.height-1);

	return image.data[py*image.width+px];
}

inline float ProbePdf(const Probe& image, const Vec3& d)
{
	Vec2 uv = ProbeDirToUV(d);

	int col = uv.x * image.width;
	int row = uv.y * image.height;

	float pdf = image.pdfValuesX[row*image.width + col]*image.pdfValuesY[row];

	float sinTheta = sinf(uv.y*kPi);
	if (sinTheta == 0.0f)
		pdf = 0.0f;
	else
		pdf *= image.width*image.height/(2.0f*kPi*kPi*sinTheta);

	return pdf;
}



inline void ProbeSample(const Probe& image, Vec3& dir, Color& color, float& pdf, Random& rand)
{
	float r1 = rand.Randf();
	float r2 = rand.Randf();

	// sample rows
	float* rowPtr = std::lower_bound(image.cdfValuesY, image.cdfValuesY+image.height, r1);
	int row = rowPtr - image.cdfValuesY;

	// sample cols of row
	float* colPtr = std::lower_bound(&image.cdfValuesX[row*image.width], &image.cdfValuesX[(row+1)*image.width], r2);
	int col = colPtr - &image.cdfValuesX[row*image.width];

	color = image.data[row*image.width + col];
	pdf = image.pdfValuesX[row*image.width + col]*image.pdfValuesY[row];

	float u = col/float(image.width);
	float v = row/float(image.height);

	float sinTheta = sinf(v*kPi);
	if (sinTheta == 0.0f)
		pdf = 0.0f;
	else
		pdf *= image.width*image.height/(2.0f*kPi*kPi*sinTheta);

	dir = ProbeUVToDir(Vec2(u, v));

}

inline Probe ProbeLoadFromFile(const char* path)
{
	PfmImage image;
	//PfmLoad(path, image);
	HdrLoad(path, image);

	Probe probe;
	probe.width = image.width;
	probe.height = image.height;

	int numPixels = image.width*image.height;

	// convert image data to color data, apply pre-exposure etc
	probe.data = new Color[numPixels];

	for (int i=0; i < numPixels; ++i)
		probe.data[i] = Color(image.data[i*3+0], image.data[i*3+1], image.data[i*3+2]);
	
	probe.BuildCDF();

	delete[] image.data;

	return probe;
}

inline Probe ProbeCreateTest()
{
	Probe p;
	p.width = 100;
	p.height = 50;
	p.data = new Color[p.width*p.height];

	Vec3 axis = Normalize(Vec3(.0f, 1.0f, 0.0f));

	for (int i=0; i < p.width; i++)
	{
		for (int j=0; j < p.height; j++)
		{
			// add a circular disc based on a view dir
			float u = i/float(p.width);
			float v = j/float(p.height);

			Vec3 dir = ProbeUVToDir(Vec2(u, v));

			if (Dot(dir, axis) >= 0.95f)
			{
				p.data[j*p.width+i] = Color(10.0f);

			}
			else
			{
				p.data[j*p.width+i] = 0.0f;
			}
		}
	}

	p.BuildCDF();


	return p;
}


inline void ProbeMark(Probe& p)
{
	Random rand;

	// sample probe a number of times
	for (int i=0; i < 500; ++i)
	{
		Vec3 dir;
		Color c;
		float pdf;

		ProbeSample(p, dir, c, pdf, rand);

		Vec2 uv = ProbeDirToUV(dir);

		int px = Clamp(int(uv.x*p.width), 0, p.width-1);
		int py = Clamp(int(uv.y*p.height), 0, p.height-1);

		//printf("%d %d\n", px, py);

		p.data[py*p.width+px] = Color(1.0f, 0.0f, 0.0f);


	}
}
