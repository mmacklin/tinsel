#pragma once

struct PfmImage
{
	// set m_depth to 1 for standard pfm compatability, > 1 will act as a volume texture (non-standard)
	int width;
	int height;
	int depth;

	// optional
	float maxDepth;

	float emin;
	float emax;
	
	float* data;
};

bool PfmLoad(const char* filename, PfmImage& image);
void PfmSave(const char* filename, const PfmImage& image);

bool HdrLoad(const char* filename, PfmImage& image);