#include "pfm.h"

#include <cassert>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "maths.h"

namespace
{
	// RAII wrapper to handle file pointer clean up
	struct FilePointer
	{
		FilePointer(FILE* ptr) : p(ptr) {}
		~FilePointer() { if (p) fclose(p); }

		operator FILE*() { return p; }

		FILE* p;
	};
}

bool PfmLoad(const char* filename, PfmImage& image)
{
	FilePointer f = fopen(filename, "rb");
	if (!f)
		return false;

	memset(&image, 0, sizeof(PfmImage));
	
	const uint32_t kBufSize = 1024;
	char buffer[kBufSize];
	
	if (!fgets(buffer, kBufSize, f))
		return false;
	
	if (strcmp(buffer, "PF\n") != 0)
		return false;
	
	if (!fgets(buffer, kBufSize, f))
		return false;

	image.depth = 1;
	sscanf(buffer, "%d %d %d", &image.width, &image.height, &image.depth);

	if (!fgets(buffer, kBufSize, f))
		return false;
	
	sscanf(buffer, "%f", &image.maxDepth);
	
	uint32_t dataStart = ftell(f);
	fseek(f, 0, SEEK_END);
	uint32_t dataEnd = ftell(f);
	fseek(f, dataStart, SEEK_SET);
	
	uint32_t dataSize = dataEnd-dataStart;

	// must be 4 byte aligned
	assert((dataSize&0x3) == 0);
	
	image.data = new float[dataSize/4];
	
	if (fread(image.data, dataSize, 1, f) != 1)
		return false;
	
	return true;
}

void PfmSave(const char* filename, const PfmImage& image)
{
	FILE* f = fopen(filename, "wb");
	if (!f)
		return;

	fprintf(f, "PF\n");
	if (image.depth > 1)
		fprintf(f, "%d %d %d\n", image.width, image.height, image.depth);
	else
		fprintf(f, "%d %d\n", image.width, image.height);

	fprintf(f, "-%f\n", *std::max_element(image.data, image.data+(image.width*image.height*image.depth*3)));

	fwrite(image.data, image.width*image.height*image.depth*sizeof(float)*3, 1, f);
}

///--------

typedef unsigned char RGBE[4];
#define R			0
#define G			1
#define B			2
#define E			3

#define  MINELEN	8				// minimum scanline length for encoding
#define  MAXELEN	0x7fff			// maximum scanline length for encoding

static void workOnRGBE(RGBE *scan, PfmImage& res, float *cols );
static bool decrunch(RGBE *scanline, int len, FILE *file);
static bool oldDecrunch(RGBE *scanline, int len, FILE *file);

bool HdrLoad(const char *fileName, PfmImage& res)
{
	int i;
	char str[200];
	FILE *file;

	file = fopen(fileName, "rb");
	if (!file)
		return false;

	fread(str, 10, 1, file);
	if (memcmp(str, "#?RADIANCE", 10)) {
		fclose(file);
		return false;
	}

	fseek(file, 1, SEEK_CUR);

	char cmd[200];
	i = 0;
	char c = 0, oldc;
	while(true) {
		oldc = c;
		c = fgetc(file);
		if (c == 0xa && oldc == 0xa)
			break;
		cmd[i++] = c;
	}

	char reso[200];
	i = 0;
	while(true) {
		c = fgetc(file);
		reso[i++] = c;
		if (c == 0xa)
			break;
	}

	int w, h;
	if (!sscanf(reso, "-Y %d +X %d", &h, &w)) {
		fclose(file);
		return false;
	}

	res.width = w;
	res.height = h;

	float *cols = new float[w * h * 3];
	res.data = cols;
	res.emin = 127;
	res.emax = -127;

	RGBE *scanline = new RGBE[w];
	if (!scanline) {
		fclose(file);
		return false;
	}

	// convert image 
	for (int y = h - 1; y >= 0; y--) {
		if (decrunch(scanline, w, file) == false)
			break;
		workOnRGBE(scanline, res, cols );
		cols += w * 3;
	}

	delete [] scanline;
	fclose(file);

	return true;
}

float convertComponent(int expo, int val)
{
	if( expo == -128 ) return 0.0;
	float v = val / 256.0f;
	float d = (float) powf(2.0f, expo);
	return v * d;
}

void workOnRGBE(RGBE *scan, PfmImage& res, float *cols )
{

	int len = res.width;

	while (len-- > 0) {
		int expo = scan[0][E] - 128;
		if( expo > res.emax ) res.emax = expo;
		if( expo != -128 && expo < res.emin ) res.emin = expo;
		cols[0] = convertComponent(expo, scan[0][R]);
		cols[1] = convertComponent(expo, scan[0][G]);
		cols[2] = convertComponent(expo, scan[0][B]);
		cols += 3;
		scan++;
	}
}

bool decrunch(RGBE *scanline, int len, FILE *file)
{
	int  i, j;
					
	if (len < MINELEN || len > MAXELEN)
		return oldDecrunch(scanline, len, file);

	i = fgetc(file);
	if (i != 2) {
		fseek(file, -1, SEEK_CUR);
		return oldDecrunch(scanline, len, file);
	}

	scanline[0][G] = fgetc(file);
	scanline[0][B] = fgetc(file);
	i = fgetc(file);

	if (scanline[0][G] != 2 || scanline[0][B] & 128) {
		scanline[0][R] = 2;
		scanline[0][E] = i;
		return oldDecrunch(scanline + 1, len - 1, file);
	}

	// read each component
	for (i = 0; i < 4; i++) {
	    for (j = 0; j < len; ) {
			unsigned char code = fgetc(file);
			if (code > 128) { // run
			    code &= 127;
			    unsigned char val = fgetc(file);
			    while (code--)
					scanline[j++][i] = val;
			}
			else  {	// non-run
			    while(code--)
					scanline[j++][i] = fgetc(file);
			}
		}
    }

	return feof(file) ? false : true;
}

bool oldDecrunch(RGBE *scanline, int len, FILE *file)
{
	int i;
	int rshift = 0;
	
	while (len > 0) {
		scanline[0][R] = fgetc(file);
		scanline[0][G] = fgetc(file);
		scanline[0][B] = fgetc(file);
		scanline[0][E] = fgetc(file);
		if (feof(file))
			return false;

		if (scanline[0][R] == 1 &&
			scanline[0][G] == 1 &&
			scanline[0][B] == 1) {
			for (i = scanline[0][E] << rshift; i > 0; i--) {
				memcpy(&scanline[0][0], &scanline[-1][0], 4);
				scanline++;
				len--;
			}
			rshift += 8;
		}
		else {
			scanline++;
			len--;
			rshift = 0;
		}
	}
	return true;
}


