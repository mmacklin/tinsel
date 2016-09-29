/* 
 * Tiny PNG Output (C)
 * 
 * Copyright (c) 2016 Project Nayuki
 * https://www.nayuki.io/page/tiny-png-output-c
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program (see COPYING.txt).
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include "maths.h"

/* 
 * TinyPngOut data structure. Treat this as opaque; do not read or write any fields directly.
 */
struct TinyPngOut {
    
    // Configuration
    int32_t width;   // Measured in bytes, not pixels. A row has (width * 3 + 1) bytes.
    int32_t height;  // Measured in pixels
    
    // State
    FILE *outStream;
    int32_t positionX;  // Measured in bytes
    int32_t positionY;  // Measured in pixels
    int32_t deflateRemain;  // Measured in bytes
    int32_t deflateFilled;  // Number of bytes filled in the current block (0 <= n < 65535)
    uint32_t crc;    // For IDAT chunk
    uint32_t adler;  // For DEFLATE data within IDAT
    
};


/* 
 * Enumeration of status codes
 */
enum TinyPngOutStatus {
	TINYPNGOUT_OK,
	TINYPNGOUT_DONE,
	TINYPNGOUT_INVALID_ARGUMENT,
	TINYPNGOUT_IO_ERROR,
	TINYPNGOUT_IMAGE_TOO_LARGE,
};


/* 
 * Initialization function.
 * 
 * Example usage:
 *   #define WIDTH 640
 *   #define HEIGHT 480
 *   FILE *fout = fopen("image.png", "wb");
 *   struct TinyPngOut pngout;
 *   if (fout == NULL || TinyPngOut_init(&pngout, fout, WIDTH, HEIGHT) != TINYPNGOUT_OK) {
 *     ... (handle error) ...
 *   }
 */
enum TinyPngOutStatus TinyPngOut_init(struct TinyPngOut *pngout, FILE *fout, int32_t width, int32_t height);

/* 
 * Pixel-writing function. The function reads 3*count bytes from the array.
 * Pixels are presented in the array in RGB order, from top to bottom, left to right.
 * It is an error to write more pixels in total than width*height.
 * After all the pixels are written, every subsequent call will return TINYPNGOUT_DONE (which is considered success).
 * 
 * Example usage:
 *   uint8_t pixels[WIDTH * HEIGHT * 3];
 *   ... (fill pixels) ...
 *   if (TinyPngOut_write(&pngout, pixels, WIDTH * HEIGHT) != TINYPNGOUT_OK) {
 *     ... (handle error) ...
 *   }
 *   if (TinyPngOut_write(&pngout, NULL, 0) != TINYPNGOUT_DONE) {
 *     ... (handle error) ...
 *   }
 *   fclose(fout);
 */
enum TinyPngOutStatus TinyPngOut_write(struct TinyPngOut *pngout, const uint8_t *pixels, int count);


/* 
 * Tiny PNG Output (C)
 * 
 * Copyright (c) 2014 Project Nayuki
 * https://www.nayuki.io/page/tiny-png-output-c
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program (see COPYING.txt).
 * If not, see <http://www.gnu.org/licenses/>.
 */





/* Local declarations */

#define DEFLATE_MAX_BLOCK_SIZE 65535
#define MIN(x, y) ((x) < (y) ? (x) : (y))

static enum TinyPngOutStatus finish(const struct TinyPngOut *pngout);
static uint32_t crc32  (uint32_t state, const uint8_t *data, size_t len);
static uint32_t adler32(uint32_t state, const uint8_t *data, size_t len);


/* Public function implementations */

enum TinyPngOutStatus TinyPngOut_init(struct TinyPngOut *pngout, FILE *fout, int32_t width, int32_t height) {
	// Check arguments
	if (fout == NULL || width <= 0 || height <= 0)
		return TINYPNGOUT_INVALID_ARGUMENT;
	
	// Calculate data sizes
	if (width > (UINT32_MAX - 1) / 3)
		return TINYPNGOUT_IMAGE_TOO_LARGE;
	uint32_t lineSize = width * 3 + 1;
	
	if (lineSize > UINT32_MAX / height)
		return TINYPNGOUT_IMAGE_TOO_LARGE;
	uint32_t size = lineSize * height;  // Size of DEFLATE input
	pngout->deflateRemain = size;
	
	uint32_t overhead = size / DEFLATE_MAX_BLOCK_SIZE;
	if (overhead * DEFLATE_MAX_BLOCK_SIZE < size)
		overhead++;  // Round up to next block
	overhead = overhead * 5 + 6;
	if (size > UINT32_MAX - overhead)
		return TINYPNGOUT_IMAGE_TOO_LARGE;
	size += overhead;  // Size of zlib+DEFLATE output
	
	// Set most of the fields
	pngout->width = lineSize;  // In bytes
	pngout->height = height;   // In pixels
	pngout->outStream = fout;
	pngout->positionX = 0;
	pngout->positionY = 0;
	pngout->deflateFilled = 0;
	pngout->adler = 1;
	
	// Write header (not a pure header, but a couple of things concatenated together)
	#define HEADER_SIZE 43
	uint8_t header[HEADER_SIZE] = {
		// PNG header
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
		// IHDR chunk
		0x00, 0x00, 0x00, 0x0D,
		0x49, 0x48, 0x44, 0x52,
		width  >> 24, width  >> 16, width  >> 8, width  >> 0,
		height >> 24, height >> 16, height >> 8, height >> 0,
		0x08, 0x02, 0x00, 0x00, 0x00,
		0, 0, 0, 0,  // IHDR CRC-32 to be filled in (starting at offset 29)
		// IDAT chunk
		size >> 24, size >> 16, size >> 8, size >> 0,
		0x49, 0x44, 0x41, 0x54,
		// DEFLATE data
		0x08, 0x1D,
	};
	uint32_t crc = crc32(0, &header[12], 17);
	header[29] = crc >> 24;
	header[30] = crc >> 16;
	header[31] = crc >>  8;
	header[32] = crc >>  0;
	if (fwrite(header, 1, HEADER_SIZE, fout) != HEADER_SIZE)
		return TINYPNGOUT_IO_ERROR;
	
	pngout->crc = crc32(0, &header[37], 6);
	return TINYPNGOUT_OK;
}


enum TinyPngOutStatus TinyPngOut_write(struct TinyPngOut *pngout, const uint8_t *pixels, int count) {
	int32_t width  = pngout->width;
	int32_t height = pngout->height;
	if (pngout->positionY == height)
		return TINYPNGOUT_DONE;
	if (count < 0 || count > INT_MAX / 3 || pngout->positionY < 0 || pngout->positionY > height)
		return TINYPNGOUT_INVALID_ARGUMENT;
	
	count *= 3;
	FILE *f = pngout->outStream;
	while (count > 0) {
		// Start DEFLATE block
		if (pngout->deflateFilled == 0) {
			#define BLOCK_HEADER_SIZE 5
			uint16_t size = (uint16_t)MIN(pngout->deflateRemain, DEFLATE_MAX_BLOCK_SIZE);
			uint8_t blockheader[BLOCK_HEADER_SIZE] = {
				pngout->deflateRemain <= DEFLATE_MAX_BLOCK_SIZE ? 1 : 0,
				size >> 0,
				size >> 8,
				(size ^ UINT16_C(0xFFFF)) >> 0,
				(size ^ UINT16_C(0xFFFF)) >> 8,
			};
			if (fwrite(blockheader, 1, BLOCK_HEADER_SIZE, f) != BLOCK_HEADER_SIZE)
				return TINYPNGOUT_IO_ERROR;
			pngout->crc = crc32(pngout->crc, blockheader, BLOCK_HEADER_SIZE);
		}
		
		// Calculate number of bytes to write in this loop iteration
		int n = MIN(count, width - pngout->positionX);
		n = MIN(DEFLATE_MAX_BLOCK_SIZE - pngout->deflateFilled, n);
		if (n <= 0)  // Impossible
			exit(EXIT_FAILURE);
		
		// Beginning of row - write filter method
		if (pngout->positionX == 0) {
			uint8_t b = 0;
			if (fputc(b, f) == EOF)
				return TINYPNGOUT_IO_ERROR;
			pngout->crc = crc32(pngout->crc, &b, 1);
			pngout->adler = adler32(pngout->adler, &b, 1);
			pngout->deflateRemain--;
			pngout->deflateFilled++;
			pngout->positionX++;
			n--;
		}
		
		// Write bytes and update checksums
		if (fwrite(pixels, 1, n, f) != n)
			return TINYPNGOUT_IO_ERROR;
		pngout->crc = crc32(pngout->crc, pixels, n);
		pngout->adler = adler32(pngout->adler, pixels, n);
		
		// Increment the position
		count -= n;
		pixels += n;
		
		pngout->deflateRemain -= n;
		pngout->deflateFilled += n;
		if (pngout->deflateFilled == DEFLATE_MAX_BLOCK_SIZE)
			pngout->deflateFilled = 0;
		
		pngout->positionX += n;
		if (pngout->positionX == width) {
			pngout->positionX = 0;
			pngout->positionY++;
			if (pngout->positionY == height) {
				if (count > 0)
					return TINYPNGOUT_INVALID_ARGUMENT;
				return finish(pngout);
			}
		}
	}
	return TINYPNGOUT_OK;
}


/* Private function implementations */

static enum TinyPngOutStatus finish(const struct TinyPngOut *pngout) {
	#define FOOTER_SIZE 20
	uint32_t adler = pngout->adler;
	uint8_t footer[FOOTER_SIZE] = {
		adler >> 24, adler >> 16, adler >> 8, adler >> 0,
		0, 0, 0, 0,  // IDAT CRC-32 to be filled in (starting at offset 4)
		// IEND chunk
		0x00, 0x00, 0x00, 0x00,
		0x49, 0x45, 0x4E, 0x44,
		0xAE, 0x42, 0x60, 0x82,
	};
	uint32_t crc = crc32(pngout->crc, &footer[0], 4);
	footer[4] = crc >> 24;
	footer[5] = crc >> 16;
	footer[6] = crc >>  8;
	footer[7] = crc >>  0;
	
	if (fwrite(footer, 1, FOOTER_SIZE, pngout->outStream) != FOOTER_SIZE)
		return TINYPNGOUT_IO_ERROR;
	return TINYPNGOUT_OK;
}


static uint32_t crc32(uint32_t state, const uint8_t *data, size_t len) {
	state = ~state;
	size_t i;
	for (i = 0; i < len; i++) {
		unsigned int j;
		for (j = 0; j < 8; j++) {  // Inefficient bitwise implementation, instead of table-based
			uint32_t bit = (state ^ (data[i] >> j)) & 1;
			state = (state >> 1) ^ ((-bit) & 0xEDB88320);
		}
	}
	return ~state;
}


static uint32_t adler32(uint32_t state, const uint8_t *data, size_t len) {
	uint16_t s1 = state >>  0;
	uint16_t s2 = state >> 16;
	size_t i;
	for (i = 0; i < len; i++) {
		s1 = (s1 + data[i]) % 65521;
		s2 = (s2 + s1) % 65521;
	}
	return (uint32_t)s2 << 16 | s1;
}


inline uint8_t Quantize(float x)
{
	return uint8_t(Clamp(x, 0.0f, 255.0f));
}

void WritePng(const Color* pixels, int width, int height, const char* filename)
{
	uint8_t* buffer = new uint8_t[width*height*3];

	Random rand;

	for (int i=0; i < width*height; ++i)
	{
		Color c = pixels[i];

		buffer[i*3+0] = Quantize(c.x*255.0 + rand.Randf() + rand.Randf() - 0.5f);
		buffer[i*3+1] = Quantize(c.y*255.0 + rand.Randf() + rand.Randf() - 0.5f);
		buffer[i*3+2] = Quantize(c.z*255.0 + rand.Randf() + rand.Randf() - 0.5f);
	}

	FILE *fout = fopen(filename, "wb");
	
	struct TinyPngOut pngout;
	if (fout == NULL || TinyPngOut_init(&pngout, fout, width, height) != TINYPNGOUT_OK)
		goto error;
	
	// Write image data
	if (TinyPngOut_write(&pngout, buffer, width * height) != TINYPNGOUT_OK)
		goto error;
	
	// Check for proper completion
	if (TinyPngOut_write(&pngout, NULL, 0) != TINYPNGOUT_DONE)
		goto error;

error:

	delete[] buffer;
	fclose(fout);


}