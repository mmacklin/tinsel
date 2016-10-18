#pragma once

#include "maths.h"

void NonLocalMeansFilter(const Color* in, Color* out, int width, int height, float falloff, int radius);