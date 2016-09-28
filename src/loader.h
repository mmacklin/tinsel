#pragma once

struct Scene;
struct Camera;
struct Options;

bool LoadTin(const char* filename, Scene* scene, Camera* camera, Options* options);