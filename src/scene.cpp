#include "scene.h"
#include "intersection.h"

void Scene::Build()
{
	// build scene bvh
	std::vector<Bounds> primitiveBounds;
	for (int i=0; i < primitives.size(); ++i)
	{
		Bounds r = PrimitiveBounds(primitives[i]);
		primitiveBounds.push_back(r);
	}

	BVHBuilder builder;
	bvh = builder.Build(&primitiveBounds[0], primitiveBounds.size());
}