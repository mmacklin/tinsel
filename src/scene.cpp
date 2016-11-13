#include "scene.h"
#include "intersection.h"

void Scene::Build()
{
	// build scene bvh
	std::vector<Bounds> primitiveBounds;
	for (int i=0; i < primitives.size(); ++i)
	{
		Bounds r = PrimitiveBounds(primitives[i]);

		printf("%f %f %f - %f %f %f\n", r.lower.x, r.lower.y, r.lower.z, r.upper.x, r.upper.y, r.upper.z);
		primitiveBounds.push_back(r);
	}

	BVHBuilder builder;
	bvh = builder.Build(&primitiveBounds[0], primitiveBounds.size());

	Bounds r = primitiveBounds[0];
	printf("%f %f %f - %f %f %f\n", r.lower.x, r.lower.y, r.lower.z, r.upper.x, r.upper.y, r.upper.z);
}