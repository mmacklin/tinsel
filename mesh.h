#pragma once

#include <vector>

#include "maths.h"

struct Mesh
{
    void AddMesh(Mesh& m);

	void DuplicateVertex(int i);

    void CalculateNormals();
    void Transform(const Mat44& m);
	void Normalize(float s=1.0f);	// scale so bounds in any dimension equals s and lower bound = (0,0,0)

    void GetBounds(Vec3& minExtents, Vec3& maxExtents) const;

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> texcoords[2];
    std::vector<Color> Colors;

    std::vector<int> indices;
};

// create mesh from file
Mesh* ImportMeshFromObj(const char* path);
Mesh* ImportMeshFromPly(const char* path);

// just switches on filename
Mesh* ImportMesh(const char* path);

// create procedural primitives
Mesh* CreateQuadMesh(float size, float y=0.0f);
Mesh* CreateDiscMesh(float radius, int segments);
Mesh* CreateTetrahedron();

