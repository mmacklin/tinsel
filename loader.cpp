#include "loader.h"

#include "scene.h"
#include "mesh.h"
#include "maths.h"

#include <stdio.h>

#include <map>
#include <string>

static const int kMaxLineLength = 2048;


Scene* LoadTin(const char* filename)
{
	FILE* file = fopen(filename, "r");

	if (!file)
	{
		printf("Couldn't open %s for reading.", filename);
		return NULL;
	}

	std::map<std::string, Mesh*> meshes;
	std::map<std::string, Material> materials;

	Scene* scene = new Scene();

	// tokens must be less than 2048 characters
	char line[kMaxLineLength];

	while (fgets(line, kMaxLineLength, file))
	{
		// skip comments
		if (line[0] == '#')
			continue;

		// name used for materials and meshes
		char name[kMaxLineLength] = { 0 };

		//--------------------------------------------
		// Material

		if (sscanf(line, " material %s", name) == 1)
		{
			printf("%s", line);

			Material material;

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " name %s", name);
				sscanf(line, " emission %f %f %f", &material.emission.x, &material.emission.y, &material.emission.z);
				sscanf(line, " color %f %f %f", &material.color.x, &material.color.y, &material.color.z);
				
				sscanf(line, " metallic %f", &material.metallic);
				sscanf(line, " subsurface %f", &material.subsurface);
				sscanf(line, " specular %f", &material.specular);
				sscanf(line, " roughness %f", &material.roughness);
				sscanf(line, " specularTint %f", &material.specularTint);
				sscanf(line, " anisotropic %f", &material.anisotropic);
				sscanf(line, " sheen %f", &material.sheen);
				sscanf(line, " sheenTint %f", &material.sheenTint);
				sscanf(line, " clearcoat %f", &material.clearcoat);
				sscanf(line, " clearcoatGloss %f", &material.clearcoatGloss);
			}

			// add material to map
			materials[name] = material;
		}

		//--------------------------------------------
		// Primitive

		if (strstr(line, "primitive"))
		{			
			Primitive primitive;

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				char type[2048];
				if (sscanf(line, " type %s", type) == 1)
				{					
					if (strcmp(type, "sphere") == 0)
					{
						primitive.type = eSphere;
					}
					else if (strcmp(type, "plane") == 0)
					{
						primitive.type = ePlane;
					}
					else if (strcmp(type, "mesh") == 0)
					{
						primitive.type = eMesh;
					}
				}

				sscanf(line, " radius %f", &primitive.sphere.radius);
				sscanf(line, " plane %f %f %f %f", &primitive.plane.plane[0], &primitive.plane.plane[1], &primitive.plane.plane[2], &primitive.plane.plane[3]);
				sscanf(line, " lightSamples %d", &primitive.light);

				char path[2048];

				if (sscanf(line, " material %s", path) == 1)
				{
					// look up material in dictionary
					if (materials.find(path) != materials.end())
					{
						primitive.material = materials[path];
					}
				}

				if (sscanf(line, " mesh %s", path) == 1)
				{
					// look up in the mesh array
					if (meshes.find(path) != meshes.end())
					{
						primitive.mesh = GeometryFromMesh(meshes[path]);
					}
					else
					{
						char relativePath[kMaxLineLength];

						// get base path of tin file
						const char* lastSlash = NULL;

						if (!lastSlash)
							lastSlash = strrchr(filename, '\\');
						if (!lastSlash)
							lastSlash = strrchr(filename, '/');

						int baseLength = 0;

						if (lastSlash)
						{
							baseLength = (lastSlash-filename)+1;

							// copy base path (including slash to relative path)
							memcpy(relativePath, filename, baseLength);
						}

						// append mesh filename
						strcpy(relativePath + baseLength, path);

						// import mesh
						Mesh* mesh = ImportMeshFromObj(relativePath);

						if (mesh)
						{
							mesh->Normalize();
							mesh->CalculateNormals();
							mesh->RebuildBVH();

							primitive.mesh = GeometryFromMesh(mesh);

							meshes[path] = mesh;
						}
						else
						{
							printf("Failed to import mesh %s", path);
							fflush(stdout);
						}
					}
				}

				if (strstr(line, "transform"))
				{
					Mat44 m;

					/*
					todo: decompose transform

					for (int row=0; row < 4; ++row)
					{
						if (fgets(line, kMaxLineLength, file))
						{
							sscanf(line, " %f %f %f %f", 
								&m.cols[0][row],
								&m.cols[1][row],
								&m.cols[2][row],
								&m.cols[3][row]);
						}
					}

					primitive.transform = m;
					primitive.lastTransform = m;
					*/
				}
			}

			// add to scene
			scene->AddPrimitive(primitive);
		}

		//--------------------------------------------
		// Mesh

		if (sscanf(line, " mesh %s", name) == 1)
		{
			Mesh* mesh = new Mesh();

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				int numVerts = 0;
				if (sscanf(line, " verts %d", &numVerts) == 1)
				{
					mesh->positions.resize(numVerts);
					mesh->normals.resize(numVerts);

					for (int i=0; i < numVerts; ++i)
					{
						if (fgets(line, kMaxLineLength, file))
						{
							Vec3 v;
							sscanf(line, " %f %f %f", &v.x, &v.y, &v.z);

							mesh->positions[i] = v;
						}
					}
				}

				int numTris = 0;
				if (sscanf(line, " tris %d", &numTris) == 1)
				{
					for (int i=0; i < numTris; ++i)
					{
						if (fgets(line, kMaxLineLength, file))
						{
							int a, b, c;
							
							sscanf(line, " %d %d %d", &a, &b, &c);

							mesh->indices.push_back(a);
							mesh->indices.push_back(b);
							mesh->indices.push_back(c);
						}
					}
				}
			}

			mesh->CalculateNormals();
			mesh->RebuildBVH();

			meshes[name] = mesh;
		}
	}

	return scene;
}