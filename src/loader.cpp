#include "loader.h"

#include "scene.h"
#include "mesh.h"
#include "maths.h"
#include "render.h"
#include "util.h"
#include "pfm.h"

#include <stdio.h>

#include <map>
#include <string>

static const int kMaxLineLength = 2048;

void MakeRelativePath(const char* filePath, const char* fileRelativePath, char* fullPath)
{
	// get base path of file
	const char* lastSlash = NULL;

	if (!lastSlash)
		lastSlash = strrchr(filePath, '\\');
	if (!lastSlash)
		lastSlash = strrchr(filePath, '/');

	int baseLength = 0;

	if (lastSlash)
	{
		baseLength = (lastSlash-filePath)+1;

		// copy base path (including slash to relative path)
		memcpy(fullPath, filePath, baseLength);
	}

	// append mesh filename
	strcpy(fullPath + baseLength, fileRelativePath);
}


bool LoadTin(const char* filename, Scene* scene, Camera* camera, Options* options)
{
	FILE* file = fopen(filename, "r");

	if (!file)
	{
		printf("Couldn't open %s for reading.", filename);
		return false;
	}

	std::map<std::string, Mesh*> meshes;
	std::map<std::string, Material> materials;

	char line[kMaxLineLength];

	while (fgets(line, kMaxLineLength, file))
	{
		// skip comments
		if (line[0] == '#')
			continue;

		// name used for materials and meshes
		char name[kMaxLineLength] = { 0 };

		//--------------------------------------------
		// Include files

		if (sscanf(line, " include %s", name) == 1)
		{
			assert(0);

		}

		//--------------------------------------------
		// Options
		if (strstr(line, "options"))
		{
			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " width %d", &options->width);
				sscanf(line, " height %d", &options->height);
				sscanf(line, " maxSamples %d", &options->maxSamples);
				sscanf(line, " maxDepth %d", &options->maxDepth);


				sscanf(line, " clamp %f", &options->clamp);
				sscanf(line, " limit %f", &options->limit);
				sscanf(line, " exposure %f", &options->exposure);


				char type[kMaxLineLength];
				sscanf(line, " filter %s %f %f", type, &options->filter.width, &options->filter.falloff);
				if (strcmp(type, "box") == 0)
					options->filter.type = eFilterBox;
				if (strcmp(type, "gaussian") == 0)
					options->filter.type = eFilterGaussian;				
			}
		}

		//--------------------------------------------
		// Camera

		if (strstr(line, "camera"))
		{
			Vec3 target;
			bool targetValid = false;

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " position %f %f %f", &camera->position.x, &camera->position.y, &camera->position.z);
				sscanf(line, " rotation %f %f %f %f", &camera->rotation.x, &camera->rotation.y, &camera->rotation.z, &camera->rotation.w);
				
				if (sscanf(line, " target %f %f %f", &target.x, &target.y, &target.z) == 3)
					targetValid = true;
				
				if (sscanf(line, " fov %f", &camera->fov) == 1)
					camera->fov = DegToRad(camera->fov);

				sscanf(line, " shutterstart %f", &camera->shutterStart);
				sscanf(line, " shutterend %f", &camera->shutterEnd);

				// todo: load transform directly
			}

			printf("shutter [%f, %f]\n", camera->shutterStart, camera->shutterEnd);
	

			if (targetValid)
			{
				Mat44 lookat = AffineInverse(LookAtMatrix(camera->position, target));
				Mat33 m = Mat33(Vec3(lookat.GetCol(0)), Vec3(lookat.GetCol(1)), Vec3(lookat.GetCol(2)));
				Quat q = Quat(m);
				camera->rotation = Normalize(q);
			}
		}

		//-------------------------------------------
		// Sky

		if (strstr(line, "sky"))
		{
			Sky sky;

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;
	
				sscanf(line, " horizon %f %f %f", &sky.horizon.x, &sky.horizon.y, &sky.horizon.z);
				sscanf(line, " zenith %f %f %f", &sky.zenith.x, &sky.zenith.y, &sky.zenith.z);

				char probeName[kMaxLineLength];
				if (sscanf(line, " probe %s", probeName) == 1)
				{
					char path[kMaxLineLength];
					MakeRelativePath(filename, probeName, path);

					sky.probe = ProbeLoadFromFile(path);
				}
			}

			scene->sky = sky;
		}


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

			bool valid = true;

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

				int count = 0;

				count = sscanf(line, " position %f %f %f , %f %f %f", 
					&primitive.startTransform.p.x, 
					&primitive.startTransform.p.y, 
					&primitive.startTransform.p.z,
					&primitive.endTransform.p.x, 
					&primitive.endTransform.p.y, 
					&primitive.endTransform.p.z);

				if (count > 0 && count < 6)
					primitive.endTransform.p = primitive.startTransform.p;

				count = sscanf(line, " rotation %f %f %f %f , %f %f %f %f",
					&primitive.startTransform.r.x,
					&primitive.startTransform.r.y,
					&primitive.startTransform.r.z, 
					&primitive.startTransform.r.w,
					&primitive.endTransform.r.x,
					&primitive.endTransform.r.y,
					&primitive.endTransform.r.z, 
					&primitive.endTransform.r.w);
		
				if (count > 0 && count < 8)
					primitive.endTransform.r = primitive.startTransform.r;

				count = sscanf(line, " scale %f , %f", 
					&primitive.startTransform.s,
					&primitive.endTransform.s);

				if (count > 0 && count < 2)
					primitive.endTransform.s = primitive.startTransform.s;

				sscanf(line, " radius %f", &primitive.sphere.radius);
				sscanf(line, " plane %f %f %f %f", &primitive.plane.plane[0], &primitive.plane.plane[1], &primitive.plane.plane[2], &primitive.plane.plane[3]);
				sscanf(line, " lightSamples %d", &primitive.lightSamples);

				char path[2048];

				if (sscanf(line, " material %s", path) == 1)
				{
					// look up material in dictionary
					if (materials.find(path) != materials.end())
					{
						primitive.material = materials[path];
					}
					else
					{
						printf("Could not find material %s\n", path);
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

						// make relative path to .tin
						MakeRelativePath(filename, path, relativePath);

						// import mesh
						Mesh* mesh = ImportMesh(relativePath);

						if (mesh)
						{
							primitive.mesh = GeometryFromMesh(mesh);							

							meshes[path] = mesh;
						}
						else
						{
							printf("Failed to import mesh %s", path);
							fflush(stdout);

							valid = false;
						}
					}
				}
			}

			// add to scene
			if (valid)
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

	return true;
}