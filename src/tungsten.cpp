
#include "scene.h"
#include "mesh.h"
#include "maths.h"
#include "render.h"
#include "util.h"
#include "pfm.h"

#include "cjson/cjson.h"

#include <map>



void ReadParam(cJSON* object, const char* name, std::string& out)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param && param->valuestring)
		out = param->valuestring;	
}

void ReadParam(cJSON* object, const char* name, float& out)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param)
		out = param->valuedouble;
}

void ReadParam(cJSON* object, const char* name, bool& out)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param)
		out = param->valueint;
}

void ReadParam(cJSON* object, const char* name, Color& out)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param)
	{
		if (param->child)
		{
			cJSON* p = param->child;

			out.x = p->valuedouble; p = p->next;
			out.y = p->valuedouble; p = p->next;
			out.z = p->valuedouble; p = p->next;
		}
		else
		{
			out = param->valuedouble;
		}
	}
}


bool LoadTungsten(const char* filename, Scene* scene, Camera* camera, Options* options)
{
	FILE* f = fopen(filename, "r");

	fseek(f, 0, SEEK_END);
	int length = ftell(f);
	rewind(f);

	char* buffer = new char[length];
	fread(buffer, length, 1, f);
	fclose(f);

	cJSON* root = cJSON_Parse(buffer);

	std::map<std::string, Mesh*> meshes;
	std::map<std::string, Material> materials;

	root = root->child;

	while (root)
	{
		if (strcmp(root->string, "bsdfs") == 0)
		{
			cJSON* node = root->child;
			while (node)
			{
				Material material;
				std::string materialName;
				std::string materialType;
				bool refraction = false;

				ReadParam(node, "name", materialName);
				ReadParam(node, "type", materialType);
				ReadParam(node, "albedo", material.color);
				ReadParam(node, "ior", material.eta);
				ReadParam(node, "roughness", material.roughness);
				ReadParam(node, "enable_refraction", refraction);

				if (materialName == "RoughSteel")
				{
					material.color = 0.05f;
					material.specular = 1.0f;
				}

				if (refraction)
					material.transmission = 1.0f;

				if (materialType == "plastic")
				{
					material.metallic = 0.0f;
					material.roughness = 0.0f;
					material.specular = 0.8f;
				}

				if (materialType == "dielectric")
				{
					material.roughness = 0.0f;
				}

				if (materialType == "mirror")
				{
					material.specular = 1.0f;
					material.metallic = 1.0f;
					material.roughness = 0.0f;
				}

				if (materialType == "rough_dielectric" || materialType == "rough_plastic")
					material.metallic = 0.0f;
				if (materialType == "rough_conductor")
					material.metallic = 1.0f;

				materials[materialName] = material;

				// next mat
				node = node->next;
			}
		}
		if (strcmp(root->string, "primitives") == 0)
		{
			cJSON* node = root->child;
			while (node)
			{		
				Primitive primitive;			

				std::string type;
				std::string path;
				std::string bsdf;
				
				ReadParam(node, "type", type);
				ReadParam(node, "file", path);
				ReadParam(node, "bsdf", bsdf);

				primitive.material = materials[bsdf];

				// apply emission to primitive's copy of the material
				ReadParam(node, "emission", primitive.material.emission);
				if (LengthSq(primitive.material.emission) > 0.0f)
					primitive.lightSamples = 1;

				if (type == "mesh")
				{
					primitive.type = eMesh;

					// look up in the mesh array
					if (meshes.find(path) != meshes.end())
					{
						primitive.mesh = GeometryFromMesh(meshes[path]);
					}
					else
					{
						char relativePath[2048];

						// make relative path to .tin
						MakeRelativePath(filename, path.c_str(), relativePath);

						// import mesh
						Mesh* mesh = ImportMesh(relativePath);

						if (mesh)
						{
							primitive.mesh = GeometryFromMesh(mesh);							

							meshes[path] = mesh;
						}
						else
						{
							printf("Failed to import mesh %s", path.c_str());
							fflush(stdout);
						}
					}

					scene->primitives.push_back(primitive);
				}

				// next prim
				node = node->next;
			}
		}
		if (strcmp(root->string, "camera") == 0)
		{
			//ParseCamera(root->child);
		}

		root = root->next;
	}

	Primitive light;
	light.type = eSphere;
	light.sphere.radius = 1.0f;
	light.startTransform.p = Vec3(-10.0f, 10.0f, 2.0f);
	light.endTransform.p = Vec3(-10.0f, 10.0f, 2.0f);
	light.lightSamples = 1;
	light.material.emission = 200.0f;
	light.material.color = 0.0f;
	light.material.specular = 0.0f;

	options->maxDepth = 64;

	scene->primitives.push_back(light);


	return true;
}