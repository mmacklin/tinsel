
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

void ReadParam(cJSON* object, const char* name, int& x)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param)
		x = param->valueint;

}


void ReadParam(cJSON* root, const char* name, int& x, int& y)
{
	cJSON* param = cJSON_GetObjectItem(root, name);
	if (param && param->child)
	{
		cJSON* p = param->child;

		x = p->valueint; p = p->next;
		y = p->valueint; p = p->next;
	}

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

void ReadParam(cJSON* object, const char* name, Vec3& out)
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

void ReadParam(cJSON* object, const char* name, Transform& t, Vec3& scale)
{
	cJSON* param = cJSON_GetObjectItem(object, name);
	if (param)
	{
		ReadParam(param, "position", t.p);
		
		Vec3 r;
		ReadParam(param, "rotation", r);
		
		ReadParam(param, "scale", scale);

		//t.r = Quat(Vec3(0.0f, 1.0f, 0.0f), DegToRad(r.x))*Quat(Vec3(1.0f, 0.0f, 0.0f), DegToRad(r.y))*Quat(Vec3(0.0f, 0.0f, 1.0f), DegToRad(r.z));
		t.r = Quat(Vec3(0.0f, 1.0f, 0.0f), DegToRad(r.y))* 
			Quat(Vec3(1.0f, 0.0f, 0.0f), DegToRad(r.x))*
			  
			  Quat(Vec3(0.0f, 0.0f, 1.0f), DegToRad(r.z));
	}
}

void ReadMaterial(cJSON* node, Material& material, std::string& materialName, std::string& materialType)
{
	bool refraction = false;

	ReadParam(node, "name", materialName);
	ReadParam(node, "type", materialType);
	ReadParam(node, "albedo", material.color);
	ReadParam(node, "ior", material.eta);
	ReadParam(node, "roughness", material.roughness);
	ReadParam(node, "enable_refraction", refraction);

	//material.color = SrgbToLinear(material.color);


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
		material.specular = 1.0f;
	}

	if (materialType == "thinsheet")
	{
		material.transmission = 1.0f;
	}

	if (materialType == "dielectric")
	{
		material.roughness = 0.0f;
	}

	if (materialType == "null")
	{
		material.color = 0.0f;
		material.specular = 0.0f;
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

	if (materialType == "lambert")
	{
		material.specular = 0.0f;
		material.roughness = 1.0f;
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
				
				ReadMaterial(node, material, materialName, materialType);

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
				
				Vec3 scale;

				ReadParam(node, "type", type);
				ReadParam(node, "file", path);
				ReadParam(node, "bsdf", bsdf);
				ReadParam(node, "transform", primitive.startTransform, scale);
				ReadParam(node, "transform", primitive.endTransform, scale);

				primitive.material = materials[bsdf];

				// apply emission to primitive's copy of the material
				ReadParam(node, "emission", primitive.material.emission);
				if (LengthSq(primitive.material.emission) > 0.0f)
					primitive.lightSamples = 1;

				// inline bsdf
				cJSON* bsdfNode = cJSON_GetObjectItem(node, "bsdf");
				if (bsdf == "" && bsdfNode->child)
				{
					std::string name;
					std::string type;

					ReadMaterial(bsdfNode, primitive.material, name, type);
				}

				if (type == "quad")
				{
					Mesh* quad = CreateQuadMesh(0.5f, 0.0f);
					

					for (int i=0; i < 4; ++i)
					{
						quad->positions[i].x *= scale.x;
						quad->positions[i].y *= scale.y;
						quad->positions[i].z *= scale.z;
					}

					quad->RebuildBVH();

					primitive.type = eMesh;			
					primitive.mesh = GeometryFromMesh(quad);

					scene->primitives.push_back(primitive);

				}
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
							printf("Failed to import mesh %s\n", relativePath);//.c_str());
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
			ReadParam(root, "resolution", options->width, options->height);

			//options->width /= 2;
			//options->height /= 2;
			
			cJSON* transform = cJSON_GetObjectItem(root, "transform");

			Vec3 pos, target, up;

			ReadParam(transform, "position", pos);
			ReadParam(transform, "look_at", target);
			ReadParam(transform, "up", up);

			// set up camera
			Mat44 lookat = AffineInverse(LookAtMatrix(pos, target));
			Mat33 m = Mat33(Vec3(lookat.GetCol(0)), Vec3(lookat.GetCol(1)), Vec3(lookat.GetCol(2)));
			Quat q = Quat(m);
			camera->position = pos;
			camera->rotation = Normalize(q);

			float fov = camera->fov;
			ReadParam(root, "fov", fov);
			camera->fov = DegToRad(fov)*(options->height/float(options->width));

			printf("fov: %f\n", fov);
		}

		if (strcmp(root->string, "integrator") == 0)
		{
			ReadParam(root, "max_bounces", options->maxDepth);
		}

		if (strcmp(root->string, "renderer") == 0)
		{
			ReadParam(root, "spp", options->maxSamples);
		}
    
		root = root->next;
	}

	/*
	Primitive light;
	light.type = eSphere;
	light.sphere.radius = 1.0f;
	light.startTransform.p = Vec3(-10.0f, 10.0f, 2.0f);
	light.endTransform.p = Vec3(-10.0f, 10.0f, 2.0f);
	light.lightSamples = 1;
	light.material.emission = 200.0f;
	light.material.color = 0.0f;
	light.material.specular = 0.0f;

	scene->primitives.push_back(light);
	*/
	
	//scene->sky.probe = ProbeLoadFromFile("data/probes/vankleef.hdr");

	//options->maxDepth = 8;
	
	options->maxSamples = 100000;
	
	//options->width /= 4;
	//options->height /= 4;


	return true;
}