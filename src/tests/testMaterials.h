#pragma once

#include "../probe.h"
#include "../tga.h"

void TestMaterials(Scene* scene, Camera* camera, Options* options)
{

    // add primitives
	const int rowSize = 6;

	float r = 0.5f;
	float y = r;
	
	float dx = r*2.2f;
	float x = (rowSize-1)*r*2.2f*0.5f;	

    for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.startTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.endTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.color = Vec3(.82f, .67f, .16f);
        sphere.material.metallic = float(i)/(rowSize-1);
        sphere.material.roughness = 0.25f;

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.startTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.endTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.color = Vec3(SrgbToLinear(Color(.05f, .57f, .36f)));
        sphere.material.metallic = 0.0f;

        float shiny = Max(0.0f, Sqr(1.0f-float(i)/(rowSize-1)));
		sphere.material.specular = 0.75f;
		sphere.material.roughness = shiny;
        
        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.startTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.endTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.color = Vec3(SrgbToLinear(Color(0.9)));
        sphere.material.metallic = 0.0f;
		sphere.material.transmission = float(i)/(rowSize-1);		
		sphere.material.roughness = 0.01f;
        
        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	/*
    for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.startTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.endTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.subsurface = float(i)/(rowSize-1);
		sphere.material.color = SrgbToLinear(Color(0.7f));
		sphere.material.specular = 0.0f;

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;
	*/

	Material gold;
	gold.color = Vec3(1.0f, 0.71f, 0.29f);
	gold.roughness = 0.2f;
	gold.metallic = 1.0f;

    Material silver;
	silver.color = Vec3(0.95f, 0.93f, 0.88f);
	silver.roughness = 0.2f;
	silver.metallic = 1.0f;

    Material copper;
	copper.color = Vec3(0.95f, 0.64f, 0.54f);
	copper.roughness = 0.2f;
	copper.metallic = 1.0f;

    Material iron;
	iron.color = Vec3(0.56f, 0.57f, 0.58f);
	iron.roughness = 0.2f;
	iron.metallic = 1.0f;

    Material aluminum;
	aluminum.color = Vec3(0.91f, 0.92f, 0.92f);
	aluminum.roughness = 0.2f;
	aluminum.metallic = 1.0f;

    Material plaster;
	plaster.color = Vec3(0.94f, 0.94f, 0.94f);
	plaster.roughness = 0.5;
	plaster.specular = 0.1;

	Material mats[6] = { gold, silver, copper, iron, aluminum, plaster };

	for (int i=0; i < 6; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.startTransform = Transform(Vec3(-x + i*dx, y, 0.0f));
        sphere.endTransform = Transform(Vec3(-x+ i*dx, y, 0.0f));
        sphere.material = mats[i];
        
        scene->AddPrimitive(sphere);
    }


    Primitive plane;
    plane.type = ePlane;
    plane.plane.plane[0] = 0.0f;
    plane.plane.plane[1] = 1.0f;
    plane.plane.plane[2] = 0.0f;
    plane.plane.plane[3] = 0.0f;
    plane.material.color = Vec3(0.5);

    Primitive back;
    back.type = ePlane;
    back.plane.plane[0] = 0.0f;
    back.plane.plane[1] = 0.0f;
    back.plane.plane[2] = 1.0f;
    back.plane.plane[3] = 5.0f;
    back.material.color = Vec3(0.1);

    Primitive light;
    light.type = eSphere;
    light.sphere.radius = 1.0f;
    light.startTransform = Transform(Vec3(0.0f, 6.0f, 6.0f));
	light.endTransform = light.startTransform;
    light.material.color = Vec3(0.0f);
    light.material.emission = Vec3(15.0f);
    light.lightSamples = 1;

    
    scene->AddPrimitive(plane);
	scene->AddPrimitive(back);
	//scene->AddPrimitive(light);

	//scene->sky.horizon = Color(0.1f, 0.3f, 0.6f)*2.0f;
    //scene->sky.zenith = scene->sky.horizon;

    scene->sky.probe = ProbeLoadFromFile("data/probes/vankleef.hdr");


    // set up camera
    camera->position = Vec3(0.0f, 2.0f, 20.0f);
    camera->fov = DegToRad(15.0f);
}


void TestPaniq(Scene* scene, Camera* camera, Options* options)
{
#if 0
	Vec3 colors[16] =
	{
		Vec3(1.0f, 0.254f, 0.287f),		// light pink
		Vec3(1.0f, 0.823f, 0.036f),		// yellow
		Vec3(0.209f, 1.0f, 0.521f),		// light blue
		Vec3(0.371f, 0.027f, 0.456f),	// purple
		
		Vec3(1.0f, 0.019f, 0.051f),		// bright pink
		Vec3(0.0f, 0.708f, 0.072f),		// bright green
		Vec3(0.0f, 0.275f, 0.730f),		// mid blue
		Vec3(1.0f, 0.305f, 0.026f),		// orange

		Vec3(0.168f, 0.0f, 0.01f),		// dark red
		Vec3(0.0f, 0.178f, 0.018f),		// dark green
		Vec3(0.015f, 0.016f, 0.178f),	// dark blue
		Vec3(0.546f, 0.082f, 0.041f),	// dark orange

		Vec3(0.0f, 0.0f, 0.0f),			// black
		Vec3(0.06f, 0.091f, 0.130f),	// dark grey
		Vec3(0.491f, 0.491f, 0.491f),	// light grey
		Vec3(1.0f, 1.0f, 1.0f)			// white
	};

	float radius = 1.0f;
	float spacing = 2.5f;

	Mesh* obj = ImportMeshFromObj("data/brain.obj");
	obj->Normalize(1.f);
	MeshGeometry mesh = GeometryFromMesh(obj);

	for (int y=0; y < 4; ++y)
	{
		for (int x=0; x < 4; ++x)
		{
			Primitive sphere;
			//sphere.type = eSphere;
			//sphere.sphere.radius = radius;
			sphere.type = eMesh;
			sphere.mesh = mesh;
			sphere.startTransform = Transform(Vec3(x*spacing, y*spacing, 0.0f));
			sphere.endTransform = Transform(Vec3(x*spacing, y*spacing, 0.0f));
			sphere.material.color = colors[y*4 + x];
			sphere.material.metallic = 0.0f;
			sphere.material.roughness = 0.01f;
			//sphere.material.clearcoat = 1.0f;
			//sphere.material.transmission = 0.0f;
			//sphere.material.absorption = Max(0.0f, Vec3(0.75f)-sphere.material.color);//Vec3(sqrtf(sphere.material.color.x), sqrtf(sphere.material.color.y), sqrtf(sphere.material.color.z))*0.5f;

			scene->AddPrimitive(sphere);
		}
	}
	
	float center = 3.0f*spacing*0.5f;

#else

	TgaImage img;
	TgaLoad("data/palette.tga", img);
	
	
	float radius = 1.0f;
	float spacing = 2.5f;

	Mesh* obj = ImportMeshFromObj("data/meshes/brain.obj");
	obj->Normalize(2.f);
	obj->Transform(TranslationMatrix(Vec3(-1.0f)));
	obj->RebuildBVH();
	//MeshGeometry mesh = GeometryFromMesh(obj);

	for (int y=0; y < img.m_height; ++y)
	{
		for (int x=0; x < img.m_width; ++x)
		{
			Primitive sphere;
			//sphere.type = eSphere;
			//sphere.sphere.radius = radius;

			sphere.type = eMesh;
			sphere.mesh = GeometryFromMesh(obj);

			sphere.startTransform = Transform(Vec3(x*spacing, y*spacing, 0.0f));
			sphere.endTransform = Transform(Vec3(x*spacing, y*spacing, 0.0f));
			
			unsigned int c = img.m_data[y*img.m_width + x];

			float r = (c>>0)&0xff;
			float g = (c>>8)&0xff;
			float b = (c>>16)&0xff;

			printf("%f %f %f\n", r, g, b);

			Color col(r/255.0f, g/255.0f, b/255.0f);
			col = SrgbToLinear(col);

			sphere.material.color = Vec3(col);
			//sphere.material.specular = 0.0f;
			//sphere.material.emission = Vec3(col)*0.25f;
			sphere.material.metallic = 0.0f;
			sphere.material.roughness = 0.01f;
			sphere.material.clearcoat = 0.0f;
			sphere.material.clearcoatGloss = 1.0f;		
			sphere.material.subsurface = 0.0f;
			//sphere.material.transmission = 1.0f;
			//sphere.material.absorption = Max(0.0f, Vec3(0.75f)-sphere.material.color);//Vec3(sqrtf(sphere.material.color.x), sqrtf(sphere.material.color.y), sqrtf(sphere.material.color.z))*0.5f;

			scene->AddPrimitive(sphere);
		}
	}

	TgaFree(img);

	float center = (img.m_width-1)*spacing*0.5f;

#endif





    Primitive plane;
    plane.type = ePlane;
    plane.plane.plane[0] = 0.0f;
    plane.plane.plane[1] = 1.0f;
    plane.plane.plane[2] = 0.0f;
    plane.plane.plane[3] = 1.0f;
    plane.material.color = Vec3(0.8);

    Primitive back;
    back.type = ePlane;
    back.plane.plane[0] = 0.0f;
    back.plane.plane[1] = 0.0f;
    back.plane.plane[2] = 1.0f;
    back.plane.plane[3] = 1.5f;
    back.material.color = Vec3(0.8);

    Primitive light;
    light.type = eSphere;
    light.sphere.radius = 1.0f;
    light.startTransform = Transform(Vec3(10.f, 15.0f, 15.0f));
	light.endTransform = light.startTransform;
    light.material.color = Vec3(0.0f);
    light.material.emission = Vec3(150.0f);
    light.lightSamples = 1;

    
    //scene->AddPrimitive(plane);
	scene->AddPrimitive(back);
	//scene->AddPrimitive(light);

	//scene->sky.horizon = Color(0.1f, 0.3f, 0.6f)*2.0f;
    //scene->sky.zenith = scene->sky.horizon;

    scene->sky.probe = ProbeLoadFromFile("data/probes/nature.hdr");
	
	options->width = 1920;
	options->height = options->width/2;
	//options->clamp = 2.0f;

    // set up camera
    camera->position = Vec3(center, 3.75f, 40.0f);
	camera->rotation = Quat();
    camera->fov = DegToRad(15.0f);
}

