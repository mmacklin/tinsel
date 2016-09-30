#pragma once

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
        sphere.material.color = Color(.82f, .67f, .16f);
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
        sphere.material.color= SrgbToLinear(Color(.05f, .57f, .36f));
        sphere.material.metallic = 0.0f;
		sphere.material.specular = 0.75f;//float(i)/(rowSize-1);
		sphere.material.roughness = Max(0.01f, Sqr(1.0f-float(i)/(rowSize-1)));

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
        //sphere.material.reflectance = Color(.82f, .67f, .16f);
        sphere.material.subsurface = float(i)/(rowSize-1);
		sphere.material.color = SrgbToLinear(Color(0.7f));
		sphere.material.specular = 0.0f;

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	Material gold;
	gold.color = Color(1.0f, 0.71f, 0.29f);
	gold.roughness = 0.2f;
	gold.metallic = 1.0f;

    Material silver;
	silver.color = Color(0.95f, 0.93f, 0.88f);
	silver.roughness = 0.2f;
	silver.metallic = 1.0f;

    Material copper;
	copper.color = Color(0.95f, 0.64f, 0.54f, 120.0f);
	copper.roughness = 0.2f;
	copper.metallic = 1.0f;

    Material iron;
	iron.color = Color(0.56f, 0.57f, 0.58f, 120.0f);
	iron.roughness = 0.2f;
	iron.metallic = 1.0f;

    Material aluminum;
	aluminum.color = Color(0.91f, 0.92f, 0.92f, 120.0f);
	aluminum.roughness = 0.2f;
	aluminum.metallic = 1.0f;

    Material plaster;
	plaster.color = Color(0.94f, 0.94f, 0.94f);
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
    plane.material.color = Color(0.1);

    Primitive back;
    back.type = ePlane;
    back.plane.plane[0] = 0.0f;
    back.plane.plane[1] = 0.0f;
    back.plane.plane[2] = 1.0f;
    back.plane.plane[3] = 5.0f;
    back.material.color = Color(0.1);

    Primitive light;
    light.type = eSphere;
    light.sphere.radius = 1.0f;
    light.startTransform = Transform(Vec3(0.0f, 6.0f, 6.0f));
	light.endTransform = light.startTransform;
    light.material.color = Vec4(0.0f);
    light.material.emission = Vec4(15.0f);
    light.lightSamples = 1;

    
    scene->AddPrimitive(plane);
	scene->AddPrimitive(back);
    scene->AddPrimitive(light);

	scene->sky.horizon = Color(0.1f, 0.3f, 0.6f)*2.0f;
    scene->sky.zenith = scene->sky.horizon;

    // set up camera
    camera->position = Vec3(0.0f, 2.0f, 10.0f);
}


