#pragma once

void TestVeach(Scene* scene, Camera* camera, Options* options)
{
    Material background;
	background.color = Color(0.7f, 0.7f, 0.7f);
	background.roughness = 0.75;
	background.specular = 0.1;

    Material gloss;
    gloss.color = Color(0.6f, 0.6f, 0.6f);
    gloss.roughness = 0.1f;
	gloss.specular = 0.75f;
    gloss.metallic = 1.0f;

    Primitive ground;
    ground.type = ePlane;
    ground.plane.plane[0] = 0.0f;
    ground.plane.plane[1] = 1.0f;
    ground.plane.plane[2] = 0.0f;
    ground.plane.plane[3] = 0.0f;
    ground.material = background;

    Primitive back;
    back.type = ePlane;
    back.plane.plane[0] = 0.0f;
    back.plane.plane[1] = 0.0f;
    back.plane.plane[2] = 1.0f;
    back.plane.plane[3] = 3.0f;
    back.material = background;

	Vec3 verts[4] = 
	{ 
		Vec3(-1, 0, 0.25),
		Vec3(1, 0, 0.25),
		Vec3(1, 0, -0.25),
		Vec3(-1, 0, -0.25)
	};

	int indices[6] = 
	{
		0, 1, 2, 
		0, 2, 3
	};

	Mesh* plateMesh = new Mesh();
	plateMesh->positions.assign(verts, verts+4);
	plateMesh->indices.assign(indices, indices+6);
	
	plateMesh->CalculateNormals();
	plateMesh->RebuildBVH();

    // set up camera
    camera->position = Vec3(0.0f, 2.0f, 3.0f);
	camera->rotation = Quat(Vec3(1.0f, 0.0f, 0.0f), -DegToRad(15.0f));

	Vec3 lightsCenter = Vec3(0.0f, 1.75f, -2.0f);

	for (int i=0; i < 4; ++i)
	{
		Primitive plate;

		// distribute planes on a circle
		float a = i*DegToRad(7.5f);
		float radius = 5.0f;
		Vec3 pos = Vec3(0.0f, 0.5f + radius-cosf(a)*radius, -sinf(a)*radius);
				
		// orient planes so that half angle reflects view rays directly to lights
		Vec3 wi = lightsCenter-pos;
		Vec3 wo = camera->position-pos;
		Vec3 half = Normalize(Normalize(wi) + Normalize(wo));
		float angle = atanf(half.z/half.y);

		plate.startTransform = Transform(pos, Quat(Vec3(1.0f, 0.0f, 0.0f), angle));
		plate.endTransform = plate.startTransform;

		plate.material = gloss;
		plate.material.roughness = Sqr(Lerp(0.3f, 0.01f, i/3.0f));
		plate.type = eMesh;
		plate.mesh = GeometryFromMesh(plateMesh);
		
		scene->AddPrimitive(plate);
	}


	// lights
	Color colors[4] =
	{
		SrgbToLinear(Color(250.0f, 180.0f, 220.0f)/255.f),
		SrgbToLinear(Color(250.0, 240.0, 170.0)/255.f),
		SrgbToLinear(Color(180.0, 250.0, 170.0)/255.f),
		SrgbToLinear(Color(120.0, 150.0, 220.0)/155.f)
	};


	float radii[4] = { 0.01f, 0.03f, 0.07f, 0.2f };

	for (int i=0; i < 4; ++i)
	{
		float area = 4.0f*kPi*radii[i]*radii[i];
		float power = 1.0f;

		Material mat;
		mat.emission = colors[i]*power/area;
		mat.color = 0.0f;
		
		Primitive light;
		light.type = eSphere;
		light.sphere.radius = radii[i];
		light.startTransform = Transform(lightsCenter + Vec3(-1.0f + 2.0f*i/3.0f, 0.0f, 0.0f));
		light.endTransform = light.startTransform;
		light.material = mat;
		light.lightSamples = 1;

		scene->AddPrimitive(light);
	}

    Primitive light;
	light.type = eSphere;
	light.sphere.radius = 0.1f;
	light.startTransform = Transform(Vec3(3.0f, 7.0f, 7.0f));
	light.endTransform = light.startTransform;
	light.material.emission = Color(10.0f, 10.0f, 10.0f)*200.0f;
	light.lightSamples = 1;
	
	scene->AddPrimitive(light);
	scene->AddPrimitive(ground);
    //scene->AddPrimitive(back);

	// original dimensions from Veach's paper
	options->width = 500;
	options->height = 450;
	options->exposure = 0.5f;

	// only used as a poor mans tone mapping
	//options->clamp = 4.0f;
}

