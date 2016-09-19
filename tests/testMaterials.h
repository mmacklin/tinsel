#pragma once

Scene* TestMaterials()
{
    Scene* scene = new Scene();


    /*
    // default materials
    Material lambert = Material(Color(0.6f, 0.55f, 0.53f));    
    Material blinnPhong = Material(Color(0.5f, 0.3f, 0.3f), 120.0f);
    
    Material gold = Material(Color(1.0f, 0.71f, 0.29f, 120.0f));
    Material silver = Material(Color(0.95f, 0.93f, 0.88f, 120.0f));
    Material copper = Material(Color(0.95f, 0.64f, 0.54f, 120.0f));
    Material iron = Material(Color(0.56f, 0.57f, 0.58f, 120.0f));
    Material aluminum = Material(Color(0.91f, 0.92f, 0.92f, 120.0f));
    Material plaster = Material(Color(0.94f, 0.94f, 0.94f));
    */
    //Mesh* mesh = ImportMeshFromObj("../../data/happy.obj");
    //Mesh* mesh = ImportMeshFromPly("../../data/thearena.ply");
    //Mesh* mesh = ImportMeshFromPly("models/bunny/reconstruction/bun_zipper_res4.ply");
    //Mesh* mesh = ImportMeshFromPly("models/happy_recon/happy_vrip_res3.ply");
    //Mesh* mesh = ImportMeshFromPly("models/dragon/xyzrgb_dragon.ply"); yoffset = 22.1f;
    //Mesh* mesh = ImportMeshFromPly("models/xyzrgb_statuette.ply"); 
    //Mesh* mesh = ImportMeshFromObj("models/elephant/elefant_frojotero_com.obj"); mesh->Transform(RotationMatrix(DegToRad(-150.0f), Vec3(1.0f, 0.0f, 0.0f)));
    //Mesh* mesh = ImportMeshFromObj("models/ajax/Ajax_Jotero_com.obj");
    //Mesh* mesh = ImportMeshFromObj("models/aphrodite/Aphrodite_frojotero_com.obj");
    //Mesh* mesh = ImportMeshFromObj("models/figure/figure_frojotero_com.obj");
    //Mesh* mesh = ImportMeshFromObj("models/sanktkilian/Sankt_Kilian_Sockel_jotero_com.obj"); mesh->Transform(RotationMatrix(DegToRad(-90.0f), Vec3(1.0f, 0.0f, 0.0f)));
    //Mesh* mesh = ImportMeshFromPly("models/lucy.ply"); mesh->Transform(RotationMatrix(DegToRad(-90.0f), Vec3(1.0f, 0.0f, 0.0f)));

    //Mesh* lightMesh = ImportMeshFromPly("models/bunny/reconstruction/bun_zipper_res4.ply");
    //lightMesh->Transform(TranslationMatrix(Vec3(-0.0f, 0.5f, 0.0f)));
/*
    // move at 5% of diagonal
    Vec3 minExtent, maxExtent, edgeLength, center;
    mesh->GetBounds(minExtent, maxExtent);
    
    // offset by 
    maxExtent.y += yoffset;
    minExtent.y += yoffset;
    edgeLength = 0.5f*(maxExtent-minExtent);

    center = 0.5f*(minExtent+maxExtent);

    // set mouse speed based on mesh size
    g_flySpeed = 0.05f*Length(maxExtent-minExtent);   
*/
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
        sphere.transform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.lastTransform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.color = Color(.82f, .67f, .16f);
        sphere.material.metallic = float(i)/(rowSize-1);

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.transform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.lastTransform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.material.color= SrgbToLinear(Color(.05f, .57f, .36f));
        sphere.material.metallic = 0.0f;
		//sphere.material.specular = float(i)/(rowSize-1);
		sphere.material.roughness = Max(0.2f, 1.0f-float(i)/(rowSize-1));

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

    for (int i=0; i < rowSize; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.transform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.lastTransform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        //sphere.material.reflectance = Color(.82f, .67f, .16f);
        sphere.material.subsurface = float(i)/(rowSize-1);
		sphere.material.color = SrgbToLinear(Color(0.7f));
		sphere.material.specular = 0.0f;

        scene->AddPrimitive(sphere);
    }

	y += r*2.2f;

	Material gold;
	gold.color = Color(1.0f, 0.71f, 0.29f);
	gold.roughness = 0.5f;
	gold.metallic = 1.0f;

    Material silver;
	silver.color = Color(0.95f, 0.93f, 0.88f);
	silver.roughness = 0.5f;
	silver.metallic = 1.0f;

    Material copper;
	copper.color = Color(0.95f, 0.64f, 0.54f, 120.0f);
	copper.roughness = 0.5f;
	copper.metallic = 1.0f;

    Material iron;
	iron.color = Color(0.56f, 0.57f, 0.58f, 120.0f);
	iron.roughness = 0.5f;
	iron.metallic = 1.0f;

    Material aluminum;
	aluminum.color = Color(0.91f, 0.92f, 0.92f, 120.0f);
	aluminum.roughness = 0.5f;
	aluminum.metallic = 1.0f;

    Material plaster;
	plaster.color = Color(0.94f, 0.94f, 0.94f);
	plaster.roughness = 0.75;
	plaster.specular = 0.1;

	Material mats[6] = { gold, silver, copper, iron, aluminum, plaster };

	for (int i=0; i < 6; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = r;
        sphere.transform = TranslationMatrix(Vec3(-x + i*dx, y, 0.0f));
        sphere.lastTransform = TranslationMatrix(Vec3(-x+ i*dx, y, 0.0f));
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
    light.transform = TranslationMatrix(Vec3(0.0f, 6.0f, 6.0f));
	light.lastTransform = light.transform;
    light.material.color = Vec4(0.0f);
    light.material.emission = Vec4(15.0f);
    light.light = true;

    
    scene->AddPrimitive(plane);
	scene->AddPrimitive(back);
    scene->AddPrimitive(light);

	scene->sky = Color(0.1f, 0.3f, 0.6f)*2.0f;

    // set up camera
    g_camPos = Vec3(0.0f, 2.0f, 10.0f);
    g_camTarget = Vec3(0.0f, 2.0f, 0.0f);

    return scene;
}

