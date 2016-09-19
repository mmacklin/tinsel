#pragma once

#pragma once

Scene* TestMaterials()
{
    Scene* scene = new Scene();


    Mesh* mesh = ImportMeshFromObj("../../data/happy.obj");
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

	
    Material plaster;
	plaster.color = Color(0.94f, 0.94f, 0.94f);
	plaster.roughness = 0.75;
	plaster.specular = 0.1;

	Primitive mesh;
	mesh.mesh = mesh;
	mesh.bvh.Build(Bounds)
	
	

    Primitive plane;
    plane.type = ePlane;
    plane.plane.plane[0] = 0.0f;
    plane.plane.plane[1] = 1.0f;
    plane.plane.plane[2] = 0.0f;
    plane.plane.plane[3] = 0.0f;
    plane.material.color = Color(0.1);

    Primitive light;
    light.type = eSphere;
    light.sphere.radius = 1.0f;
    light.transform = TranslationMatrix(Vec3(0.0f, 6.0f, 6.0f));
	light.lastTransform = light.transform;
    light.material.color = Vec4(0.0f);
    light.material.emission = Vec4(15.0f);
    light.light = true;

    
    scene->AddPrimitive(plane);
	scene->AddPrimitive(light);

	scene->sky = Color(0.1f, 0.3f, 0.6f)*2.0f;

    // set up camera
    g_camPos = Vec3(0.0f, 2.0f, 10.0f);
    g_camTarget = Vec3(0.0f, 2.0f, 0.0f);

    return scene;
}

