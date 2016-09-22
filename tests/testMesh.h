#pragma once

Scene* TestMesh()
{
	/*
    Scene* s = LoadTin("data/example.tin");

    // set up camera
    g_camPos = Vec3(0.0f, 2.0f, 10.0f);
    g_camTarget = Vec3(0.0f, 2.0f, 0.0f);

    return s;
	*/

    Scene* scene = new Scene();


    //Mesh* buddha = ImportMeshFromObj("data/Ajax_Jotero_com.obj");
	
	Mesh* buddha = ImportMeshFromObj("data/octopus.obj");
	//Mesh* buddha = ImportMeshFromObj("data/manifold.obj");
	//Mesh* buddha = ImportMeshFromPly("data/lion.ply");
    
	//Mesh* buddha = ImportMeshFromObj("data/elefant_from_jotero_com.obj"); buddha->Transform(RotationMatrix(DegToRad(-150.0f), Vec3(1.0f, 0.0f, 0.0f)));
	//Mesh* buddha = ImportMeshFromObj("data/Aphrodite_from_jotero_com.obj"); buddha->Transform(RotationMatrix(DegToRad(-90.0f), Vec3(1.0f, 0.0f, 0.0f)));
	
    buddha->Normalize(4.0f);
	buddha->CalculateNormals();
	buddha->RebuildBVH();

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
	Mesh* tet = CreateTetrahedron();
    tet->Transform(TranslationMatrix(Vec3(0.0f, 1.0f, 0.0f)));
	tet->RebuildBVH();


    printf("nodes: %d\n", tet->bvh.nodes.size());

    for (int i=0; i < tet->bvh.nodes.size(); ++i)
    {
        const BVH::Node& node = tet->bvh.nodes[i];

        printf("node %d : (%f, %f, %f) - (%f, %f, %f), [%d, %d] leaf: %d\n", 
            i, 
            node.bounds.lower.x, node.bounds.lower.y, node.bounds.lower.z, 
            node.bounds.upper.x, node.bounds.upper.y, node.bounds.upper.z,
            node.leftIndex, node.rightIndex, node.leaf);
    }
*/

    Material plaster;
	plaster.color = Color(0.94f, 0.94f, 0.94f);
	plaster.roughness = 0.75;
	plaster.specular = 0.1;

    Material gold;
    gold.color = Color(1.0f, 0.71f, 0.29f);
    gold.roughness = 0.1f;
	gold.specular = 0.75f;
    gold.metallic = 1.0f;

	Primitive mesh;
	mesh.type = eMesh;
	mesh.mesh = GeometryFromMesh(buddha);
    mesh.material = gold;
	mesh.transform = Transform(Vec3(0.0f, 1.0f, 0.0f), Quat(Vec3(0.0f, 1.0f, 0.0f), 0.0f), 1.0f);//*RotationMatrix(DegToRad(90.0f), Vec3(0.0f, 0.0f, 1.0f))*ScaleMatrix(Vec3(3.0f));
	mesh.lastTransform = Transform(Vec3(0.0f, 1.0f, 0.0f), Quat(Vec3(0.0f, 1.0f, 0.0f), DegToRad(0.0f)), 2.0f);
	

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
    light.transform = Transform(Vec3(0.0f, 6.0f, 0.0f));
	light.lastTransform = light.transform;
    light.material.color = Vec4(0.0f);
    light.material.emission = Vec4(10.0f);
    light.light = true;

    
	scene->AddPrimitive(mesh);
    scene->AddPrimitive(plane);
	scene->AddPrimitive(light);

	scene->sky = Color(0.1f, 0.3f, 0.6f)*2.0f;

    // set up camera
    g_camPos = Vec3(0.0f, 2.0f, 10.0f);
    g_camTarget = Vec3(0.0f, 2.0f, 0.0f);

    return scene;
}

