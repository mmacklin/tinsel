#include "scene.h"
#include "camera.h"
#include "bvh.h"
#include "maths.h"
#include "render.h"

#if _WIN32
#include <glfw/glfw3.h>

#elif __APPLE__
#define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED 
#include <opengl/gl3.h>
#include <glut/glut.h>

#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLRenderers.h>
#include <OpenGL/CGLTypes.h>
#include <OpenGL/OpenGL.h>

#endif

#include <iostream>

using namespace std;

Vec4* g_pixels;
Color* g_filtered;
int g_iterations;

Vec3 g_camPos;
Vec3 g_camTarget;

Scene* g_scene;

// render dimensions
int g_width = 640;
int g_height = 360;

// window dimensions
int g_windowWidth = g_width;
int g_windowHeight = g_height;

float g_exposure = 1.f;
float g_zoom = 1.0f;
float g_sunTheta = 0.41389135;
float g_sunPhi = 0.95993042;
float g_sunTurbidity = 2.0f;

float g_flySpeed = 0.5f;

Renderer* g_renderer;

RenderMode g_mode = ePathTrace;

double GetSeconds();

void Render()
{
    // update camera
    Camera camera(
        AffineInverse(LookAtMatrix(g_camPos, Vec3(0.0f))),
        45.0f,
        1.0f,
        10000.0f,
        g_windowWidth,
        g_windowHeight);

    double startTime = GetSeconds();

    // take one more sample per-pixel each frame for progressive rendering
    g_renderer->Render(&camera, g_pixels, g_windowWidth, g_windowHeight, 1, g_mode);

    double endTime = GetSeconds();

    printf("%d (%.2fs)\n", g_iterations, (endTime-startTime));
    fflush(stdout);

    Color* presentMem = g_pixels;

    ++g_iterations;

    /*
    for (int i=0; i < g_windowWidth; ++i)
    {
        for (int j=0; j < g_windowHeight; ++j)
        {
            g_pixels[j*g_windowWidth + i] = Vec4(0.5f, 1.0f, 0.25f);
        }
    }
    */

    if (g_mode == ePathTrace)
    {
        float s = g_exposure / g_iterations;

        for (int i=0; i < g_width*g_height; ++i)
        {
            g_filtered[i] = LinearToSrgb(g_pixels[i] * s);
        }

        presentMem = g_filtered;
    }

/*
	static int s_counter=0;
    if (s_counter % 10)
    {
        cout << "Trace took: " << (endTime-startTime)*1000.0f << "ms" << " rays/s: " << g_width*g_height/(endTime-startTime) << endl;
    }
    ++s_counter;
*/

    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    
    glPixelZoom(float(g_windowWidth)/g_width, float(g_windowHeight)/g_height);
    glDrawPixels(g_width,g_height,GL_RGBA,GL_FLOAT, presentMem);
}

void InitFrameBuffer()
{
    delete[] g_pixels;
    delete[] g_filtered;

    g_width = g_windowWidth*g_zoom;
    g_height = g_windowHeight*g_zoom;

    g_pixels = new Color[g_width*g_height];
    g_filtered = new Color[g_width*g_height];

    g_iterations = 0;
}

Scene* TestScene1()
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
    for (int i=0; i < 10; ++i)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = 0.5f;
        sphere.transform = TranslationMatrix(Vec3(-5.0f + i*1.2f, 0.5f, 0.0f));
        sphere.lastTransform = TranslationMatrix(Vec3(-4.0f + i*1.2f, 0.5f, 0.0f));
        //sphere.material.reflectance = Color(.82f, .67f, .16f);
        sphere.material.metallic = float(i)/9;

        scene->AddPrimitive(sphere);
    }

    Primitive plane;
    plane.type = ePlane;
    plane.plane.plane[0] = 0.0f;
    plane.plane.plane[1] = 1.0f;
    plane.plane.plane[2] = 0.0f;
    plane.plane.plane[3] = 0.0f;
    plane.material.color = Color(0.25);

    Primitive light;
    light.type = eSphere;
    light.sphere.radius = 1.0f;
    light.transform = TranslationMatrix(Vec3(0.0f, 4.0f, 4.0f));
    //light.material.reflectance = Vec4(0.0f);
    light.material.emission = Vec4(10.0f);
    light.light = true;

    
    scene->AddPrimitive(plane);
    scene->AddPrimitive(light);
 //   scene->AddLight(light);

    // set up camera
    g_camPos = Vec3(0.0f, 0.5f, 10.0f);
    g_camTarget = Vec3(0.0f);

    return scene;
}


void Init()
{
    g_scene = TestScene1();

    // create renderer
    g_renderer = CreateCpuRenderer(g_scene);
        
    InitFrameBuffer();
}

void GLUTUpdate()
{
    Render();

	// flip
	glutSwapBuffers();
}

void GLUTReshape(int width, int height)
{
    g_windowWidth = width;
    g_windowHeight = height;

    InitFrameBuffer();
}

void GLUTArrowKeys(int key, int x, int y)
{
}

void GLUTArrowKeysUp(int key, int x, int y)
{
}

void GLUTKeyboardDown(unsigned char key, int x, int y)
{
    Mat44 v = AffineInverse(LookAtMatrix(g_camPos, g_camTarget));

    bool resetFrame = false;

 	switch (key)
	{
    case 'w':
        g_camPos -= Vec3(v.GetCol(2))*g_flySpeed; resetFrame = true;
		break;
    case 's':
        g_camPos += Vec3(v.GetCol(2))*g_flySpeed; resetFrame = true; 
        break;
    case 'a':
        g_camPos -= Vec3(v.GetCol(0))*g_flySpeed; resetFrame = true;
        break;
    case 'd':
        g_camPos += Vec3(v.GetCol(0))*g_flySpeed; resetFrame = true;
        break;
	case '1':
		g_mode = eNormals;
		break;
	case '2':
		g_mode = eComplexity;
		break;
    case '3':
        g_mode = ePathTrace; resetFrame = true;
        break;
    case '+':
        g_zoom = min(1.0f, g_zoom+0.30f); resetFrame = true;
        break;
    case '-':
        g_zoom = max(0.1f, g_zoom-0.30f); resetFrame = true;
        break;
	case '[':
		g_exposure -= 0.01f;
		break;
	case ']':
		g_exposure += 0.01f;
		break;
	case '8': 
		g_sunTheta += DegToRad(1.0f); resetFrame = true;
		break;
	case '5':
		g_sunTheta -= DegToRad(1.0f); resetFrame = true;
		break;
	case '9': 
		g_sunPhi += DegToRad(1.0f); resetFrame = true;
		break;
	case '6':
		g_sunPhi -= DegToRad(1.0f); resetFrame = true;
		break;
	case '7':
		g_sunTurbidity += 0.01f; resetFrame = true;
		break;
	case '4':
		g_sunTurbidity -= 0.01f; resetFrame = true;
		break;
    case '0':
        g_zoom = 1.0f; resetFrame = true;
        break;
    case 'q':
	case 27:
		exit(0);
		break;
	};

    // reset image if there are any camera changes
    if (resetFrame == true)
    {
        InitFrameBuffer();
    }
}

void GLUTKeyboardUp(unsigned char key, int x, int y)
{
// 	switch (key)
// 	{
// 	case 27:
// 		exit(0);
// 		break;
// 	};

}

static int lastx;
static int lasty;

void GLUTMouseFunc(int b, int state, int x, int y)
{
	switch (state)
	{
	case GLUT_UP:
		{
			lastx = x;
			lasty = y;			
		}
	case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;
		}
	}
}

void GLUTMotionFunc(int x, int y)
{
    /*
    int dx = x-lastx;
    int dy = y-lasty;

    const float sensitivity = 0.1f;

    g_camDir.yaw -= dx*sensitivity;
    g_camDir.roll += dy*sensitivity;
    */

	lastx = x;
	lasty = y;

    if (g_mode == ePathTrace)
    {
        InitFrameBuffer();
    }
}


/*
void Application::JoystickFunc(int x, int y, int z, unsigned long buttons)
{
g_app->JoystickFunc(x, y, z, buttons);
}
*/

int main(int argc, char* argv[])
{	
	// init gl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

	glutInitWindowSize(g_width, g_height);
	glutCreateWindow("Tinsel");
	glutPositionWindow(200, 200);

    Init();

    glutMouseFunc(GLUTMouseFunc);
	glutReshapeFunc(GLUTReshape);
	glutDisplayFunc(GLUTUpdate);
	glutKeyboardFunc(GLUTKeyboardDown);
	glutKeyboardUpFunc(GLUTKeyboardUp);
	glutIdleFunc(GLUTUpdate);	
	glutSpecialFunc(GLUTArrowKeys);
	glutSpecialUpFunc(GLUTArrowKeysUp);
	glutMotionFunc(GLUTMotionFunc);

	glutMainLoop();
}



 