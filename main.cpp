#include "scene.h"
#include "camera.h"
#include "bvh.h"
#include "maths.h"
#include "render.h"

#if _WIN32

#include "freeglut/include/GL/glut.h"

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
Vec3 g_camAngle;
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

RenderMode g_mode = eNormals;//PathTrace;

double GetSeconds();

Mat44 g_cameraTransform;
Camera g_camera;

void Render()
{
	// generate camera transform from translation, yaw, pitch
	g_cameraTransform = TranslationMatrix(g_camPos)*RotationMatrix(g_camAngle.x, Vec3(0.0f, 1.0f, 0.0f))*RotationMatrix(g_camAngle.y, Vec3(1.0f, 0.0f, 0.0f));

    // update camera
    g_camera = Camera(
		g_cameraTransform,
        DegToRad(35.0f),
        1.0f,
        10000.0f,
        g_width,
        g_height);

    double startTime = GetSeconds();

	const int numSamples = 1;

    // take one more sample per-pixel each frame for progressive rendering
    g_renderer->Render(&g_camera, g_pixels, g_width, g_height, numSamples, g_mode);

    double endTime = GetSeconds();

    printf("%d (%.4fms)\n", g_iterations, (endTime-startTime)*1000.0f);
    fflush(stdout);
                

    Color* presentMem = g_pixels;

    g_iterations += numSamples;

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

	printf("%d %d\n", g_width, g_height);

	g_renderer->Init(g_width, g_height);
}

#include "tests/testMaterials.h"
#include "tests/testMesh.h"
#include "tests/testMotionBlur.h"

void Init()
{
    g_scene = TestMesh();
	//g_scene = TestMaterials();
    //g_scene = TestMIS();

    // create renderer
    g_renderer = CreateGpuRenderer(g_scene);
        
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
    bool resetFrame = false;

 	switch (key)
	{
    case 'w':
        g_camPos -= Vec3(g_cameraTransform.GetCol(2))*g_flySpeed; resetFrame = true;
		break;
    case 's':
        g_camPos += Vec3(g_cameraTransform.GetCol(2))*g_flySpeed; resetFrame = true; 
        break;
    case 'a':
        g_camPos -= Vec3(g_cameraTransform.GetCol(0))*g_flySpeed; resetFrame = true;
        break;
    case 'd':
        g_camPos += Vec3(g_cameraTransform.GetCol(0))*g_flySpeed; resetFrame = true;
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
    int dx = x-lastx;
    int dy = y-lasty;

    const float sensitivity = 0.01f;

    g_camAngle.x -= dx*sensitivity;
    g_camAngle.y -= dy*sensitivity;

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
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );

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



 