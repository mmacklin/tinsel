


#ifdef _WIN32

#include <windows.h>
#include <commdlg.h>
#include <mmsystem.h>

double GetSeconds()
{
	static LARGE_INTEGER lastTime;
	static LARGE_INTEGER freq;
	static bool first = true;
	
	if (first)
	{	
		QueryPerformanceCounter(&lastTime);
		QueryPerformanceFrequency(&freq);

		first = false;
	}
	
	static double time = 0.0;
	
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	
	__int64 delta = t.QuadPart-lastTime.QuadPart;
	double deltaSeconds = double(delta) / double(freq.QuadPart);
	
	time += deltaSeconds;

	lastTime = t;

	return time;

}

#else


// linux, mac platforms
#include <sys/time.h>

double GetSeconds()
{
	// Figure out time elapsed since last call to idle function
	static struct timeval last_idle_time;
	static double time = 0.0;	

	struct timeval time_now;
	gettimeofday(&time_now, NULL);

	if (last_idle_time.tv_usec == 0)
		last_idle_time = time_now;

	float dt = (float)(time_now.tv_sec - last_idle_time.tv_sec) + 1.0e-6*(time_now.tv_usec - last_idle_time.tv_usec);

	time += dt;
	last_idle_time = time_now;

	return time;
}

#endif
