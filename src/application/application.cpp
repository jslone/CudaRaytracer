#include <iostream>
#include "application.h"
#include <GL/glew.h>
#include <GL/glut.h>

namespace acr
{
	Application *app;

	void keyboardCB( unsigned char key, int x, int y )
	{
  		switch ( key )
		{
			case 27: // Escape key
				exit (0);
				break;
		}
		if (x || y)
		{
			app->renderer.moveCamera(math::vec3(0, 0, 0), math::normalize(math::vec3(x, y, 0) / 1000.0f));
		}
		glutPostRedisplay();
	}

	Application::Application(const Args args)
		: renderer(args.renderer)
		, scene(args.scene)
		, frameRate(args.frameRate)
	{
		glutKeyboardFunc(keyboardCB);
	}

	Application::~Application()
	{}

	void Application::start()
	{
		std::cout << "Starting application..." << std::endl;
		running = true;
		renderer.loadScene(scene);
		glutMainLoop();
	}

} // namespace acr


int main(int argc, char **argv)
{
	glutInit(&argc,argv);

	// Setup
	acr::Application::Args args;
	args.renderer.title = "CudaRenderer";
	args.renderer.pos.x = 0;
	args.renderer.pos.y = 0;
	args.renderer.dim.x = 800;
	args.renderer.dim.y = 600;
	args.renderer.dim.z = 16;
	args.frameRate = 60;
	args.scene.filePath = argv[1]; //!!!! Should check for argc bound
	// Start the app
	acr::Application app(args);
	app.start();

	return 0;
}
