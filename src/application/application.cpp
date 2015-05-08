#include <iostream>
#include "application.h"
#include <GL/glew.h>
#include <GL/glut.h>

namespace acr
{
	Application *app;

	const float movSpeed = 0.5f;
	void keyboardCB( unsigned char key, int x, int y )
	{
  		switch ( key )
		{
			case 'w':
				app->renderer.moveCamera(math::vec2(0, 1)*movSpeed, math::vec2(0,0));
				break;
			case 'a':
				app->renderer.moveCamera(math::vec2(-1, 0)*movSpeed, math::vec2(0, 0));
				break;
			case 's':
				app->renderer.moveCamera(math::vec2(0, -1)*movSpeed, math::vec2(0, 0));
				break;
			case 'd':
				app->renderer.moveCamera(math::vec2(1, 0)*movSpeed, math::vec2(0, 0));
				break;
			case 27: // Escape key
				exit (0);
				break;
		}
		glutPostRedisplay();
	}

	bool shouldRot = false;
	math::vec2 mousePos(-1, -1);

	void mousePressCB(int button, int state, int x, int y)
	{
		mousePos.x = x;
		mousePos.y = y;
		shouldRot = state == GLUT_DOWN;
	}
	
	const float rotSpeed = 1.0f / 1000.0f;

	void mouseMoveCB(int x, int y)
	{
		math::vec2 nextPos(x, y);

		if (shouldRot)
		{
			math::vec2 delta = (nextPos - mousePos) * rotSpeed;
			delta.x *= -1;
			app->renderer.moveCamera(math::vec2(0, 0), delta);
		}
		mousePos = nextPos;
	}

	Application::Application(const Args args)
		: renderer(args.renderer)
		, scene(args.scene)
		, frameRate(args.frameRate)
	{
		glutKeyboardFunc(keyboardCB);
		glutMotionFunc(mouseMoveCB);
		glutMouseFunc(mousePressCB);
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
	args.renderer.dim.z = 1;
	args.frameRate = 60;
	args.scene.filePath = argv[1]; //!!!! Should check for argc bound
	// Start the app
	acr::Application app(args);
	app.start();

	return 0;
}
