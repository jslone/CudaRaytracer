#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include "math/math.h"
#include "renderer/renderer.h"
#include "scene/scene.h"

namespace acr
{

	/* Application class.
	 *
	 * Application implements a class to manage each submodule of the
	 * entire application. Application is in charge of managing the
	 * entry point and the per frame execution until the application
	 * terminates.
	 */
	class Application
	{
	public:

		struct Args
		{
			Renderer::Args renderer;
			uint8_t frameRate;
			Scene::Args scene;
		};

		Application(const Args args);
		~Application();

		void start();
		void quit();

	private:
		Renderer renderer;
		Scene scene;
		bool running;
		int32_t lastTick;
		uint32_t frameRate;

		void run();
		void handle_events();
	};

} // namespace acr

#endif //_APPLICATION_H_
