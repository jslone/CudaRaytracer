#ifndef _RENDERER_H_
#define _RENDERER_H_


#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <SDL.h>

#include <curand.h>

#include "math/math.h"
#include "scene/scene.h"

namespace acr
{

	class Renderer
	{
	public:

		struct Args
		{
			const char *title;
			math::u32vec2 pos;
			math::u32vec3 dim;
		};

		Renderer(const Args &args);
		~Renderer();

		void loadScene(const Scene &scene);

		void render();
	private:
		SDL_Window *window;
		SDL_Renderer *renderer;
		SDL_GLContext glCtx;

		const char *title;
		math::u32vec3 dim;

		GLuint drawBuffer;
		GLuint textureId;

		state *cuRandState;

	};

} // namespace acr

#endif //_RENDERER_H_
