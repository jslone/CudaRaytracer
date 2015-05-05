#ifndef _RENDERER_H_
#define _RENDERER_H_


#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <curand.h>
#include <curand_kernel.h>

#include "math/math.h"
#include "scene/scene.h"

typedef curandState curandState_t;

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
		int winId;
		
		const char *title;
		math::u32vec3 dim;

		GLuint drawBuffer;
		GLuint textureId;

		curandState *cuRandStates;

	};

} // namespace acr

#endif //_RENDERER_H_
