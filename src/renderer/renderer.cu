#include <cstring>
#include <iostream>
#include <gl/gl.h>
#include <gl/glu.h>
#include "renderer.h"

namespace acr
{
	struct DevParams
	{
		Scene *scene;
		Color4 *screen;
		char sceneData[sizeof(Scene)];
		uint32_t width,height;
	};

	__constant__
	DevParams devParams;

	Renderer::Renderer(const Renderer::Args &args)
		: title(args.title)
		, dim(args.dim)
	{
		// sdl initialization
		if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
		{
			std::cerr << "SDL_InitSubSystem Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}

		window = SDL_CreateWindow(title, args.pos.x, args.pos.y,
								  dim.x, dim.y, SDL_WINDOW_OPENGL);
		if (window == nullptr)
		{
			std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}

		renderer = SDL_CreateRenderer(window, -1, 0);
		if (renderer == nullptr)
		{
			std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
			exit(EXIT_FAILURE);
		}

		// open gl initialization
		
		/* Set the clear color. */
		glClearColor( 0, 0, 0, 0 );

		/* Setup our viewport. */
		glViewport( 0, 0, width, height );
		
		/* Setup the projection and world matrix */
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity( );

		glOrtho(0,1.0f,0,1.0f,-1.0f,1.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// cuda interop initialization
		uint32_t numDevices;
		uint32_t maxNumDevices = 1;
		int devices[maxNumDevices];
		cudaError_t cudaErr = cudaGLGetDevices(&numDevices, devices, maxNumDevices, cudaGLDeviceListAll);
		if(cudaErr != cudaSuccess)
		{
			std::cout << "cudaGLGetDevices: ";
			switch(err)
			{
				case cudaErrorNoDevice:
					std::cout << "No device found." << std::endl;
					break;
				default:
					std::cout << "Error unknown." << std::endl;
			}
			exit(EXIT_FAILURE);
		}

		cudaErr = cudaGLSetGLDevice(devices[0]);
		if(cudaErr != cudaSuccess)
		{
			std::cout << "cudaGLSetGLDevice: ";
			switch(err)
			{
				case cudaErrorInvalidDevice:
					std::cout << "Invalid device." << std::endl;
					break;
				default:
					std::cout << "Device already set." << std::endl;
			}
			exit(EXIT_FAILURE);
		}

		// setup draw buffer
		glGenBuffers(1,&drawBuffer);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, drawBuffer);

		glBufferData(GL_PIXEL_UNPACK_BUFFER, dim.x * dim.y * sizeof(Color4), NULL, GL_DYNAMIC_COPY);

		cudaGLRegisterBufferObject(drawBuffer);

		// setup texture
		glEnable(GL_TEXTURE_2D);
		glGenTexture(&textureId);
		
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, dim.x, dim.y, 0, GL_RGBA32F, GL_FLOAT, nullptr);

		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	}

	Renderer::~Renderer()
	{
		if (renderer)
		{
			SDL_DestroyRenderer(renderer);
		}
		if (window)
		{
			SDL_DestroyWindow(window);
		}
	}

	void Renderer::loadScene(const Scene &scene)
	{
		DevParams params;
		params.scene = (Scene*)(&devParams.sceneData[0]);
		std::memcpy(params.sceneData, &scene, sizeof(Scene));
		params.width = dim.x;
		params.height = dim.y;

		cudaMemcpyToSymbol(&devParams, &params, sizeof(DevParams));
	}

	__global__
	void scatterTrace()
	{
		int x = blockIdx.x * gridDim.x + threadIdx.x;
		int y = blockIdx.y * gridDim.y + threadIdx.y;
		int sample = blockIdx.z * gridDim.z + threadIdx.z;

		devParams.screen[x + devParams.width * y] = math::Color4(0,0,0,1);
	}

	void Renderer::render()
	{
		// bind draw buffer to device ptr
		cudaGLMapBufferObject((void**)&devParams.screen, drawBuffer);
		
		// call kernel to render pixels then draw to screen
		dim3 block(4,4,16);
		dim3 grid(dim.x / block.x, dim.y / block.y, 1);
		
		scatterTrace<<<grid,block>>>();
		
		// unbind draw buffer so openGL can use
		cudaGLUnmapBufferObject(drawBuffer);

		// create texture from draw buffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, drawBuffer);

		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexSubImage(GL_TEXTURE_2D, 0, 0, 0, Width, Height, GL_RGBA32F, GL_FLOAT, nullptr);

		// draw fullscreen quad
		glBegin(GL_QUADS);
			glTexCoord2f( 0, 1.0f);
			glVertex3f(0,0,0);
			glTexCoord2f(0,0);
			glVertex3f(0,1.0f,0);
			glTexCoord2f(1.0f,0);
			glVertex3f(1.0f,1.0f,0);
			glTexCoord2f(1.0f,1.0f);
			glVertex3f(1.0f,0,0);
		glEnd();
		
		// swap buffers
		SDL_GL_SwapBuffers();
	}

}
