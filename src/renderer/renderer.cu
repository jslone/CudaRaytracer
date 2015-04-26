#include "renderer.h"
#include <cstring>
#include <iostream>
#include <cuda_gl_interop.h>
#include <curand.h>

namespace acr
{
	struct DevParams
	{
		Scene *scene;
		Color4 *screen;
		char sceneData[sizeof(Scene)];
		uint32_t width,height,samples;
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
		glCtx = SDL_GL_CreateContext(window);
		
		/* Set the clear color. */
		glClearColor( 0, 0, 0, 0 );

		/* Setup our viewport. */
		glViewport( 0, 0, dim.x, dim.y );
		
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
			switch(cudaErr)
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
			switch(cudaErr)
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

		glBufferData(GL_PIXEL_UNPACK_BUFFER, dim.x * dim.y * dim.z * sizeof(Color4), NULL, GL_DYNAMIC_COPY);

		cudaGLRegisterBufferObject(drawBuffer);

		// setup texture
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1,&textureId);
		
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, dim.x, dim.y, 0, GL_RGBA32F, GL_FLOAT, nullptr);

		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

		cudaMalloc((void**)&cuRandState, sizeof(state) * dim.x * dim.y);
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
		params.samples = dim.z;

		cudaMemcpyToSymbol(&devParams, &params, sizeof(DevParams));
	}

	__global__
	void scatterTrace(currentState *randState, unsigned long seed)
	{
		
		int x = blockIdx.x * gridDim.x + threadIdx.x;
		int y = blockIdx.y * gridDim.y + threadIdx.y;
		int sample = blockIdx.z * gridDim.z + threadIdx.z;
		int index = sample + samples * x + (samples * width) * y;
		
		devParams.screen[x + devParams.width * y] = Color4(0,0,0,0);
		
		curand_init(seed,index,0,&randState[index]);

		Scene *scene = devParams.scene;

		float dx = 1.0f / devParams.width;
		float dy = 1.0f / devParams.height;
		
		float i = 2.0f*(float(x)+curand_uniform(randState[index]))*dx - 1.0f;
    float j = 2.0f*(float(y)+curand_uniform(randState[index]))*dy - 1.0f;

		
		Ray r;
		r.o = math::vec3(scene->camera.globalTransform * math::vector4(0,0,0,1));
		r.d = Ray::get_pixel_dir(scene->camera, i, j);
		 
		HitInfo info;
		Color4 contribution;
		if(scene->intersect(r,info))
		{
			contribution = scene->materials[info.materialIndex].diffuse;
		}
		else
		{
			contribution = Color4(0,0,0,1);
		}
		
		for(int i = 0; i < 4; i++)
		{
			atomicAdd(&devParams.screen[x + devParams.width * y][i], contribution[i] / devParams.samples);
		}
	}

	void Renderer::render()
	{
		// bind draw buffer to device ptr
		cudaGLMapBufferObject((void**)&devParams.screen, drawBuffer);
		
		// call kernel to render pixels then draw to screen
		dim3 block(4,4,16);
		dim3 grid(dim.x / block.x, dim.y / block.y, dim.z / block.z);
		
		scatterTrace<<<grid,block>>>();
		
		// unbind draw buffer so openGL can use
		cudaGLUnmapBufferObject(drawBuffer);

		// create texture from draw buffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, drawBuffer);

		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim.x, dim.y, GL_RGBA32F, GL_FLOAT, nullptr);

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
		SDL_GL_SwapWindow(window);
	}

}
