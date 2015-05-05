#include "renderer.h"
#include <cstring>
#include <iostream>
#include <cuda_gl_interop.h>

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
	
	Renderer *renderer;
	void globalRender()
	{
		renderer->render();
	}

	Renderer::Renderer(const Renderer::Args &args)
		: title(args.title)
		, dim(args.dim)
	{
		renderer = this;

		/* Create window */
		glutInitDisplayMode(GLUT_RGBA);
		glutInitWindowSize(dim.x, dim.y);

		winId = glutCreateWindow(title);

		GLenum err = glewInit();
		if (err != GLEW_OK)
		{
			std::cerr << "glewInit: " << glewGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		glutDisplayFunc(globalRender);

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
		const uint32_t maxNumDevices = 1;
		int devices[maxNumDevices];
		cudaError_t cudaErr = cudaGLGetDevices(&numDevices, devices, maxNumDevices, cudaGLDeviceListAll);
		if(cudaErr != cudaSuccess)
		{
			std::cout << "cudaGLGetDevices[" << cudaErr << "]: ";
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

		cudaMalloc((void**)&cuRandStates, sizeof(curandState) * dim.x * dim.y * dim.z);
	}

	Renderer::~Renderer()
	{
		glutDestroyWindow(winId);
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

	__device__ __host__
	math::vec3 get_pixel_dir(const Camera &camera, int ni, int nj)
	{

		math::vec3 dir;
		math::vec3 up;
		float AR;

		math::vec3 cR;
		math::vec3 cU;
		float dist;
		math::vec3 pos;
    
		dir = camera.forward;
		up = camera.up;
		AR = camera.aspectRatio;
		cR = math::cross(dir, up);
		cU = math::cross(cR, dir);
		pos = camera.position;
		dist = math::tan(camera.horizontalFOV/2.0);
		
		return math::normalize(dir + dist*(float(nj)*cU + AR*float(ni)*cR));
	}

	__global__
	void scatterTrace(curandState *randState, unsigned long seed)
	{
		const int width = devParams.width;
		const int height = devParams.height;

		int x = blockIdx.x * gridDim.x + threadIdx.x;
		int y = blockIdx.y * gridDim.y + threadIdx.y;
		int index = x + width * y;
		
		devParams.screen[x + devParams.width * y] = Color4(0,0,0,0);
		
		curand_init(seed,index,0,&randState[index]);

		Scene *scene = devParams.scene;

		float dx = 1.0f / devParams.width;
		float dy = 1.0f / devParams.height;
		
		float i = 2.0f*(float(x)+curand_uniform(&randState[index]))*dx - 1.0f;
		float j = 2.0f*(float(y)+curand_uniform(&randState[index]))*dy - 1.0f;

		
		Ray r;
		r.o = scene->camera.position;
		r.d = get_pixel_dir(scene->camera, i, j);
		 
		HitInfo info;
		Color4 contribution;
		if(scene->intersect(r,info))
		{
			contribution = Color4(scene->materials[info.materialIndex].diffuse,1);
		}
		else
		{
			contribution = Color4(0,0,0,1);
		}
		
		devParams.screen[index] = contribution;
	}

	void Renderer::render()
	{
		// bind draw buffer to device ptr
		cudaGLMapBufferObject((void**)&devParams.screen, drawBuffer);
		
		// call kernel to render pixels then draw to screen
		dim3 block(16,16,1);
		dim3 grid(dim.x / block.x, dim.y / block.y, 1);
		
		scatterTrace<<<grid,block>>>(cuRandStates,glutGet(GLUT_ELAPSED_TIME));
		
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
		glutSwapBuffers();
		glutPostRedisplay();
	}

}
