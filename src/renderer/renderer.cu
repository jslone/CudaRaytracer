#include "renderer.h"
#include <cstring>
#include <iostream>
#include <cuda_gl_interop.h>

namespace acr
{
	struct DevParams
	{
		char sceneData[sizeof(Scene)];
		uint32_t width,height,samples;
	};

	__constant__
	DevParams devParams[1];
	
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
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(dim.x, dim.y);

		dim.x *= dim.z;
		dim.y *= dim.z;
		dim.z = 1;

		winId = glutCreateWindow(title);

		GLenum err = glewInit();
		if (err != GLEW_OK)
		{
			std::cerr << "glewInit: " << glewGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		glutDisplayFunc(globalRender);

		/* Set the clear color. */
		glClearColor( 0, 0, 0, 1 );
		glClear(GL_COLOR_BUFFER_BIT);

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

		glBufferData(GL_PIXEL_UNPACK_BUFFER, dim.x * dim.y * sizeof(Color4), NULL, GL_DYNAMIC_COPY);

		cudaGLRegisterBufferObject(drawBuffer);

		// setup texture
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1,&textureId);
		
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dim.x, dim.y, 0, GL_RGBA, GL_FLOAT, nullptr);

		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	}

	Renderer::~Renderer()
	{
		glutDestroyWindow(winId);
	}

	void Renderer::moveCamera(const math::vec2 &pos, const math::vec2 &dir)
	{
		DevParams param;
		cudaMemcpyFromSymbol(&param, devParams, sizeof(DevParams));

		Scene *scene = (Scene*)&param;

		scene->camera.forward = math::rotate(scene->camera.forward, dir.x, scene->camera.up);

		math::vec3 right = math::cross(scene->camera.forward, scene->camera.up);

		scene->camera.forward = math::rotate(scene->camera.forward, dir.y, right);
		scene->camera.up = math::rotate(scene->camera.up, dir.y, right);

		math::vec3 delta = scene->camera.forward * pos.y + right * pos.x;

		scene->camera.position += delta;

		cudaMemcpyToSymbol(devParams, &param, sizeof(DevParams));
	}

	void Renderer::loadScene(const Scene &scene)
	{
		DevParams params;
		std::memcpy(params.sceneData, &scene, sizeof(Scene));
		params.width = dim.x;
		params.height = dim.y;
		params.samples = dim.z;

		Scene *myScene = (Scene*)&params;
		myScene->camera.aspectRatio = float(dim.x) / float(dim.y);

		cudaMemcpyToSymbol(devParams, &params, sizeof(DevParams));
	}

	__device__
	math::vec3 get_pixel_dir(const Camera &camera, float ni, float nj)
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
		dist = math::fastertanfull(camera.horizontalFOV/2.0f);
		
		return math::normalize(dir + dist*(float(nj)*cU + AR*float(ni)*cR));
	}

	__device__
	math::vec3 get_pixel_pos(const Camera &camera, float ni, float nj)
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
		dist = math::fastertanfull(camera.horizontalFOV / 2.0f);

		return camera.position + dist*(float(nj)*cU + AR*float(ni)*cR);
	}

	__global__
	void scatterTrace(Color4 *screen, unsigned long seed)
	{
		const int width = devParams->width;
		const int height = devParams->height;

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int index = x + width * y;

		if (x >= width || y >= height)
		{
			return;
		}
		
		curandState state;
		curand_init(seed, 0, 0, &state);

		Scene *scene = (Scene*)devParams;

		float dx = 1.0f / width;
		float dy = 1.0f / height;
		
		float i = 2.0f*(float(x) + curand_uniform(&state))*dx - 1.0f;
		float j = 2.0f*(float(y) + curand_uniform(&state))*dy - 1.0f;

		Ray r;
		r.o = scene->camera.position;
		r.d = get_pixel_dir(scene->camera, i, -j);

		HitInfo info;
		info.t = FLT_MAX;
		Color4 contribution = Color4(0, 0, 0, 1);
		if(scene->intersect(r,info))
		{
			Material &mat = scene->materials[info.materialIndex];
			Color3 c = mat.ambient
				+ mat.diffuse * scene->lightPoint(info.point.position, info.point.normal);
			contribution = Color4(c, 1);
			
			if (x == width / 2 && y == height / 2)
			{
				//printf("Pos: (%f,%f,%f), Norm: (%f,%f,%f)\n", info.point.position.x, info.point.position.y, info.point.position.z, info.point.normal.x, info.point.normal.y, info.point.normal.z);
			}
			//contribution = Color4(info.point.position / Color3(-6,6,6), 1); // render position
			//contribution = Color4((info.point.normal + Color3(1,1,1)) / 2.0f, 1); // render normals
		}

		screen[index] = contribution;
	}

	void Renderer::render()
	{
		// bind draw buffer to device ptr
		Color4 *screen;
		cudaError_t err = cudaGLMapBufferObject((void**)&screen, drawBuffer);
		if (err != cudaSuccess)
		{
			std::cerr << "cudaGLMapBufferObject: " << cudaGetErrorName(err) << std::endl;
		}

		// call kernel to render pixels then draw to screen
		dim3 block(16,16);
		dim3 grid((dim.x + block.x - 1) / block.x, (dim.y + block.y - 1) / block.y);

		scatterTrace<<<grid,block>>>(screen,glutGet(GLUT_ELAPSED_TIME));
		cudaDeviceSynchronize();

		// unbind draw buffer so openGL can use
		cudaGLUnmapBufferObject(drawBuffer);

		// create texture from draw buffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, drawBuffer);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim.x, dim.y, GL_RGBA, GL_FLOAT, nullptr);
		GLenum glErr = glGetError();
		if (glErr != GL_NO_ERROR)
		{
			std::cerr << "glTexImage2D: " << gluErrorString(glErr) << std::endl;
		}

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
