#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "scene.h"
#include "utils/vector.h"

//#define LOAD_VERBOSE

namespace acr
{
	inline math::vec3 getVec3(aiVector3D aivec)
	{
		return math::vec3(aivec.x, aivec.y, aivec.z);
	}

	inline void getMathMatrix(const aiMatrix4x4& aiMatrix, math::mat4& mathMat)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				mathMat[i][j] = aiMatrix[j][i];
			}
		}
	}

	Camera::Camera() {}

	Camera::Camera(const aiCamera *cam)
	{
		aspectRatio = cam->mAspect;
		horizontalFOV = cam->mHorizontalFOV;
		aiVector3D eye = cam->mPosition;
		aiVector3D up = cam->mUp;
		aiVector3D center = cam->mLookAt;
		globalTransform = math::lookAt(getVec3(eye), getVec3(center), getVec3(up));
	}

	Scene::Scene(const Scene::Args &args)
	{
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(args.filePath,
		                                         aiProcess_Triangulate
		                                       | aiProcess_JoinIdenticalVertices
		                                       | aiProcess_SortByPType);

		// If the import failed, report it
		//if(!scene)
		//To error log: importer.GetErrorString();

		// Use the scene
		//DoTheSceneProcessing( scene);
		loadScene(scene);

		//Flush scene
		//objects.flushToDevice();
		//materials.flushToDevice();
		//meshes.flushToDevice();
	}

	Scene::~Scene()
	{
		//Todo
	}

	__host__
	void Scene::loadScene(const aiScene* scene)
	{
		//Load textures ??? scene->mTextures[]	scene->mNumTextures
		printf("Loading scene...\n");
		//Load camera
		loadCamera(scene); //NULL CHECK
		printf("Successfully loaded 1 camera.\n");
		//Load lights
		loadLights(scene);
		printf("Successfully loaded %d light(s).\n", lights.size());
		//Load materials
		loadMaterials(scene);
		printf("Successfully loaded %d material(s).\n", materials.size());

		//Load meshes
		loadMeshes(scene);
		printf("Successfully loaded %d mesh(es).\n", meshes.size());

		//Load object hierarchy
		rootIndex = loadObject(scene->mRootNode, NULL);
		printf("Successfully loaded hierarchy.\n");

		for (int i = 0; i < objects.size(); i++)
		{
			printf("Object[%d]: %s\n", i, objects[i].name.c_str());
		}
	}

	__host__
	void Scene::loadMeshes(const aiScene* scene)
	{
		meshes = vector<Mesh>(thrust::host_vector<Mesh>(scene->mMeshes, scene->mMeshes + scene->mNumMeshes));
	}

	__host__
	void Scene::loadMaterials(const aiScene* scene)
	{
		materials = vector<Material>(thrust::host_vector<Material>(scene->mMaterials, scene->mMaterials + scene->mNumMaterials));
	}
	
	__host__
	void Scene::loadCamera(const aiScene *scene)
	{
		camera = Camera(scene->mCameras[0]);
		std::string name = std::string(scene->mCameras[0]->mName.C_Str());
		camera_map.insert({ name, &camera });
	}

	__host__
	void Scene::loadLights(const aiScene* scene)
	{
		lights = vector<Light>(thrust::host_vector<Light>(scene->mLights, scene->mLights + scene->mNumLights));
		
		for(int i = 0; i < scene->mNumLights; i++)
		{
			//Put light into hash so we can retrieve it later by name :'(
			std::string name = std::string(scene->mLights[i]->mName.C_Str());
			light_map.insert({ name, &lights[i] });
		}
	}

	__host__
	int Scene::loadObject(const aiNode* node, Object *parent)
	{
		// Initialize space, meshes, transforms
		Object tmp(node,objects.size(),parent);
		objects.push_back(tmp);

		// Get handle to data actually in vector
		Object &obj = objects[tmp.index];
		
		// Check to see if these objects are lights or cameras
		std::unordered_map<std::string, Light*>::const_iterator light_got = light_map.find(obj.name);
		std::unordered_map<std::string, Camera*>::const_iterator cam_got = camera_map.find(obj.name);

		if (light_got != light_map.end())
		{
			Light* l = light_got->second;
			l->position = math::vec3(obj.globalTransform * math::vec4(l->position, 1.0));
		}

		if (cam_got != camera_map.end())
		{
			Camera* c = cam_got->second;
			camera.globalTransform = obj.globalTransform * c->globalTransform;

#ifdef LOAD_VERBOSE
			printf("\n");
			printf("Camera:\n");
			math::vec3 pos = math::vec3(camera.globalTransform * math::vec4(0,0,0,1));
			printf("-->Position[%f %f %f]\n", pos.x, pos.y, pos.z);
			printf("-->Aspect Ratio[%f]\n", camera.aspectRatio);
			printf("-->HorizontalFOV[%f]\n", camera.horizontalFOV);
#endif
		}

		// Load children
		for(int i = 0; i < node->mNumChildren; i++)
		{
			obj.children[i] = loadObject(node->mChildren[i], &obj);
		}
		
		return obj.index;
	}

	bool Scene::intersect(const Ray &r, HitInfo &info)
	{
		bool intersected = false;
		for (int i = 0; i < objects.size(); i++)
		{
			intersected = objects[i].intersect(r, info);
		}
		return intersected;
	}

	Light::Light(const aiLight *aiLight)
		: attConstant(aiLight->mAttenuationConstant)
		, attLinear(aiLight->mAttenuationLinear)
		, attQuadratic(aiLight->mAttenuationQuadratic)
		, innerConeAngle(aiLight->mAngleInnerCone)
		, outerConeAngle(aiLight->mAngleOuterCone)
	{
		position = getVec3(aiLight->mPosition);
		direction = getVec3(aiLight->mDirection);

		ambient = getColor3(aiLight->mColorAmbient);
		diffuse = getColor3(aiLight->mColorDiffuse);
		specular = getColor3(aiLight->mColorSpecular);
	}

	Object::Object(const aiNode *node, int index, Object *parent)
		: name(node->mName.C_Str())
		, index(index)
		, parentIndex(parent ? parent->index : -1)
		, children(node->mNumChildren)
		, meshes(node->mMeshes, node->mMeshes + node->mNumMeshes)
	{
		getMathMatrix(node->mTransformation,localTransform);
		globalTransform = parent ? parent->globalTransform * localTransform : localTransform;
		globalInverseTransform = math::inverse(globalTransform);
		globalNormalTransform = math::transpose(globalInverseTransform);
		globalInverseNormalTransform = math::inverse(globalNormalTransform);
	}

	bool Object::intersect(const Ray &r, HitInfo &info)
	{
		Ray lr;
		lr.o = math::vec3(globalInverseTransform * math::vec4(r.o, 1.0));
		lr.d = math::vec3(globalInverseNormalTransform * math::vec4(r.d, 1.0));

		// mesh intersection
		bool intersected = false;
		for (int i = 0; i < meshes.size(); i++)
		{
			intersected = false;//objects[i].intersect(r, info);
		}

		// transform to world space
		if(intersected)
		{
			info.point.position = math::vec3(globalTransform * math::vec4(info.point.position, 1.0));
			info.point.normal = math::vec3(globalNormalTransform * math::vec4(info.point.normal, 1.0));
		}
		return intersected;
	}
}
