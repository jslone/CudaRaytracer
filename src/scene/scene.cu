#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "scene.h"
#include "utils/vector.h"
#include <cstring>

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

	Scene::Scene(const Scene::Args &args)
	{
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(args.filePath,
		                                         aiProcess_Triangulate
		                                       | aiProcess_JoinIdenticalVertices
		                                       | aiProcess_SortByPType);

		// If the import failed, report it
		if(!scene)
		{
			std::cout << "ASSIMP importer error: " << importer.GetErrorString() << std::endl;
			exit(EXIT_FAILURE);
		}

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
		std::string camName;
		loadCamera(scene,camName); //NULL CHECK
		printf("Successfully loaded 1 camera.\n");
		
		//Load lights
		std::unordered_map<std::string,int> lightMap;
		thrust::host_vector<Light> hLights(scene->mNumLights);
		loadLights(scene,hLights,lightMap);
		printf("Successfully loaded %d light(s).\n", hLights.size());
		//Load materials
		loadMaterials(scene);
		printf("Successfully loaded %d material(s).\n", materials.size());

		//Load meshes
		loadMeshes(scene);
		printf("Successfully loaded %d mesh(es).\n", meshes.size());

		//Load object hierarchy
		loadObjects(scene,camName,lightMap,hLights);
		printf("Successfully loaded hierarchy.\n");
		
		lights = vector<Light>(hLights);

		printf("Done loading.\n");
	}

	__host__
	void Scene::loadMeshes(const aiScene* scene)
	{
		meshes = vector<Mesh>(thrust::host_vector<Mesh>(scene->mMeshes, scene->mMeshes + scene->mNumMeshes));
	}

	__host__
	void Scene::loadMaterials(const aiScene* scene)
	{
		thrust::host_vector<Material> mats(scene->mMaterials, scene->mMaterials + scene->mNumMaterials);
		for (auto&& mat : mats)
		{
			Color3 &c = mat.diffuse;
			printf("Diffuse: %f, %f, %f, %f\n", c.r, c.g, c.b);
		}

		materials = vector<Material>(thrust::host_vector<Material>(scene->mMaterials, scene->mMaterials + scene->mNumMaterials));
	}
	
	__host__
	void Scene::loadCamera(const aiScene *scene, std::string &camName)
	{
		camera = Camera(scene->mCameras[0]);
		camName = std::string(scene->mCameras[0]->mName.C_Str());
	}

	__host__
	void Scene::loadLights(const aiScene* scene, thrust::host_vector<Light> &hLights, std::unordered_map<std::string,int> &lightMap)
	{
		hLights = thrust::host_vector<Light>(scene->mLights, scene->mLights + scene->mNumLights);
		
		for(int i = 0; i < scene->mNumLights; i++)
		{
			//Put light into hash so we can retrieve it later by name :'(
			std::string name = std::string(scene->mLights[i]->mName.C_Str());
			lightMap.insert({ name, i });
		}
	}
	
	__host__
	void Scene::loadObjects(const aiScene *scene, std::string &camName, std::unordered_map<std::string,int> &lightMap, thrust::host_vector<Light> &hLights)
	{
		thrust::host_vector<Object> objs;
		rootIndex = loadObject(scene->mRootNode, NULL, objs, camName, lightMap, hLights);
		std::cout << rootIndex << std::endl;
		objects = vector<Object>(objs);
	}

	__host__
	int Scene::loadObject(const aiNode* node, Object *parent, thrust::host_vector<Object> &objs, std::string &camName, std::unordered_map<std::string,int> &lightMap, thrust::host_vector<Light> &hLights)
	{
		// Initialize space, meshes, transforms
		Object tmp(node,objs.size(),parent);
		objs.push_back(tmp);
		tmp.meshes.clear();
		tmp.children.clear();

		int i = tmp.index;

		// Check to see if these objects are lights or cameras
		std::string name = std::string(node->mName.C_Str());

		math::vec3 pos = math::vec3(tmp.globalTransform * math::vec4(0, 0, 0, 1));
		std::cout << name << ": " << pos.x << "," << pos.y << "," << pos.z << std::endl;

		auto light_got = lightMap.find(name);

		if (light_got != lightMap.end())
		{
			int lIndex = light_got->second;
			Light &l = hLights[lIndex];
			l.position = math::vec3(objs[i].globalTransform * math::vec4(l.position, 1.0));
		}

		if (name == camName)
		{
			camera.position = math::vec3(tmp.globalTransform * math::vec4(camera.position,1.0f));
		}
		
		// Load children
		thrust::host_vector<int> children(node->mNumChildren);
		for(int i = 0; i < node->mNumChildren; i++)
		{
			children[i] = loadObject(node->mChildren[i], &tmp, objs, camName, lightMap, hLights);
		}
		objs[i].children = vector<int>(children);
		
		return i;
	}

	bool Scene::intersect(const Ray &r, HitInfo &info)
	{
		bool intersected = false;
		for (int i = 0; i < objects.size(); i++)
		{
			intersected = objects[i].intersect(r, info,meshes);
		}
		return intersected;
	}

	Camera::Camera(const aiCamera *cam)
	{
		aspectRatio = cam->mAspect;
		horizontalFOV = cam->mHorizontalFOV;
		position = getVec3(cam->mPosition);
		up = getVec3(cam->mUp);
		forward = getVec3(cam->mLookAt);
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

	Object::Object(const Object &obj)
		: index(obj.index)
		, parentIndex(obj.parentIndex)
		, children(obj.children)
		, meshes(obj.meshes)
		, globalTransform(obj.globalTransform)
		, globalNormalTransform(obj.globalNormalTransform)
		, globalInverseTransform(obj.globalInverseTransform)
		, globalInverseNormalTransform(obj.globalInverseNormalTransform)
	{
		for(int i = 0; i < sizeof(name); i++)
		{
			name[i] = obj.name[i];
		}
	}
	Object::Object(Object &obj)
		: index(obj.index)
		, parentIndex(obj.parentIndex)
		, children(obj.children)
		, meshes(obj.meshes)
		, globalTransform(obj.globalTransform)
		, globalNormalTransform(obj.globalNormalTransform)
		, globalInverseTransform(obj.globalInverseTransform)
		, globalInverseNormalTransform(obj.globalInverseNormalTransform)
	{
		for(int i = 0; i < sizeof(name); i++)
		{
			name[i] = obj.name[i];
		}
	}

	Object::Object(const aiNode *node, int index, Object *parent)
		: index(index)
		, parentIndex(parent ? parent->index : -1)
	{
		std::strncpy(name,node->mName.C_Str(),sizeof(name));
		getMathMatrix(node->mTransformation,localTransform);
		globalTransform = parent ? parent->globalTransform * localTransform : localTransform;
		globalInverseTransform = math::inverse(globalTransform);
		globalNormalTransform = math::transpose(globalInverseTransform);
		globalInverseNormalTransform = math::inverse(globalNormalTransform);

		meshes = vector<int>(thrust::host_vector<int>(node->mMeshes, node->mMeshes + node->mNumMeshes));
	}

	bool Object::intersect(const Ray &r, HitInfo &info, const vector<Mesh> &meshMap)
	{
		Ray lr;
		lr.o = math::vec3(globalInverseTransform * math::vec4(r.o, 1.0));
		lr.d = math::vec3(globalInverseNormalTransform * math::vec4(r.d, 1.0));

		// mesh intersection
		bool intersected = false;
		for (int i = 0; i < meshes.size(); i++)
		{
			HitInfo tmpInfo;
			tmpInfo.t = FLT_MAX;
			if (meshMap[meshes[i]].intersect(lr, tmpInfo))
			{
				intersected = true;
				tmpInfo.point.position = math::vec3(globalTransform * math::vec4(tmpInfo.point.position, 1.0));
				tmpInfo.point.normal = math::vec3(globalNormalTransform * math::vec4(tmpInfo.point.normal, 1.0));
				float t = math::length(tmpInfo.point.position - r.o);
				if (t < info.t)
				{
					info = tmpInfo;
					info.t = t;
				}
			}
		}

		return intersected;
	}
}
