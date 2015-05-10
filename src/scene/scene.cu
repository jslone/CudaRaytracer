#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "scene.h"
#include "utils/vector.h"
#include "utils/bih.h"
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
		loadScene(scene);
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
		thrust::host_vector<Mesh> hMeshes(scene->mNumMeshes);
		loadMeshes(scene,hMeshes);
		printf("Successfully loaded %d mesh(es).\n", meshes.size());

		//Load object hierarchy
		loadObjects(scene,camName,lightMap,hLights,hMeshes);
		printf("Successfully loaded hierarchy.\n");
		
		lights = vector<Light>(hLights);
		meshes = vector<Mesh>(hMeshes);

		printf("Done loading.\n");
	}

	__host__
	void Scene::loadMeshes(const aiScene* scene, thrust::host_vector<Mesh> &hMeshes)
	{
		hMeshes = thrust::host_vector<Mesh>(scene->mMeshes, scene->mMeshes + scene->mNumMeshes);
	}

	__host__
	void Scene::loadMaterials(const aiScene* scene)
	{
		thrust::host_vector<Material> mats(scene->mMaterials, scene->mMaterials + scene->mNumMaterials);
		for (auto&& mat : mats)
		{
			Color3 &c = mat.diffuse;
			std::cout << "Diffuse: " << math::to_string(c) << std::endl;
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
		void Scene::loadObjects(const aiScene *scene, std::string &camName, std::unordered_map<std::string, int> &lightMap, thrust::host_vector<Light> &hLights, thrust::host_vector<Mesh> &hMeshes)
	{
		thrust::host_vector<Object> objs;
		rootIndex = loadObject(scene->mRootNode, NULL, objs, camName, lightMap, hLights, hMeshes);
		std::cout << rootIndex << std::endl;

		boundingBox.max = math::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		boundingBox.min = math::vec3(FLT_MAX, FLT_MAX, FLT_MAX);

		for (int i = 0; i < objs.size(); i++){
			boundingBox.max = math::max(boundingBox.max, objs[i].boundingBox.max);
			boundingBox.min = math::min(boundingBox.min, objs[i].boundingBox.min);
		}

		//bih = BIH<Object>(objs,boundingBox);
		objects = vector<Object>(objs);
	}

	__host__
		int Scene::loadObject(const aiNode* node, Object *parent, thrust::host_vector<Object> &objs, std::string &camName, std::unordered_map<std::string, int> &lightMap, thrust::host_vector<Light> &hLights, thrust::host_vector<Mesh> &hMeshes)
	{
		// Initialize space, meshes, transforms
		Object tmp(node,objs.size(),parent, hMeshes);
		objs.push_back(tmp);
		tmp.meshes.clear();
		tmp.children.clear();

		int i = tmp.index;

		// Check to see if these objects are lights or cameras
		std::string name = std::string(node->mName.C_Str());

		math::vec3 pos = math::translate(tmp.globalTransform, math::vec3(0, 0, 0));
		std::cout << name << ": " << math::to_string(pos) << std::endl;
		//std::cout << name << " transform: " << math::to_string(tmp.globalTransform) << std::endl;
		//std::cout << name << " normal trans: " << math::to_string(tmp.globalNormalTransform) << std::endl;

		auto light_got = lightMap.find(name);

		if (light_got != lightMap.end())
		{
			int lIndex = light_got->second;
			Light &l = hLights[lIndex];
			l.position = math::translate(objs[i].globalTransform, l.position);
			l.direction = math::translaten(tmp.globalNormalTransform, l.direction);
			std::cout << light_got->first << " pos: " << math::to_string(l.position) << std::endl;
		}

		if (name == camName)
		{
			camera.position = math::translate(tmp.globalTransform, camera.position);
			camera.forward = math::translaten(tmp.globalNormalTransform, camera.forward);
			camera.up = math::translaten(tmp.globalNormalTransform, camera.up);
			std::cout << "Camera dir: " << math::to_string(camera.forward) << std::endl;
		}
		
		// Load children
		thrust::host_vector<int> children(node->mNumChildren);
		for(int i = 0; i < node->mNumChildren; i++)
		{
			children[i] = loadObject(node->mChildren[i], &tmp, objs, camName, lightMap, hLights, hMeshes);
		}
		objs[i].children = vector<int>(children);
		
		return i;
	}

	bool Scene::intersect(const Ray &r, HitInfo &info)
	{
		bool intersected = false;
		for (int i = 0; i < objects.size(); i++)
		{
			intersected |= objects[i].intersect(r, info, meshes);
		}
		return intersected;
	}

	Color3 Scene::pointLightAccum(const Light &l, const math::vec3 &pos, const math::vec3 &norm, curandState &state)
	{
		math::vec3 dir = (l.position + 0.5f*math::randNorm(&state)) - pos;
		float t = math::length(dir);

		dir = normalize(dir);

		float cosTheta = math::max(math::dot(dir, norm), 0.0f);
		Color3 c = cosTheta / (l.attConstant + (l.attLinear + l.attQuadratic*t)*t) * l.diffuse;
		if (math::length(c) < math::epsilon<float>()) return c;

		Ray r;
		r.o = pos;
		r.d = dir;

		HitInfo info;
		info.t = t;
		if (intersect(r, info) && info.t + math::epsilon<float>() < t)
		{
			return Color3(0,0,0);
		}
		return c;
	}


	Color3 Scene::spotLightAccum(const Light &l, const math::vec3 &pos, const math::vec3 &norm)
	{
		math::vec3 dir = l.position - pos;
		float t = math::length(dir);

		dir = normalize(dir);

		float cosTheta = math::max(math::dot(dir, norm), 0.0f);
		float cosLight = math::dot(-dir, normalize(l.direction));
		float lightTheta = math::acos(cosLight);
		float inner = l.outerConeAngle - 1.57079632679f;
		float outer = l.innerConeAngle;
		float falloff_distance = outer-inner;

		if (math::abs(lightTheta) >= outer/2.0f)
			return Color3(0, 0, 0);

		Color3 c = cosTheta / (l.attConstant + (l.attLinear + l.attQuadratic*t)*t) * l.diffuse;
		if (math::length(c) < math::epsilon<float>()) return c;

		float val = (outer/2.0f - math::abs(lightTheta));
		float falloff = val / (falloff_distance/2.0f);

		if (math::abs(lightTheta) >= inner / 2.0)
			c *= falloff;

		Ray r;
		r.o = pos;
		r.d = dir;

		HitInfo info;
		info.t = t;
		if (intersect(r, info) && info.t + math::epsilon<float>() < t)
		{
			return Color3(0, 0, 0);
		}
		return c;
	}

	Color3 Scene::lightPoint(const math::vec3 &pos, const math::vec3 &norm, curandState &state)
	{
		Color3 light(0, 0, 0);
		for (int i = 0; i < lights.size(); i++)
		{
			const Light &l = lights[i];

			switch (l.type)
			{
				case Light::Type::DIRECTIONAL:
				case Light::Type::SPOT:
					light += spotLightAccum(l, pos, norm);
					break;
				case Light::Type::POINT:
					light += pointLightAccum(l, pos, norm, state);
					break;
			}
		}
		return light;
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

		switch (aiLight->mType)
		{
			case aiLightSourceType::aiLightSource_DIRECTIONAL:
				type = Type::DIRECTIONAL;
				break;
			case aiLightSourceType::aiLightSource_SPOT:
				type = Type::SPOT;
				break;
			case aiLightSourceType::aiLightSource_POINT:
			default:
				type = Type::POINT;
				break;
		}
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
		, centroid(obj.centroid)
		, boundingBox(obj.boundingBox)
	{
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
		, centroid(obj.centroid)
		, boundingBox(obj.boundingBox)
	{
	}

	BoundingBox Object::transformBoundingBox(BoundingBox bb)
	{	
		BoundingBox bounds;
		bounds.max = math::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		bounds.min = math::vec3(FLT_MAX, FLT_MAX, FLT_MAX);

		//Read bb struct as float array
		float* coords = (float*)&bb;

		//Iterate through vertices
		for (int i = 0; i < 8; i++){
			int xi = 3 * (i >> 2);				//Get 3rd bit, multiply by stride
			int yi = 3 * ((i >> 1) & 1) + 1;	//Get 2nd bit, multiply by stride, and add offset
			int zi = 3 * (i & 1) + 2;			//Get 1st bit, multiply by stride, and add offset

			//Get local and global vertices
			math::vec3 localVertex = math::vec3(coords[xi], coords[yi], coords[zi]);
			math::vec3 globalVertex = math::translate(globalTransform, localVertex);

			//Compare to current min/max
			bounds.min = math::min(globalVertex, bounds.min);
			bounds.max = math::max(globalVertex, bounds.max);
		}

		return bounds;
	}

	Object::Object(const aiNode *node, int index, Object *parent, thrust::host_vector<Mesh> &hMeshes)
		: index(index)
		, parentIndex(parent ? parent->index : -1)
	{
		getMathMatrix(node->mTransformation,localTransform);
		globalTransform = parent ? parent->globalTransform * localTransform : localTransform;
		globalInverseTransform = math::inverse(globalTransform);
		globalNormalTransform = math::transpose(math::inverse(math::mat3(globalTransform)));
		globalInverseNormalTransform = math::inverse(globalNormalTransform);

		thrust::host_vector<int> objMeshes = thrust::host_vector<int>(node->mMeshes, node->mMeshes + node->mNumMeshes);
		centroid = math::vec3(0, 0, 0);
		boundingBox.min = math::vec3(0, 0, 0);
		boundingBox.max = math::vec3(0, 0, 0);

		// Get global centroid and AABB
		if (objMeshes.size() > 0){

			math::vec3 minBound(FLT_MAX, FLT_MAX, FLT_MAX);
			math::vec3 maxBound(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			math::vec3 sumCentroids(0, 0, 0);
			for (int i = 0; i < objMeshes.size(); i++){
				sumCentroids += hMeshes[objMeshes[i]].centroid;

				minBound = math::min(minBound, hMeshes[objMeshes[i]].boundingBox.min);
				maxBound = math::max(maxBound, hMeshes[objMeshes[i]].boundingBox.max);
			}

			BoundingBox localBoundingBox;
			localBoundingBox.min = minBound;
			localBoundingBox.max = maxBound;

			//Global bounding box transform
			boundingBox = transformBoundingBox(localBoundingBox);

			/*printf("\n----------> BoundingBox\n");
			printf("min(%f, %f, %f)\n", boundingBox.min.x, boundingBox.min.y, boundingBox.min.z);
			printf("max(%f, %f, %f)\n", boundingBox.max.x, boundingBox.max.y, boundingBox.max.z);
			printf("\n");*/

			math::vec3 avgCentroid = sumCentroids / float(objMeshes.size());
			centroid = math::translate(globalTransform, avgCentroid);
		}

		// Flush object's meshes to GPU
		meshes = vector<int>(objMeshes);
	}

	bool Object::intersect(const Ray &r, HitInfo &info, const vector<Mesh> &meshMap)
	{
		Ray lr;
		lr.o = math::translate(globalInverseTransform, r.o);
		lr.d = math::translaten(globalInverseNormalTransform, r.d);

		// mesh intersection
		bool intersected = false;
		
		HitInfo tmpInfo;
		tmpInfo.t = FLT_MAX;

		for (int i = 0; i < meshes.size(); i++)
		{
			intersected |= meshMap[meshes[i]].intersect(lr, tmpInfo);
		}

		if (intersected)
		{
			tmpInfo.point.position = math::translate(globalTransform, tmpInfo.point.position);
			tmpInfo.point.normal = math::translaten(globalNormalTransform, tmpInfo.point.normal);

			float t = math::length(tmpInfo.point.position - r.o);
			if (t < info.t)
			{
				info = tmpInfo;
				info.t = t;
			}
		}

		return intersected;
	}
}
