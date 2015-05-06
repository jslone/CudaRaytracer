#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"
#include "camera.h"

#include <unordered_map>
#include <string>

namespace acr
{

	class Object
	{
	public:
		Object() = default;
		Object(const aiNode *node, int index, Object *parent);
		Object(const Object &obj);
		Object(Object &obj);

		char name[64];
		int index;
		int parentIndex;
		
		vector<int> children;
		vector<int> meshes;
		math::mat4 globalTransform;
		math::mat4 localTransform;
		math::mat4 globalNormalTransform;

		math::mat4 globalInverseTransform;
		math::mat4 globalInverseNormalTransform;
		
		__host__ __device__
		bool intersect(const Ray &r, HitInfo &info, const vector<Mesh> &meshes);
	};

	class Light
	{
	public:
		Light() = default;
		Light(const aiLight *aiLight);
		
		enum Type
		{
			POINT,
			DIRECTIONAL,
			SPOT
		};
		
		Type type;
		
		Color3 ambient;
		Color3 diffuse;
		Color3 specular;

		math::vec3 position;
		math::vec3 direction;

		float attConstant;
		float attLinear;
		float attQuadratic;

		float innerConeAngle;
		float outerConeAngle;
		
		float getFlux(math::vec3 position);
	};
	
	class Scene
	{
	friend class Object;
	public:
		struct Args
		{
			const char* filePath;
		};

		Scene() = default;
		Scene(const Args &args);
		~Scene();

		__host__ __device__
		bool intersect(const Ray& r, HitInfo &info);
	private:
		void loadScene(const aiScene* scene);
		void loadLights(const aiScene* scene, thrust::host_vector<Light> &hLights, std::unordered_map<std::string,int> &lightMap);
		void loadMaterials(const aiScene* scene);
		void loadCamera(const aiScene *scene, std::string &camName);
		void loadMeshes(const aiScene* scene);
		void loadObjects(const aiScene *scene, std::string &camName, std::unordered_map<std::string,int> &lightMap, thrust::host_vector<Light> &hLights);

		int loadObject(const aiNode* node, Object *parent, thrust::host_vector<Object> &objs, std::string &name, std::unordered_map<std::string,int> &lightMap, thrust::host_vector<Light> &hLights);
	public:
		vector<Object>		objects;
		vector<Material>	materials;
		vector<Mesh>		meshes;
		vector<Light>		lights;
		
		int rootIndex;
		Camera camera;
	
	};

} // namespace acr

#endif //_SCENE_H_
