#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"
#include <unordered_map>
#include <string>

namespace acr
{

	class Object
	{
	public:
		Object(const aiNode *node, int index, Object *parent);
		Object(const Object &obj);
		Object(Object &obj);

		char name[16];
		int index;
		int parentIndex;
		
		vector<int> children;
		vector<int> meshes;
		math::mat4 globalTransform;
		math::mat4 localTransform;
		math::mat4 globalNormalTransform;

		math::mat4 globalInverseTransform;
		math::mat4 globalInverseNormalTransform;
		
		bool intersect(const Ray &r, HitInfo &info);
	};

	class Camera
	{
	public:
		Camera();
		Camera(const aiCamera *camera);
		float aspectRatio;
		float horizontalFOV;
		math::mat4 globalTransform;
	};

	class Light
	{
	public:
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
		
		virtual float getFlux(math::vec3 position);
	};
	
	class Scene
	{
	friend class Object;
	public:
		struct Args
		{
			const char* filePath;
		};

		Scene();
		Scene(const Args &args);
		~Scene();

		bool intersect(const Ray& r, HitInfo &info);
	private:
		void loadScene(const aiScene* scene);
		void loadLights(const aiScene* scene);
		void loadMaterials(const aiScene* scene);
		void loadCamera(const aiScene* scene);
		void loadMeshes(const aiScene* scene);
		void loadObjects(const aiScene* scene);

		int loadObject(const aiNode* node, Object* parent, thrust::host_vector<Object> &objs);
		std::unordered_map<std::string, Light*> light_map;
		std::unordered_map<std::string, Camera*> camera_map;

		vector<Object>		objects;
		vector<Material>	materials;
		vector<Mesh>			meshes;
		vector<Light>			lights;
		
		int rootIndex;
		Camera camera;
	
	};

	extern Scene devScene;

} // namespace acr

#endif //_SCENE_H_
