#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include "scene.h"
#include "utils/vector.h"

//#define LOAD_VERBOSE

namespace acr{
	Scene::Scene(const Scene::Args &args){
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(args.filePath, 
		        aiProcess_Triangulate            |
		        aiProcess_JoinIdenticalVertices  |
		        aiProcess_SortByPType);
		
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

	Scene::~Scene(){
		//Todo
	}

	inline Color3 getColor3(aiColor3D aicol){
		return Color3(aicol.r, aicol.g, aicol.b);
	}

	inline math::vec3 getVec3(aiVector3D aivec){
		return math::vec3(aivec.x, aivec.y, aivec.z);
	}

	inline glm::tvec3<float> getTvec3(aiVector3D aivec){
		return glm::tvec3<float>(aivec.x, aivec.y, aivec.z);
	}

	inline void aiVecToArray(aiVector3D* vec, float* arr, int size){
		for(int i = 0; i < size; i++){
			arr[3*i] = vec[i].x;
			arr[3*i + 1] = vec[i].y;
			arr[3*i + 2] = vec[i].z;
		}
	}

	inline void aiColToArray(aiColor4D* col, float* arr, int size){
		for(int i = 0; i < size; i++){
			arr[3*i] = col[i].r;
			arr[3*i + 1] = col[i].g;
			arr[3*i + 2] = col[i].b;
			arr[3*i + 3] = col[i].a;
		}
	}

	inline void aiIndicesToArray(aiFace* faces, uint32_t* arr, int size){
		for(int i = 0; i < size; i++){
			for(int j = 0; j < 3; j++){
				arr[3*i + j] = faces[i].mIndices[j];
			}
		}
	}

	void Scene::loadScene(const aiScene* scene){
		//Load textures ??? scene->mTextures[]	scene->mNumTextures
		printf("Loading scene...\n");
		//Load camera
		camera = loadCamera(scene->mCameras[0]); //NULL CHECK
		printf("Successfully loaded 1 camera.\n");
		//Load lights
		lights = loadLights(scene);
		printf("Successfully loaded %d light(s).\n", numLights);
		//Load materials
		materials = loadMaterials(scene);
		printf("Successfully loaded %d material(s).\n", numMaterials);

		//Load meshes
		//meshes = loadMeshes(scene);
		//printf("Successfully loaded %d mesh(es).\n", numMeshes);

		//Load object hierarchy
		objects = new Object[numMeshes]; 
		rootObject = loadObject(scene->mRootNode, NULL);
		printf("Successfully loaded hierarchy.\n");

		for(int i = 0; i < numObjects; i ++){
			printf("Object[%d]: %s\n", i, objects[i].name.c_str());
		}
	}

	Mesh* Scene::loadMeshes(const aiScene* scene){
		numMeshes = scene->mNumMeshes;
		Mesh* mesh_list = new Mesh[numMeshes];

		for(int i = 0; i < numMeshes; i++){
			//Check for null colors FIX THIS
			//Assuming mNumIndices == 3

			Mesh &mesh = mesh_list[i];
			aiMesh* m = scene->mMeshes[i];

			// Todo: 	instead of creating new memory for all this, simply
			//			recast pointers.

			float *pos = new float[3*m->mNumVertices];
			float *norms = new float[3*m->mNumVertices];
			float *cols = new float[4*m->mNumVertices];
			uint32_t *indices = new uint32_t[3*m->mNumFaces];

			aiVecToArray(m->mVertices, pos, m->mNumVertices);
			aiVecToArray(m->mNormals, norms, m->mNumVertices);

			if(m->mColors[0]){
				aiColToArray(m->mColors[0], cols, m->mNumVertices);
			}else{
				cols = nullptr;
			}

			aiIndicesToArray(m->mFaces, indices, m->mNumFaces);

			/* Alternatively
			float *pos = (float*)m->mVertices;
			float *norms = (float*)m->mNormals;
			float *cols = (float*)m->mColors;
			//Still have to do indices the slow way
			*/

			//Destructing (and hence freeing stuff that doesn't actually exist)
			mesh = Mesh(	pos,  
							norms,    
							cols,
							indices,
							m->mNumVertices,
							m->mNumFaces	);

		}

		return mesh_list;
	}

	void Scene::getMathMatrix(aiMatrix4x4& aiMatrix, math::mat4& mathMat){
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				mathMat[i][j] = aiMatrix[j][i];
			}
		}
	}

	Material* Scene::loadMaterials(const aiScene* scene){
		numMaterials = scene->mNumMaterials;
		Material* mats = new Material[numMaterials];

		for(int i = 0; i < numMaterials; i++){
			Material &mat = mats[i];
			aiMaterial* m = scene->mMaterials[i];

			aiColor3D diffuse, ambient, specular;
			m->Get(AI_MATKEY_COLOR_DIFFUSE,diffuse);
			m->Get(AI_MATKEY_COLOR_AMBIENT,ambient);
			m->Get(AI_MATKEY_COLOR_SPECULAR,specular);
			mat.diffuse = getColor3(diffuse);
			mat.ambient = getColor3(ambient);
			mat.specular = getColor3(specular);
			m->Get(AI_MATKEY_REFRACTI, mat.refractiveIndex);

			#ifdef LOAD_VERBOSE
			printf("\n");
			printf("Material[%d]:\n", i);
			printf("-->Diffuse[%f %f %f]\n", mat.diffuse.x, mat.diffuse.y, mat.diffuse.z);
			printf("-->Ambient[%f %f %f]\n", mat.ambient.x, mat.ambient.y, mat.ambient.z);
			printf("-->Specular[%f %f %f]\n", mat.specular.x, mat.specular.y, mat.specular.z);
			#endif
		}
	}

	Camera Scene::loadCamera(aiCamera* cam){
		Camera c;
		c.aspectRatio = cam->mAspect;
		c.horizontalFOV = cam->mHorizontalFOV;
		aiVector3D eye = cam->mPosition;
		aiVector3D up = cam->mUp;
		aiVector3D center = cam->mLookAt;
		c.globalTransform = math::lookAt(getTvec3(eye), getTvec3(center), getTvec3(up));

		std::string name = std::string(cam->mName.C_Str());
		camera_map.insert({name, &c});

		return c;
	}

	Light** Scene::loadLights(const aiScene* scene){
		numLights = scene->mNumLights;
		Light** ls = new Light*[numLights];
		for(int i = 0; i < numLights; i++){
			aiLight* currentLight = scene->mLights[i];
			Light* l;

			switch(currentLight->mType){
				case aiLightSource_DIRECTIONAL:
					l = (Light*)new DirectionalLight;
					((DirectionalLight*)l)->direction = getVec3(currentLight->mDirection);
					break;
				case aiLightSource_SPOT:
					l = (Light*)new SpotLight;
					((SpotLight*)l)->direction = getVec3(currentLight->mDirection);
					((SpotLight*)l)->innerConeAngle = currentLight->mAngleInnerCone;
					((SpotLight*)l)->outerConeAngle = currentLight->mAngleOuterCone;
					break;
				default:
					l = (Light*)new PointLight;
					break;
			}

			//Put light into hash so we can retrieve it later by name :'(
			std::string name = std::string(currentLight->mName.C_Str());
			light_map.insert({name, l});

			l->attConstant = currentLight->mAttenuationConstant;
			l->attLinear = currentLight->mAttenuationLinear;
			l->attQuadratic = currentLight->mAttenuationQuadratic;
			l->ambient = getColor3(currentLight->mColorAmbient);
			l->diffuse = getColor3(currentLight->mColorDiffuse);
			l->specular = getColor3(currentLight->mColorSpecular);
			l->position = getVec3(currentLight->mPosition);

			ls[i] = l;
		}
		return ls;
	}

	Object* Scene::loadObject(aiNode* node, Object* parent){
		Object* obj = &objects[numObjects];
		obj->parent = parent;
		obj->index = numObjects++;

		if(parent){
			//has parent, so get transform
			getMathMatrix(node->mTransformation, obj->localTransform);

			obj->globalTransform = parent->globalTransform * obj->localTransform;
			obj->globalInverseTransform = inverse(obj->globalTransform);
			obj->parentIndex = parent->index;
			obj->name = std::string(node->mName.C_Str());;

			std::unordered_map<std::string,Light*>::const_iterator light_got = light_map.find (obj->name);
			std::unordered_map<std::string,Camera*>::const_iterator cam_got = camera_map.find (obj->name);

			if(light_got != light_map.end()){
				Light* l = light_got->second;
				l->position = math::vec3(obj->globalTransform * math::vec4(l->position, 1.0));
			}

			if(cam_got != camera_map.end()){
				Camera* c = cam_got->second;
				camera.globalTransform = obj->globalTransform * c->globalTransform;

				#ifdef LOAD_VERBOSE
				printf("\n");
				printf("Camera:\n");
				math::vec3 pos = math::vec3(camera.globalTransform * math::vec4(0,0,0,1));
				printf("-->Position[%f %f %f]\n", pos.x, pos.y, pos.z);
				printf("-->Aspect Ratio[%f]\n", camera.aspectRatio);
				printf("-->HorizontalFOV[%f]\n", camera.horizontalFOV);
				#endif
			}

		}else{
			//no parent, must be root
			obj->globalTransform = math::mat4();
			obj->parentIndex = -1;
		}

		if(node->mNumMeshes <= 1){ //Only one mesh
			if(node->mNumMeshes > 0){
				obj->meshIndex = node->mMeshes[0];
			}else{
				obj->meshIndex = -1;
			}

			obj->numChildren = node->mNumChildren;
			obj->children = new Object*[node->mNumChildren];

			for(unsigned int i = 0; i < node->mNumChildren; i++){
				obj->children[i] = loadObject(node->mChildren[i], obj);
			}
		}
		else{ //More than one mesh
			obj->meshIndex = -1;
			obj->numChildren = node->mNumChildren + node->mNumMeshes;

			for(unsigned int i = 0; i < node->mNumMeshes; i++){
				Object* child = new Object;
				child->numChildren = 0;
				child->parent = obj;
				child->meshIndex = node->mMeshes[i];
				child->globalTransform = obj->globalTransform;
				child->globalInverseTransform = obj->globalInverseTransform;
			}

			for(unsigned int i = 0; i < node->mNumChildren; i++){
				obj->children[i+node->mNumMeshes] = loadObject(node->mChildren[i], obj);
			}
		}

		return obj;
	}

  bool Scene::intersect(const Ray &r, HitInfo &info) {
    bool intersected = false;
    for(int i = 0; i < numObjects; i++) {
      intersected = objects[i].intersect(r,info);
    }
    return intersected;
  }

  bool Object::intersect(const Ray &r, HitInfo &info) {
    Ray lr;
    lr.o = math::vec3(globalInverseTransform * math::vec4(r.o,1.0));
    lr.d = math::vec3(globalInverseNormalTransform * math::vec4(r.d,1.0));
    
    // mesh intersection
    if(false) {
      info.point.position = math::vec3(globalTransform * math::vec4(info.point.position,1.0));
      info.point.normal = math::vec3(globalNormalTransform * math::vec4(info.point.normal,1.0));
      return true;
    }
    return false;
  }
}
