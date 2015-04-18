#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "scene.h"

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

		//Load camera
		camera = loadCamera(scene->mCameras[0]); //NULL CHECK

		//Load lights
		lights = loadLights(scene);

		//Load materials
		materials = loadMaterials(scene);

		//Load meshes
		meshes = loadMeshes(scene);

		//Load object hierarchy
		//objects 
		rootObject = loadObject(scene->mRootNode, NULL);
	}

	Mesh* Scene::loadMeshes(const aiScene* scene){
		numMeshes = scene->mNumMeshes;
		Mesh* mesh_list = new Mesh[numMeshes];

		for(int i = 0; i < numMeshes; i++){
			//Check for null colors FIX THIS
			//Assuming mNumIndices == 3

			Mesh &mesh = mesh_list[i];
			aiMesh* m = scene->mMeshes[i];

			float *pos = new float[3*m->mNumVertices];
			float *norms = new float[3*m->mNumVertices];
			float *cols = new float[3*m->mNumVertices];
			uint32_t *indices = new uint32_t[3*m->mNumFaces];

			aiVecToArray(m->mVertices, pos, m->mNumVertices);
			aiVecToArray(m->mNormals, norms, m->mNumVertices);
			aiColToArray(m->mColors[0], cols, m->mNumVertices); //NULL CHECK
			aiIndicesToArray(m->mFaces, indices, m->mNumFaces);

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
			obj->globalTransform = obj->localTransform * parent->globalTransform;
			obj->globalInverseTransform = inverse(obj->globalTransform);
			obj->parentIndex = parent->index;
		}else{
			//no parent, must be root
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
}