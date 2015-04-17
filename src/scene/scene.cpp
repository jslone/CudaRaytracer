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

	inline glm::tvec3<float> getTvec3(aiVector3D aivec){
		return glm::tvec3<float>(aivec.x, aivec.y, aivec.z);
	}

	void Scene::loadScene(const aiScene* scene){
		//scene->mCameras[]		scene->mNumCameras
		//scene->mLights[]		scene->mNumLights
		//scene->mMaterials[]	scene->mNumMaterials
		//scene->mMeshes[]		scene->mNumMeshes
		//scene->mTextures[]	scene->mNumTextures

		//THINGS TO DO:

		//Load camera
		Camera c;
		c.aspectRatio = scene->mCameras[0]->mAspect;
		c.horizontalFOV = scene->mCameras[0]->mHorizontalFOV;
		aiVector3D eye = scene->mCameras[0]->mPosition;
		aiVector3D up = scene->mCameras[0]->mUp;
		aiVector3D center = scene->mCameras[0]->mLookAt;
		c.globalTransform = math::lookAt(getTvec3(eye), getTvec3(center), getTvec3(up));
		camera = c;

		//Load lights
		//Load materials
		//Load textures

		//Load meshes

		//Load object hierarchy
		root = loadNode(scene->mRootNode, NULL);
	}

	math::mat4& Scene::getMathMatrix(aiMatrix4x4 aiMatrix){
		math::mat4 mat;
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				mat[i][j] = aiMatrix[j][i];
			}
		}
		return mat;
	}

	Object* Scene::loadNode(aiNode* node, Object* parent){
		Object* obj = new Object;
		obj->parent = parent;
		obj->localTransform = getMathMatrix(node->mTransformation);

		//if has parent, get transform
		if(parent){
			obj->globalTransform = obj->localTransform * parent->globalTransform;
			obj->globalInverseTransform = inverse(obj->globalTransform);
		}

		if(node->mNumMeshes <= 1){ //Only one mesh
			if(node->mNumMeshes > 0){
				obj->meshIndex = node->mMeshes[0];
			}
			else{
				obj->meshIndex = -1;
			}

			obj->numChildren = node->mNumChildren;
			obj->children = new Object*[node->mNumChildren];

			for(unsigned int i = 0; i < node->mNumChildren; i++){
				obj->children[i] = loadNode(node->mChildren[i], obj);
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
				obj->children[i+node->mNumMeshes] = loadNode(node->mChildren[i], obj);
			}
		}

		return obj;
	}
}