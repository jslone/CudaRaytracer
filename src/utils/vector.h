#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace acr
{
	template<typename T>
	class vector
	{
	public:
		vector<T>();
		vector<T>(vector<T> &v);
		vector<T>(const thrust::host_vector<T> &h);
		~vector<T>();

		T& operator[] (size_t pos);
		T& operator[] (size_t pos) const;
		
		size_t size();
	private:
		T *devPtr;
		size_t devSize;
	};

	template<typename T>
	vector<T>::vector() {}
	
	template<typename T>
	vector<T>::vector(vector<T> &v)
		: devPtr(v.devPtr)
		, devSize(v.devSize)
	{
		v.devPtr = nullptr;
		v.devSize = 0;
	}

	template<typename T>
	vector<T>::vector(const thrust::host_vector<T> &h)
	{
		devSize = h.size();
		cudaMalloc((void**)&devPtr, devSize * sizeof(T));
	
		thrust::device_ptr<T> thrustPtr(devPtr);

		thrust::copy(h.begin(), h.end(), thrustPtr);
	}

	template<typename T>
	vector<T>::~vector()
	{
		if(devPtr)
			cudaFree(devPtr);
		devSize = 0;
	}
	
	template<typename T>
	T& vector<T>::operator[] (size_t pos)
	{
		return devPtr[pos];
	}

	template<typename T>
	T& vector<T>::operator[] (size_t pos) const
	{		
		return devPtr[pos];
	}

	template<typename T>
	size_t vector<T>::size()
	{
		return devSize;
	}

} // namespace acr

#endif //_VECTOR_H_
