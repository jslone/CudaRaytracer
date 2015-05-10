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
		__device__ __host__
		vector<T>() = default;
		
		__device__ __host__
		vector<T>(const vector<T> &v) = default;
		
		__host__
		vector<T>(const thrust::host_vector<T> &h);

		__device__ __host__
		T& operator[] (size_t pos);
		
		__device__ __host__
		T& operator[] (size_t pos) const;
		
		__device__ __host__
		size_t size();

		__device__ __host__
		size_t position(const T *elem);

		__device__ __host__
		void clear();
		
		__host__
		void free();
	private:
		T *devPtr;
		size_t devSize;
	};

	template<typename T>
	vector<T>::vector(const thrust::host_vector<T> &h)
	{
		devSize = h.size();
		cudaMalloc((void**)&devPtr, devSize * sizeof(T));
	
		thrust::device_ptr<T> thrustPtr(devPtr);

		thrust::copy(h.begin(), h.end(), thrustPtr);
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

	template<typename T>
	size_t vector<T>::position(const T *elem)
	{
		return devPtr - elem;
	}
	
	template<typename T>
	void vector<T>::free()
	{
		if(devPtr)
			cudaFree(devPtr);
		clear();
	}
	
	template<typename T>
	void vector<T>::clear()
	{
		devPtr = nullptr;
		devSize = 0;
	}

} // namespace acr

#endif //_VECTOR_H_
