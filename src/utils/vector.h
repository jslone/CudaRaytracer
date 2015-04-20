#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace acr
{
	template<typename T>
	class vector : public thrust::host_vector<T>
	{
	public:
		using thrust::host_vector<T>::host_vector;

		friend T& operator[] (size_t pos);
		friend T& operator[] const (size_t pos);
		
		void flushToDevice();
		friend size_t size();
		
	private:
		thrust::device_vector<T> d;
		T *devPtr;
		size_t size;
	};


	template<typename T>
	void vector<T>::flushToDevice()
	{
		d = *this;
		devPtr = thrust::raw_pointer_cast(d.data());
		size = this->size();
	}
	
#ifdef __CUDA__ARCH__
	
	template<typename T>
	inline T& operator[] (size_t pos)
	{
		return devPtr[pos];
	}

	template<typename T>
	inline T& operator[] const (size_t pos)
	{		
		return devPtr[pos];
	}

	template<typename T>
	inline size_t size()
	{
		return length;
	}

#else

	template<typename T>
	inline T& operator[] (size_t pos)
	{
		return thrust::host_vector::operator[](pos);
	}

	template<typename T>
	inline T& operator[] const (size_t pos)
	{		
		return thrust::host_vector::operator[](pos);
	}

	template<typename T>
	inline size_t size()
	{
		return thrust::host_vector::size();
	}

#endif

} // namespace acr

#endif //_VECTOR_H_
