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

		T& operator[] (size_t pos);
		T& operator[] (size_t pos) const;
		
		void flushToDevice();
		size_t size();
		
	private:
		thrust::device_vector<T> d;
		T *devPtr;
		size_t devSize;
	};


	template<typename T>
	void vector<T>::flushToDevice()
	{
		d = *this;
		devPtr = thrust::raw_pointer_cast(d.data());
		devSize = this->size();
	}
	
#ifdef __CUDA__ARCH__
	
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

#else

	template<typename T>
	T& vector<T>::operator[] (size_t pos)
	{
		return thrust::host_vector<T>::operator[](pos);
	}

	template<typename T>
	T& vector<T>::operator[] (size_t pos) const
	{		
		return thrust::host_vector<T>::operator[](pos);
	}

	template<typename T>
	size_t vector<T>::size()
	{
		return thrust::host_vector<T>::size();
	}

#endif

} // namespace acr

#endif //_VECTOR_H_
