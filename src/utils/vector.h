#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace acr
{
	template<typename T>
	class vector
	{
	public:
		vector<T>();
		vector<T>(const thrust::host_vector<T> &h);
		vector<T>(size_t size);
		
		T& operator[] (size_t pos);
		T& operator[] (size_t pos) const;
		
		void flushToDevice();
		size_t size();
		void push_back(T &elem);
	private:
		thrust::host_vector<T> h;
		thrust::device_vector<T> d;
		T *devPtr;
		size_t devSize;
	};

	template<typename T>
	vector<T>::vector() {}
	
	template<typename T>
	vector<T>::vector(size_t size)
		: h(size) {}
	
	template<typename T>
	vector<T>::vector(const thrust::host_vector<T> &h)
		: h(h) {}
	
	template<typename T>
	void vector<T>::flushToDevice()
	{
		d = h;
		devPtr = thrust::raw_pointer_cast(d.data());
		devSize = this->size();
	}

	template<typename T>
	void vector<T>::push_back(T &elem)
	{
		h.push_back(elem);
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
		return h[pos];
	}

	template<typename T>
	T& vector<T>::operator[] (size_t pos) const
	{		
		return h[pos];
	}

	template<typename T>
	size_t vector<T>::size()
	{
		return h.size();
	}

#endif

} // namespace acr

#endif //_VECTOR_H_
