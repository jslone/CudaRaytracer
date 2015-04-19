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

		void flushToDevice();
		T *devPtr;
		uint32_t length;
	private:
		thrust::device_vector<T> d;
	};


	template<typename T>
	void vector<T>::flushToDevice()
	{
		d = this;
		devPtr = thrust::raw_pointer_cast(d.data());
		length = size();
	}
} // namespace acr

#endif //_VECTOR_H_
