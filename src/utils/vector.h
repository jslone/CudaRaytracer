#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace acr {
  template<typename T>
  class vector : thrust::host_vector<T> {
    public:
      void flushToDevice();
      T *devPtr;
    private:
      thrust::device_vector<T> d;
  };


  template<typename T>
  void vector<T>::flushToDevice() {
    d = this;
    devPtr = thrust::raw_pointer_cast(d.data());
  }
} // namespace acr

#endif //_VECTOR_H_
