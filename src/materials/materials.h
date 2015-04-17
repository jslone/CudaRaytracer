#ifndef _MATERIALS_H_
#define _MATERIALS_H_

#include "math/math.h"

namespace acr {

  typedef math::vec3 Color3;

  struct Material {
    Color3 ambient,diffuse,specular;
    float reflectiveIndex;
  };

} // namespace acr

#endif //_MATERIALS_H_
