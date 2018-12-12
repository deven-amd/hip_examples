#include <iostream>
#include <type_traits>
#include <hip/hip_fp16.h>

// namespace std {

//   template<>
//     struct __is_floating_point_helper<half>
//     : public true_type { };

// };

using namespace std;

#define CHECK(T) \
  cout << "int: " << std::is_floating_point<T>::value << endl
  

int
main(int argc, const char** argv) {

  cout << "is_floating_point:" << endl;
  
  CHECK(int);

  CHECK(double);

  CHECK(float);

  CHECK(half);
   
  return 0;
}
