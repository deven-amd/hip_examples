#include <iostream>

extern "C" size_t some_func(void** ptr);

#if !defined(APPLY_FIX)

template<typename T>
size_t some_func(T** ptr) {
  std::cout << "some_func - T**" << std::endl;
  return 1;
}

#endif
