#include "some_func.h"

extern "C" size_t some_func(void** ptr) {
  std::cout << "some_func - void**" << std::endl;
  return 0;
}

