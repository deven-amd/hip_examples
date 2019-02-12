#include <mutex>
#include <iostream>

#define MAKE_STRING_HELPER(X) #X
#define MAKE_STRING(X) MAKE_STRING_HELPER(X)

std::once_flag flag;

extern "C" void initHIP() {
  try {
    std::cout << " ----- in initHIP -- " << MAKE_STRING(BUILD) << std::endl;
    std::call_once(flag, []() { std::cout << "  <<<<<<   initializing HIP runtime >>>>>>>>>>> \n"; });
  } catch(const std::exception& e) {
    std::cout << "exception in initHIP " << e.what() << std::endl;
  }
}
