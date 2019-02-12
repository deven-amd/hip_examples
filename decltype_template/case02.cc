#include "some_func.h"

int main(int argc, const char** argv) {

  void** pp_void;
  std::add_pointer<decltype(some_func)>::type func_ptr_void = some_func;
  func_ptr_void(pp_void);

  return 0;
}
