#include "some_func.h"

int main(int argc, const char** argv) {

  void** pp_void;
  some_func(pp_void);

  char** pp_char;
  some_func(pp_char);

  std::add_pointer<decltype(some_func<void>)>::type func_ptr_void = some_func;
  func_ptr_void(pp_void);

  std::add_pointer<decltype(some_func<char>)>::type func_ptr_char = some_func;
  func_ptr_char(pp_char);
  
  return 0;
}
