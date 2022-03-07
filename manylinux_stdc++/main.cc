#include <iostream>

extern void shared_library_func_1();
extern void static_library_func_1();

int main() {
  shared_library_func_1();
  static_library_func_1();
}
