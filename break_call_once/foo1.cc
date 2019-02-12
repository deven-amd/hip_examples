#include <iostream>

extern "C" void initHIP();

extern "C" void callInitHIPfromLibFoo1() {
  std::cout << "calling initHIP from Lib Foo1" << std::endl;
  initHIP();
}
