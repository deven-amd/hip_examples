#include <iostream>

extern "C" void initHIP();

extern "C" void callInitHIPfromLibFoo2() {
  std::cout << "calling initHIP from Lib Foo2" << std::endl;
  initHIP();
}
