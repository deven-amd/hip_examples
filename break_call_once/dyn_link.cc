#include <iostream>

extern "C" void callInitHIPfromLibFoo1();
extern "C" void callInitHIPfromLibFoo2();
extern "C" void initHIP();

int main() {
  
  callInitHIPfromLibFoo1();
  callInitHIPfromLibFoo1();
  
  callInitHIPfromLibFoo2();
  callInitHIPfromLibFoo2();
  
  std::cout << "calling initHIP from main" << std::endl;
  initHIP();
  initHIP();
  
  return 0;
}
