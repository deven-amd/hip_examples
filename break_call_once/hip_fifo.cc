#include <mutex>
#include <iostream>
#include <string>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAKE_STRING_HELPER(X) #X
#define MAKE_STRING(X) MAKE_STRING_HELPER(X)

class CallOnceFIFO {

public:
  
  CallOnceFIFO(std::string pfx) {
    name_ = "/tmp/" + pfx + std::to_string(getpid());
  }
  
  ~CallOnceFIFO() {
    mkfifo(name_.c_str(), 0666);
    remove(name_.c_str());
  }

  void operator()(void (*func)()) {
    if (0 == mkfifo(name_.c_str(), 0666)) {
      func();
    }
  }

private:
  // todo:
  // disable the default, copy and move constructors
  // disable the copy and move assignment operators
  
  std::string name_;
};


extern "C" void initHIP() {
  
  static CallOnceFIFO call_once("initHIP");
  
  try {
    std::cout << " ----- in initHIP -- " << MAKE_STRING(BUILD) << std::endl;
    call_once([]() { std::cout << "  <<<<<<   initializing HIP runtime >>>>>>>>>>> \n"; });
  } catch(const std::exception& e) {
    std::cout << "exception in initHIP " << e.what() << std::endl;
  }
}

