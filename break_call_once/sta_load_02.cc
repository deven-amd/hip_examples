#include <dlfcn.h>
#include <iostream>

extern "C" void initHIP();

int main() {
  
  void* handle1 = dlopen("./libfoo1.so", RTLD_LAZY);
  if (!handle1) {
    std::cout << dlerror() << "\n";
    return -1;
  }
  std::cout << "loaded libfoo1.so\n";

  void* handle2 = dlopen("./libfoo2.so", RTLD_LAZY);
  if (!handle2) {
    std::cout << dlerror() << "\n";
    dlclose(handle1);
    return -1;
  }
  std::cout << "loaded libfoo2.so\n";

  void* handle3 = dlopen("./libhip.so", RTLD_LAZY);
  if (!handle3) {
    std::cout << dlerror() << "\n";
    dlclose(handle1);
    return -1;
  }
  std::cout << "loaded libhip.so\n";

  void* sym1 = dlsym(handle1, "callInitHIPfromLibFoo1");
  if (!sym1) {
    std::cout << "unable to locate foo within libfoo1.so\n";
    std::cout << dlerror() << "\n";
    dlclose(handle1);
    dlclose(handle2);
    dlclose(handle3);
    return -1;
  }

  void* sym2 = dlsym(handle2, "callInitHIPfromLibFoo2");
  if (!sym2) {
    std::cout << "unable to locate foo within libfoo2.so\n";
    std::cout << dlerror() << "\n";
    dlclose(handle1);
    dlclose(handle2);
    dlclose(handle3);
    return -1;
  }
  
  void* sym3 = dlsym(handle3, "initHIP");
  if (!sym3) {
    std::cout << "unable to locate foo within libfoo2.so\n";
    std::cout << dlerror() << "\n";
    dlclose(handle1);
    dlclose(handle2);
    dlclose(handle3);
    return -1;
  }
  
  void(*fp1)() = reinterpret_cast<void(*)()>(sym1);
  void(*fp2)() = reinterpret_cast<void(*)()>(sym2);
  void(*fp3)() = reinterpret_cast<void(*)()>(sym3);

  fp1();
  fp1();
  
  fp2();
  fp2();
  
  std::cout << "calling dynamically loaded initHIP from main" << std::endl;
  fp3();
  fp3();
  
  std::cout << "calling statically linked initHIP from main" << std::endl;
  initHIP();
  initHIP();

  dlclose(handle1);
  dlclose(handle2);
  dlclose(handle3);

  return 0;
}
