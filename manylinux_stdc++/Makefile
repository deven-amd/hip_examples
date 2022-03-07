all : link_failure

libshared_library.so : shared_library.cc
	gcc -c -Wall -Werror -fpic shared_library.cc
	gcc -shared -o libshared_library.so shared_library.o

libstatic_library.a : static_library.cc
	gcc -c -Wall -Werror static_library.cc
	ar rcs libstatic_library.a static_library.o

main.o : main.cc
	gcc -c -Wall -Werror main.cc

link_failure : libstatic_library.a libshared_library.so main.o
	gcc -L. -Wl,-rpath=. -lstdc++ -fuse-ld=gold -lshared_library -lstatic_library main.o -o test_runner
	./test_runner

link_success : libstatic_library.a libshared_library.so main.o
	gcc -L. -Wl,-rpath=. -lstdc++ -fuse-ld=gold -lshared_library main.o -lstatic_library  -o test_runner
	./test_runner

clean :
	rm -rf test_runner *.o *.so *.a
