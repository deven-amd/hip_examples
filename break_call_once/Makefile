
HCC:=/opt/rocm/bin/hcc -std=c++11

all : clean break fix

break : clean hip_call_once.cc foo1.cc foo2.cc sta_load_02.cc
	$(HCC) -c hip_call_once.cc -DBUILD=STATIC -o hip_static.o && ar rcs libhip_static.a hip_static.o
	$(HCC) -fPIC -shared -lpthread hip_call_once.cc -DBUILD=DYNAMIC -o libhip.so
	$(HCC) -fPIC -shared -lpthread foo1.cc -o libfoo1.so -L. -Wl,--rpath=. -lhip 
	$(HCC) -fPIC -shared -lpthread foo2.cc -o libfoo2.so -L. -Wl,--rpath=. -lhip 
	 $(HCC) sta_load_02.cc -L. -lpthread -lhip_static -ldl -o $@
	@echo "Running $@" 
	-./$@

fix : clean hip_fifo.cc foo1.cc foo2.cc sta_load_02.cc
	$(HCC) -c hip_fifo.cc -DBUILD=STATIC -o hip_static.o && ar rcs libhip_static.a hip_static.o
	$(HCC) -fPIC -shared -lpthread hip_fifo.cc -DBUILD=DYNAMIC -o libhip.so
	$(HCC) -fPIC -shared -lpthread foo1.cc -o libfoo1.so -L. -Wl,--rpath=. -lhip 
	$(HCC) -fPIC -shared -lpthread foo2.cc -o libfoo2.so -L. -Wl,--rpath=. -lhip 
	 $(HCC) sta_load_02.cc -L. -lpthread -lhip_static -ldl -o $@
	@echo "Running $@" 
	-./$@

clean :
	rm -rf *.so *.a *.o break fix

