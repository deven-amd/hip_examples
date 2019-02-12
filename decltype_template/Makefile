HCC:=/opt/rocm/bin/hcc -std=c++11

.PHONY : case01 case02 case03

case01 : clean
	$(HCC) -fPIC -shared some_func.cc -o libsome_func.so
	$(HCC) case01.cc -L. -Wl,--rpath=. -lsome_func  -o $@
	@echo "Running $@"
	./$@

case02 : clean
	$(HCC) -fPIC -shared some_func.cc -o libsome_func.so
	$(HCC) case02.cc -L. -Wl,--rpath=. -lsome_func  -o $@
	@echo "Running $@"
	./$@

case03 : clean
	$(HCC) -fPIC -shared some_func.cc -o libsome_func.so
	$(HCC) case02.cc -DAPPLY_FIX -L. -Wl,--rpath=. -lsome_func  -o $@
	@echo "Running $@"
	./$@

clean :
	rm -rf *.so case01 case02 case03
