
CXX = /opt/rocm/hip/bin/hipcc
LLVM_DIS = llvm-dis

CXXFLAGS = -Wall
#CXXFLAGS += -Werror 

#SOURCE_FILE=src/fp16_check_ushort_as_half.cpp

#SOURCE_FILE=src/fp16_check_hisinf.cpp
#SOURCE_FILE=src/fp16_check_hisnan.cpp
#SOURCE_FILE=src/int64_check_abs.cpp
#SOURCE_FILE=src/double_check_lgamma.cpp
#SOURCE_FILE=src/double_check_lgamma_simple.cpp
#SOURCE_FILE=src/double_check_atomicAdd.cpp
#SOURCE_FILE=src/uint64_check_atomicAdd.cpp
#SOURCE_FILE=src/double_check_longlong_as_double.cpp
#SOURCE_FILE=src/pass_array_to_device_func.cpp
SOURCE_FILE=src/simple_test.cpp

all : clean run_check.exe

run_check.exe : $(SOURCE_FILE)
	KMDUMPLLVM=1 KMDUMPISA=1 $(CXX) $(CXXFLAGS) $^ -o $@
	$(LLVM_DIS) dump.input.bc

# run_check.exe : $(SOURCE_FILE)
# 	$(CXX) $(CXXFLAGS) $^ -o $@

clean :
	rm -rf *.exe dump*
