
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

# test_runner : $(SOURCE_FILE)
# 	KMDUMPLLVM=1 KMDUMPISA=1 $(CXX) $(CXXFLAGS) $^ -o $@

test_runner : $(SOURCE_FILE)
	$(CXX) $(CXXFLAGS) $^ -o $@

run_rpt :
	export HCC_PROFILE=2; ./test_runner 2> rpt_profile.log;  rpt rpt_profile.log --no_gaps

clean :
	rm -rf *.exe dump* *.log

#	$(LLVM_DIS) dump.input.bc
