# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mtop/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MTOP_COMMON_SRC = mtop_integrators.cpp paramnonlinearform.cpp pparamnonlinearform.cpp
MMA_SRC = MMA_Benchmark.cpp mtop_MMA.cpp mtop_MMA.hpp

MTOP_COMMON_OBJ = $(MTOP_COMMON_SRC:.cpp=.o)

SEQ_MINIAPPS = seqheat
PAR_MINIAPPS = parheat
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(MTOP_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

ASTYLE_BIN = astyle
ASTYLE = $(ASTYLE_BIN) --options=$(SRC)config/mfem.astylerc
ASTYLE_VER = "Artistic Style Version 3.1"
FORMAT_FILES = $(foreach dir,$(DIRS) $(EM_DIRS) config,$(dir)/*.?pp)
FORMAT_FILES += tests/unit/*.?pp
UNIT_TESTS_SUBDIRS = general linalg mesh fem miniapps ceed
MINIAPPS_SUBDIRS = dpg/util hooke/operators hooke/preconditioners hooke/materials hooke/kernels
FORMAT_FILES += $(foreach dir,$(UNIT_TESTS_SUBDIRS),tests/unit/$(dir)/*.?pp)
FORMAT_FILES += $(foreach dir,$(MINIAPPS_SUBDIRS),miniapps/$(dir)/*.?pp)
FORMAT_EXCLUDE = general/tinyxml2.cpp tests/unit/catch.hpp
FORMAT_LIST = $(filter-out $(FORMAT_EXCLUDE),$(wildcard $(FORMAT_FILES)))

COUT_CERR_FILES = $(foreach dir,$(DIRS),$(dir)/*.[ch]pp)
COUT_CERR_EXCLUDE = '^general/error\.cpp' '^general/globals\.[ch]pp'

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := MTop miniapp
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TEST_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf SeqHeat* ParHeat*

# $(call mfem_check_command, command-to-execute, success_msg, failed_msg)
mfem_check_command = \
  if [ -t 1 ]; then red="\033[0;31m"; green="\033[0;32m"; end="\033[0m"; fi;\
  if ! $(1); then\
    printf $$green"%s"$$end"\n" "[  OK  ] "$(strip $(2));\
  else\
    printf $$red"%s"$$end"\n"   "[FAILED] "$(strip $(3)); err_code=1;\
  fi
# Verify the C++ code styling in MFEM and check that std::cout and std::cerr are
# not used in the library (use mfem::out and mfem::err instead).
style:
	@echo "Applying C++ code style..."
	@astyle_version="$$($(ASTYLE_BIN) --version)";\
	 if [ "$$astyle_version" != $(ASTYLE_VER) ]; then\
	    printf "%s\n" "Invalid astyle version: '$$astyle_version'"\
	           "Please use: '"$(ASTYLE_VER)"'";\
	    exit 1;\
	 fi
	@err_code=0;\
	$(call mfem_check_command,\
	    $(ASTYLE) $(FORMAT_LIST) | grep Formatted,\
	    "No source files were changed",\
	    "Please make sure the changes are committed");\
	echo "Checking for use of std::cout...";\
	$(call mfem_check_command,\
	   grep cout $(MMA_SRC) | grep -v $(COUT_CERR_EXCLUDE:%=-e %),\
	   "No use of std::cout found", "Use mfem::out instead of std::cout");\
	echo "Checking for use of std::cerr...";\
	$(call mfem_check_command,\
	   grep cerr $(MMA_SRC) |\
	      grep -v $(COUT_CERR_EXCLUDE:%=-e %) -e cerrno,\
	   "No use of std::cerr found", "Use mfem::err instead of std::cerr");\
	exit $$err_code
