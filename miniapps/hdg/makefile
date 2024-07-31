# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/hdg/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
include $(CONFIG_MK)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

hdg_poisson: hdg_poisson.o HDGBilinearForm.o hdg_integrators.o
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)

hdg_poisson.o: $(SRC)hdg_poisson.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

HDGBilinearForm.o: $(SRC)HDGBilinearForm.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

hdg_integrators.o: $(SRC)hdg_integrators.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm *.o hdg_poisson mesh solution flux lambda u_star
