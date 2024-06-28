// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

double obj0(mfem::Vector& x)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i]*x[i];
    }
    
#ifdef MFEM_USE_MPI
    double grez;
    MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rez = grez;
#endif

    return rez;
}

double dobj0(mfem::Vector& x, mfem::Vector& dx)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i]*x[i];
        dx[i]=2.0*x[i];
    }
#ifdef MFEM_USE_MPI
    double grez;
    MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rez = grez;
#endif

    return rez;
}

double g0(mfem::Vector& x)
{
    int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i];
    }
    
    int gn = n;
#ifdef MFEM_USE_MPI
   double grez;
    MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rez = grez;
#endif

    rez=rez/gn;
    return rez-2.0;
}

double dg0(mfem::Vector& x, mfem::Vector& dx)
{
    const int n=x.Size();

    int gn = n;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i];
        dx[i]=1.0/gn;
    }

#ifdef MFEM_USE_MPI
    double grez;
    MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rez = grez;
#endif

    rez=rez/gn;
    return rez-1.0;
}

TEST_CASE("MMA Test", "[MMA]")
{
   int num_var=12;

   mfem::Vector x(num_var);
   mfem::Vector dx(num_var);
   mfem::Vector xmin(num_var); xmin=-1.0;
   mfem::Vector xmax(num_var); xmax=2.0;
   x=xmin; x+=0.5;

   mfem::MMAOpt* mmaa = nullptr;

#ifdef MFEM_USE_MPI
   mmaa = new mfem::MMAOpt(MPI_COMM_WORLD,num_var,1,x);
#else
   mmaa = new mfem::MMAOpt(num_var,1,x);
#endif

   double a[4]={0.0,0.0,0.0,0.0};
   double c[4]={1000.0,1000.0,1000.0,1000.0};
   double d[4]={0.0,0.0,0.0,0.0};

   mfem::Vector g(1); g=-1.0;
   mfem::Vector dg(num_var); dg=0.0;

   double o;
   for(int it=0;it<30;it++){
      o=dobj0(x,dx);
      g[0]=dg0(x,dg);

      std::cout<<"it="<<it<<" o="<<o<<" g="<<g[0]<<std::endl;

      for(int i=0;i<num_var;i++){
         std::cout<<" "<<x[i];
      }
      std::cout<<std::endl;
      for(int i=0;i<num_var;i++){
         std::cout<<" "<<dx[i];
      }
      std::cout<<std::endl;

      mmaa->Update(it,dx,g,dg,xmin,xmax,x);
      std::cout<<std::endl;
    }

   for(int i=0;i<num_var;i++){
      std::cout<<" "<<x[i];
   }
   std::cout<<std::endl;

   o=obj0(x);
   std::cout<<"Final o="<<o<<std::endl;

   delete mmaa;


   SECTION("Create block pattern from SparseMatrix")
   {

      REQUIRE(o == 0.000579085);
   }
}
