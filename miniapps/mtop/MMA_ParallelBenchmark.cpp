// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Rosenbrock benchmark example

/**
 * xval: design variables
 * fval: objective value
 * dfdx: gradient of objective
 * gx: constraint values
 * dgdx: gradient of constraints
 * xmin: lower bound on design variables
 * xmax: upper bound on design variables
 * norm2: norm of KKT residual
 * normInf: infinity norm of KKT residual
 * xo1: old design variables
 * xo2: old design variables
 * U: upper asymptote
 * L: lower asymptote
 * init: initial asymptote
 * decrease: decrease factor
 * increase: increase factor
 * n: number of design variables
 * m: number of constraints
 * a: Rosenbrock parameter
 * b: Rosenbrock parameter
*/

#include "mtop_MMA.hpp"
#include <fstream>
#include <chrono>
//#include <mpi.h>

//using namespace mfem;
using namespace mma;

void MultiVar_Rosenbrock(double* xval, double a, double b, int nVar, int nCon, double* const fval,
                double* const dfdx, double* const gx, double* const dgdx);

int main(int argc, char *argv[])
{
   int nVar = 36;
   int nCon = 5;

   if (nCon > nVar)
   {
      std::cout << "The case nCon > nVar is verified in serial, but does not work in parallel (yet)." << std::endl;
      exit(1);
   }   

   mfem::MPI_Session mpi(argc, argv); // Initialize MPI (MPI Init)
   int rank, size;
   MPI_Comm communicator = MPI_COMM_WORLD;
   MPI_Comm_size(communicator, &size);
   MPI_Comm_rank(communicator, &rank);

   int nVarGlobal = nVar;
   if (nVarGlobal % size != 0)
   {
      if (rank == 0)
      {
         std::cout << "Number of variables must be divisible by number of processors" << std::endl;
      }
      MPI_Finalize();
      exit(1);
   }
   nVar = nVar / size;
   
   //Rosenbrock parameters
   double aR = 1.0;
   double bR = 100.0;
   // Objective parameters
   double epsimin = 0.0000001;
   int sx = 1;

   double* xval = new double[nVar];
   double* xvalGlobal = new double[nVarGlobal];

   double* const fval = new double[1];
   double* const dfdx = new double[nVar];
   double* const dfdxGlobal = new double[nVarGlobal];
   double* const gx = new double[nCon];
   double* const dgdx = new double[nVar * nCon];
   double* const dgdxGlobal = new double[nVarGlobal * nCon];
   double* xmin = new double[nVar];
   double* xmax = new double[nVar];
   double* xminGlobal = new double[nVarGlobal];
   double* xmaxGlobal = new double[nVarGlobal];
   double* upper = new double[nVar];
   double* lower = new double[nVar];
   // Simulation parameters
   int maxiter = 50;
   int restart = maxiter + 1;
   double norm2 = 0.0;
   double normInf = 0.0;
   double kkttol = 0.0;
   double tol = 1.0;
   // Measuring performance
   double start, end;
   double time = 0.0;
   // -- RESTART --
   // if a restart from a previous run is desired, set iter to the iteration
   // number of the previous run
   int iter = 0;
   std::string name = "Restart.dat";

   // ----------- SET UP PROBLEM ------------
   for (int i = 0; i < nVar; i++)
   {
      xval[i] = 0.0;
      xmin[i] = -2.0;
      xmax[i] = 2.0;
   }
   for (int i = 0; i < nVarGlobal; i++)
   {
      xvalGlobal[i] = 0.0;
      xminGlobal[i] = -2.0;
      xmaxGlobal[i] = 2.0;
   }

   // ---------- CALL CONSTRUCTOR -----------
   MPI_Barrier(communicator);
   start = MPI_Wtime();
   MMA MMAmain(communicator, nVar, nCon, xval, xmin, xmax, "mpiclassic" );
   end = MPI_Wtime();
   time += end - start;
   std::ofstream mma;
   mma.open("mma.dat");
   if (rank == 0)
   {
      for (int i = 0; i < nVarGlobal; i++)
      {
         mma << xvalGlobal[i] << std::endl;
      }
      for (int i = 0; i < nVarGlobal; i++)
      {
         mma << xminGlobal[i] << std::endl;
      }
      for (int i = 0; i < nVarGlobal; i++)
      {
         mma << xmaxGlobal[i] << std::endl;
      }
   }
   
   MultiVar_Rosenbrock(xvalGlobal, aR, bR, nVarGlobal, nCon, fval, dfdxGlobal, gx, dgdxGlobal);   
   // Separate dfdx, dgdx into local and global based on rank
   for (int i = 0; i < nVar; i++)
   {
      dfdx[i] = dfdxGlobal[rank * nVar + i];
   }
   for (int j = 0; j < nCon; j++)
   {
      for (int i = 0; i < nVar; i++)
      {
         dgdx[j * nVar + i] = dgdxGlobal[rank*nVar + i + j * nVarGlobal];
      }
   }

   while (iter < maxiter)
   {
      // ----------------- RUN MMA --------------------
      MPI_Barrier(communicator);
      start = MPI_Wtime();
      MMAmain.Update(iter, dfdx, gx, dgdx, xval);  
      end = MPI_Wtime();
      time += end - start;
      // Append local variable arrays into global array
      MPI_Allgather(xval, nVar, MPI_DOUBLE, xvalGlobal, nVar, MPI_DOUBLE, communicator);    
      // ---------- UPDATE DESIGN & CONSTRAINTS -------
      MultiVar_Rosenbrock(xvalGlobal, aR, bR, nVarGlobal, nCon, fval, dfdxGlobal, gx, dgdxGlobal);
      // Separate dfdx, dgdx into local based on rank
      for (int i = 0; i < nVar; i++)
      {
         dfdx[i] = dfdxGlobal[rank * nVar + i];
      }
      for (int j = 0; j < nCon; j++)
      {
         for (int i = 0; i < nVar; i++)
         {
            dgdx[j * nVar + i] = dgdxGlobal[rank*nVar + i + j * nVarGlobal];
         }
         
      }
      // ---------- PRINT FOR VISUALIZATION------------
      lower = MMAmain.getLow();
      upper = MMAmain.getUpp();
      MPI_Gather(lower, nVar, MPI_DOUBLE, xminGlobal, nVar, MPI_DOUBLE, 0, communicator);
      MPI_Gather(upper, nVar, MPI_DOUBLE, xmaxGlobal, nVar, MPI_DOUBLE, 0, communicator);
      if (rank == 0)
      {
         for (int i = 0; i < nVarGlobal; i++)
         {
            mma << xvalGlobal[i] << std::endl;
         }
         for (int i = 0; i < nVarGlobal; i++)
         {
            mma << xminGlobal[i] << std::endl;
         }
         for (int i = 0; i < nVarGlobal; i++)
         {
            mma << xmaxGlobal[i] << std::endl;
         }
         // ------------- CHECK CONVERGENCE --------------
         if (MMAmain.getKKT() < tol)
         {
            printf("KKT satisfied: Norm = %f\n", MMAmain.getKKT());
            break;
         }
         // -------------- RESTART if desired ------------
         if (iter % restart == 0)
         {
            MMAmain.outputRestart(xvalGlobal, iter, name);
         }
      }
      iter++;
   }
   if (rank == 0)
   {
      mma << "Time: " << time << std::endl;
   }
   mma.close();
   delete[] xval;
   delete[] fval;
   delete[] dfdx;
   delete[] gx;
   delete[] dgdx;
   delete[] xmin;
   delete[] xmax;
   delete[] xminGlobal;
   delete[] xmaxGlobal;
   delete[] upper;
   delete[] lower;
   delete[] xvalGlobal;
   delete[] dfdxGlobal;
   delete[] dgdxGlobal;

   MPI_Finalize();
   return 0;
}

void MultiVar_Rosenbrock(double* xval, double a, double b, int nVar, int nCon, double* const fval,
                double* const dfdx, double* const gx, double* const dgdx)
{
   
   *fval = 0.0;
   dfdx[0] = -4.0 * b * xval[0] * (xval[1] - xval[0] * xval[0]) - 2.0 * (a - xval[0]);
   *fval = b * (xval[1] - xval[0] * xval[0]) * (xval[1] - xval[0] * xval[0]) + (a - xval[0]) * (a - xval[0]);
   for (int i = 1; i < (nVar - 1); i++) 
   {
      *fval += b * (xval[i + 1] - xval[i] * xval[i]) * (xval[i + 1] - xval[i] * xval[i]) + (a - xval[i]) * (a - xval[i]);
      dfdx[i] = -4.0 * b * xval[i] * (xval[i + 1] - xval[i] * xval[i]) - 2.0 * (a - xval[i]) + 2.0 * b * (xval[i] - xval[i - 1] * xval[i - 1]);
   }
   dfdx[nVar - 1] = 2.0 * b * (xval[nVar - 1] - xval[nVar - 2] * xval[nVar - 2]);

    // Compute gx and dgdx
   for (int i = 0; i < nCon; i++)
   {
      gx[i] = 0.0;
      for (int j = 0; j < nVar; j++)
      {
         gx[i] += (xval[j] - 1.0) * (xval[j] - 1.0);
         dgdx[i * nVar + j] = 2.0 * (xval[j] - 1.0);
      }
      gx[i] -= 4.0;
   }
}
