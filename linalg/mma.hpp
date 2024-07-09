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

#ifndef MFEM_MMA
#define MFEM_MMA

#include <iostream>
#include "vector.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

class MMA{
public:

    /// Serial constructor:
    /// nVar - number of design parameters;
    /// nCon - number of constraints;
    /// xval[nVar] - initial parameter values
    MMA(int nVar, int nCon, double *xval);

#ifdef MFEM_USE_MPI
    /// Parallel constructor:
    /// nVar - number of design parameters;
    /// nCon - number of constraints;
    /// xval[nVar] - initial parameter values
    MMA(MPI_Comm comm_, int nVar, int nCon, double *xval);
#endif

    /// Destructor
    ~MMA();

    /// Update the optimization parameters
    /// iter - current iteration number
    /// dfdx[nVar] - gradients of the objective
    /// gx[nCon] - values of the constraints
    /// dgdx[nCon*nVar] - gradients of the constraints
    /// xxmin[nVar] - lower bounds
    /// xxmax[nVar] - upper bounds
    /// xval[nVar] - (input: current optimization parameters)
    /// xval[nVar] - (output: updated optimization parameters)
    void Update(int iter, const double* dfdx, const double* gx,
                const double* dgdx,
                const double* xxmin,const double* xxmax,
                double* xval);


    /// Dump the internal state into a file
    /// xval[nVar] - current optimization parameters
    /// iter - current interation
    /// fname, fpath - file name and file path
    void WriteState(double* xval, int iter,
                   std::string fname,
                   std::string fpath = "./");

    /// Load the internal state
    void LoadState(double* xval, int iter,
                   std::string fname,
                   std::string fpath = "./");

protected:
    // Local vectors
    double *a, *b, *c, *d;
    double a0, machineEpsilon, epsimin;
    double z, zet;
    int nCon, nVar;

    // Global: Asymptotes, bounds, objective approx., constraint approx.
    double *low, *upp;
    double *x, *y, *xsi, *eta, *lam, *mu, *s;


private:

    // MMA-specific
    double asyinit, asyincr, asydecr;
    double xmamieps, lowmin, lowmax, uppmin, uppmax, zz;
    double *factor;

    /// values from the previous two iterations
    double *xo1, *xo2;

    /// KKT norm
    double kktnorm;

    /// intialization state
    bool isInitialized = false;

#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif

    /// Allocate the memory for MMA
    void AllocData(int nVar, int nCon);

    /// Free the memory for MMA
    void FreeData();

    /// Initialize data
    void  InitData(double *xval);

    /// Subproblem base class
    class MMASubBase
    {
    public:
        /// Constructor
        MMASubBase(MMA* mma){mma_ptr=mma;}

        /// Destructor
        virtual ~MMASubBase(){}

        /// Update the optimization parameters
        virtual
        void Update(const double* dfdx,
                    const double* gx,
                    const double* dgdx,
                    const double* xmin,
                    const double* xmax,
                    const double* xval)=0;

    protected:
        MMA* mma_ptr;

    };

    MMASubBase* mSubProblem;

    friend class MMASubSerial;

    /// Serial subproblem class
    class MMASubSerial:public MMASubBase{
    public:
        /// Constructor
        MMASubSerial(MMA* mma, int nVar, int nCon):MMASubBase(mma)
        {
            AllocSubData(nVar,nCon);
        }

        /// Destructor
        virtual
        ~MMASubSerial(){FreeSubData();}

        /// Update the optimization parameters
        virtual
        void Update(const double* dfdx,
                    const double* gx,
                    const double* dgdx,
                    const double* xmin,
                    const double* xmax,
                    const double* xval);
    private:
        /// Allocate the memory for the subproblem
        void AllocSubData(int nVar, int nCon);

        /// Free the memeory for the subproblem
        void FreeSubData();

        /// Return the KKT norm
        double KKTNorm(double* y,
                       const double* dfdx,
                       const double* gx,
                       const double* dgdx,
                       const double* xmin,
                       const double* xmax,
                       double* x);

        ///
        int ittt, itto, itera;

        ///
        double epsi, rez, rezet;
        double delz, dz, dzet, azz;
        double stmxx, stmalfa, stmbeta, stmalbe;
        double sum, stmalbexx, stminv, steg;
        double zold, zetold;
        double residunorm, residumax, resinew;
        double raa0, albefa, move, xmamieps;

        ///
        double *sum1, *ux1, *xl1, *plam, *qlam, *gvec, *residu, *GG, *delx, *dely, *dellam,
               *dellamyi, *diagx, *diagy, *diaglam, *diaglamyi, *bb, *bb1, *Alam, *AA, *AA1,
               *dlam, *dx, *dy, *dxsi, *deta, *dmu, *Axx, *axz, *ds, *xx, *dxx, *stepxx,
               *stepalfa, *stepbeta, *xold, *yold, *lamold, *xsiold, *etaold, *muold, *sold,
               *p0, *q0, *P, *Q, *alfa, *beta, *xmami, *b;
               
    };


    friend class MMASubParallel;

    class MMASubParallel:public MMASubBase{
    public:

        /// Constructor
        MMASubParallel(MMA* mma, int nVar, int nCon):MMASubBase(mma)
        {
            AllocSubData(nVar,nCon);

        }

        /// Destructor
        virtual
        ~MMASubParallel(){
            FreeSubData();
        }

        /// Update the optimization parameters
        virtual
        void Update(const double* dfdx,
                    const double* gx,
                    const double* dgdx,
                    const double* xmin,
                    const double* xmax,
                    const double* xval);

    private:
        int ittt, itto, itera;

        double epsi, rez, rezet, delz, dz, dzet, azz, stmxx, stmalfa, stmbeta,
               sum, stminv, steg, zold, zetold,
               residunorm, residumax, resinew, raa0, albefa, move, xmamieps;

        double *sum1, *ux1, *xl1, *plam, *qlam, *gvec, *residu, *GG, *delx, *dely, *dellam,
               *dellamyi, *diagx, *diagy, *diaglam, *diaglamyi, *bb, *bb1, *Alam, *AA, *AA1,
               *dlam, *dx, *dy, *dxsi, *deta, *dmu, *Axx, *axz, *ds, *xx, *dxx, *stepxx, *stepalfa, *stepbeta, step, *xold, *yold,
               *lamold, *xsiold, *etaold, *muold, *sold, *p0, *q0, *P, *Q, *alfa, *beta, *xmami, *b;

        // parallel helper variables
        double global_max = 0.0;
        double global_norm = 0.0;
        double stmxx_global = 0.0;
        double stmalfa_global = 0.0;
        double stmbeta_global = 0.0;

        double *b_local, *gvec_local, *Alam_local, *sum_local, *sum_global;

        /// Allocate the memory for the subproblem
        void AllocSubData(int nVar, int nCon);

        /// Free the memeory for the subproblem
        void FreeSubData();
    };
};

/// Native MFEM MMA interface
class MMAOpt{
public:
    /// Default constructor
    MMAOpt(int nVar, int nCon, mfem::Vector& xval)
    {
        opt=new mfem::MMA(nVar,nCon, xval.GetData());
    }

#ifdef MFEM_USE_MPI
    MMAOpt(MPI_Comm comm_, int nVar, int nCon, mfem::Vector& xval)
    {
        int rank = 0;
        MPI_Comm_rank(comm_, &rank);

        // create new communicator 
        char subcommunicator;
        int colour;

        if( 0 != nVar)
        {
            subcommunicator = 'A';
            colour = 0;
        }
        else
        {
            subcommunicator = 'B';
            colour = MPI_UNDEFINED;
        } 

        // Split de global communicator
        MPI_Comm_split(comm_, colour, rank, &new_comm);

        opt=new mfem::MMA(new_comm, nVar,nCon, xval.GetData());
    }
#endif

        /// Destructor
    ~MMAOpt(){
        delete opt;
    }

    /// Design update
    void Update(int iter, const mfem::Vector& dfdx,
                const mfem::Vector& gx, const mfem::Vector& dgdx,
                const mfem::Vector& xmin, const mfem::Vector& xmax,
                mfem::Vector& xval){
        opt->Update(iter, dfdx.GetData(),
                    gx.GetData(),dgdx.GetData(),
                    xmin.GetData(), xmax.GetData(),
                    xval.GetData());
    }

private:
    mfem::MMA* opt;// the actual mma optimizer

#ifdef MFEM_USE_MPI
    MPI_Comm new_comm;
#endif

};

} // mfem namespace

#endif // MFEM_MMA
