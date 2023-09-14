#ifndef __MTOP_MMA__HPP
#define __MTOP_MMA__HPP

#include "mfem.hpp"
#include <iostream>
#include <mpi.h>

namespace mma
{

class MMA
{
private:

   // Private: MMA-specific
   double asyinit, asyincr, asydecr, xmamieps, lowmin, lowmax, uppmin, uppmax, zz;
   double *factor;
   // Global: Old design variables
   double *xo1, *xo2;
   //Convergence study
   double kktnorm;
   bool isInitialized = false;

   //MPI
   MPI_Comm comm;

   void setGlobals(int nVar, int nCon);
   void setMMA(int nVar, int nCon);
   void freeGlobals();
   void freeMMA();

   // SUBPROBLEM BASE CLASS
   class SubProblemBase
   {
   protected:
      MMA* mma_ptr;

   public:
      SubProblemBase(MMA* mma);
      virtual ~SubProblemBase()
      {
         this->mma_ptr = nullptr;
      };
      virtual void Perform(double* const dfdx, double* const gx, double* const dgdx, double* const xval) = 0;
   };

   // SubProblem according to Svanberg, implemented in serial
   class SubProblemClassic : public SubProblemBase
   {
   private:
      int ittt, itto, itera;
      double epsi, rez, rezet, delz, dz, dzet, azz, stmxx, stmalfa, stmbeta, stmalbe,
             sum, stmalbexx, stminv, steg, zold, zetold, residunorm, residumax, resinew, raa0, albefa, move, xmamieps;
      double *sum1, *ux1, *xl1, *plam, *qlam, *gvec, *residu, *GG, *delx, *dely, *dellam, 
             *dellamyi, *diagx, *diagy, *diaglam, *diaglamyi, *bb, *bb1, *Alam, *AA, *AA1,
             *dlam, *dx, *dy, *dxsi, *deta, *dmu, *Axx, *axz, *ds, *xx, *dxx, *stepxx,
             *stepalfa, *stepbeta, *xold, *yold, *lamold, *xsiold, *etaold, *muold, *sold,
             *p0, *q0, *P, *Q, *alfa, *beta, *xmami, *b;

      void setSubProb(int nVar, int nCon);
      void freeSubProb();
      double kktcheck(double* y, double* const dfdx, double* const gx,
                      double* const dgdx, double* x);

   public:

      SubProblemClassic(MMA* mma, int nVar, int nCon);
      ~SubProblemClassic()
      {
         this->freeSubProb();
      };
      void Perform(double* const dfdx, double* const gx, double* const dgdx, double* const xval)  override;
   };

   // SubProblem according to Svanberg, implemented in parallel
   class SubProblemClassicMPI : public SubProblemBase
   {
   private:
      int ittt, itto, itera;
      double epsi, rez, rezet, delz, dz, dzet, azz, stmxx, stmalfa, stmbeta,
             sum, stminv, steg, zold, zetold,
             residunorm, residumax, resinew, raa0, albefa, move, xmamieps;
      double *sum1, *ux1, *xl1, *plam, *qlam, *gvec, *residu, *GG, *delx, *dely, *dellam, 
             *dellamyi, *diagx, *diagy, *diaglam, *diaglamyi, *bb, *bb1, *Alam, *AA, *AA1,
             *dlam, *dx, *dy, *dxsi, *deta, *dmu, *Axx, *axz, *ds, step, *xold, *yold,
             *lamold, *xsiold, *etaold, *muold, *sold, *p0, *q0, *P, *Q, *alfa, *beta, *xmami, *b;
             
      void setSubProb(int nVar, int nCon);
      void freeSubProb();
      double* getResidual(int rank);
      void getDelta(int i);
      void avoidNaN(double* avoid);

   public:
      SubProblemClassicMPI(MMA* mma, int nVar, int nCon);
      ~SubProblemClassicMPI()
      {
         this->freeSubProb();
      };
      void Perform(double* const dfdx, double* const gx, double* const dgdx, double* const xval) override;
   };

   //instance of class
   SubProblemBase *mSubProblem;

public:

   enum SubProblemType
   {
      CLASSIC,
      MPIClassic,
      ENDENUM
   };

   std::map<std::string, enum SubProblemType> typeMap;

   // Local vectors
   double *a, *b, *c, *d;
   double a0, machineEpsilon, epsimin;
   double z, zet;
   int nCon, nVar;

   // Global: Asymptotes, bounds, objective approx., constraint approx.
   double *xmin, *xmax, *low, *upp;
   double *x, *y, *xsi, *eta, *lam, *mu, *s;


   // Construct using subproblem
   MMA(int nVar, int nCon, double *xval, double *xxmin, double *xxmax,
       SubProblemType type);

   MMA(int nVar, int nCon, double *xval, double *xxmin, double *xxmax,
       std::string type = "classic");
       
   MMA(MPI_Comm comm, int nVar, int nCon, double *xval, double *xxmin, double *xxmax,
       std::string type = "mpiclassic");

   ~MMA()
   {
      this->freeGlobals();
      this->freeMMA();
   }

   void initializeGlobals(int nVariables, int nConstraints, double *xval,
                          double *xxmin, double *xxmax)
   {
      nVar = nVariables;
      nCon = nConstraints;
      xmin = xxmin;
      xmax = xxmax;

      for (int i = 0; i < nCon; i++)
      {
         a[i] = 0.0;
         c[i] = 1000.0;
         d[i] = 1.0;
      }
      a0 = 1.0;

      if (!isInitialized)
      {
         mfem::mfem_error("MMA::initialize() member data not initialized, call setGlobals first");
      }
   }

   void SubProblemFactory(int nVar, int nCon, SubProblemType type);

   void Update(int iter, double* const dfdx, double* const gx,
               double* const dgdx, double* xval);

   double* getLow();
   double* getUpp();
   double getKKT();

   // Options
   // Return necessary data for restart
   void setRestart(double* xval, int iter, std::string name,
                   std::string path = "./");
   void outputRestart(double* xval, int iter, std::string name,
                      std::string path = "./");

};

}

#endif

            // // Assemble list containing all used dof types. Dof types are not unique
            // MPI_Gatherv(
            //         ( ( aPdofTypeList.data() ).data() ),
            //         tNumLocalDofTypes,
            //         MPI_UNSIGNED,
            //         ( mPdofTypeList.data() ).data(),
            //         ( tNumLocalDofTypesList.data() ).data(),
            //         ( tDofTypeOffset.data() ).data(),
            //         MPI_UNSIGNED,
            //         0,
            //         MPI_COMM_WORLD );

            // // Temporary variable for mPdofTypeList size
            // uint tPdofTypeListSize;

            // if ( par_rank() == 0 )
            // {
            //     // Sort this created list
            //     std::sort( ( mPdofTypeList.data() ).data(), ( mPdofTypeList.data() ).data() + mPdofTypeList.size() );

            //     // use std::unique and std::distance to create list containing all used dof types. This list is unique
            //     auto last = std::unique( ( mPdofTypeList.data() ).data(), ( mPdofTypeList.data() ).data() + mPdofTypeList.size() );
            //     auto pos  = std::distance( ( mPdofTypeList.data() ).data(), last );

            //     mPdofTypeList.resize( pos );

            //     tPdofTypeListSize = mPdofTypeList.size();
            // }

            // // Bcast size of mPdofTypeList on processor 0
            // MPI_Bcast( &tPdofTypeListSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
