//#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "mtop_MMA.hpp"
#include "DiffusionSolver.hpp"
#include <mpi.h>
//

using namespace mma;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   mfem::MPI_Session mpi(argc, argv);
   //MPI_Init(&argc, &argv);
   //MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Define Caliper ConfigManager
#ifdef MFEM_USE_CALIPER
   cali::ConfigManager mgr;
#endif

   // Caliper instrumentation
   MFEM_PERF_FUNCTION;

   // 2. Parse command-line options
   const char *mesh_file = "./ThermalComplianceMesh.g";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 1;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 10;
   int print_level = 0;

   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 100;

   bool ConstSensFD =false;
   bool ObjSensFD =false;
   bool dQdTFD =false;
   bool dQdsFD =false;
   bool dRdsFD =false;
   bool TF = false;
   bool BreakAfterFirstIt = false;
   bool initializeRandom = false;

   bool initializeSol = false;
   bool restartDesign = false;

   const char *petscrc_file = "";

   const char* cali_config = "runtime-report";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.AddOption(&cali_config, "-p", "--caliper",
                  "Caliper configuration string.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }
   // mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Caliper configuration
#ifdef MFEM_USE_CALIPER
   mgr.add(cali_config);
   mgr.start();
#endif

   //    Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.

   std::string meshName= "ThermalComplianceMesh.g";
   mfem::Mesh *mesh = new mfem::Mesh(meshName.c_str(),0,0);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }


   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Allocate the nonlinear diffusion solver
   mfem::Diffusion_Solver* solver=new mfem::Diffusion_Solver(pmesh,1);
   mfem::PDEFilter* solverFiltered=new mfem::PDEFilter(pmesh,
                                                       1); // <----------------------------------------- FILTER

   //add boundary conditions
   solver->AddDirichletBC(1,0.0);


   mfem::ParFiniteElementSpace *fes = solver->GetFES();

   // build h1 design space
   int orderDesign = 1;
   ::mfem::H1_FECollection desFECol_H1(orderDesign, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   //gradients with respect to the design field
   mfem::Vector objgrad(desFESpace_scalar_H1.GetTrueVSize());
   objgrad=0.0; //of the objective
   mfem::Vector volgrad(desFESpace_scalar_H1.GetTrueVSize());
   volgrad=0.0; //of the volume contr.

   // set design variable bounds
   mfem::Vector xxmax(desFESpace_scalar_H1.GetTrueVSize());
   mfem::Vector xxmin(desFESpace_scalar_H1.GetTrueVSize());


   // design variable vector
   mfem::ParGridFunction designVarVec(&desFESpace_scalar_H1); designVarVec=0.3;

   mfem::ParGridFunction designVarVecFiltered(&desFESpace_scalar_H1);
   designVarVecFiltered=0.0; // <----------- FILTER

   mfem::Vector & trueDesVar = designVarVec.GetTrueVector();

   if (initializeRandom)
   {
      for (int Ij = 0; Ij< desFESpace_scalar_H1.GetTrueVSize(); Ij++)
      {
         trueDesVar[Ij] = rand() / double(RAND_MAX);
         //designVarVecFiltered[Ij] = rand() / double(RAND_MAX); // <--------------------------------------------- FILTER
      }
   }

   designVarVec.SetFromTrueVector();

   std::cout<<"opt"<<std::endl;

   if (restartDesign)
   {
      std::string tStringIn = "DesignVarVec";
      int n = 6;
      std::string tWorlsRank = std::to_string( mpi.WorldRank());

      int precision = n - tWorlsRank.size();
      std::string s = std::string(precision, '0').append(tWorlsRank);

      tStringIn= tStringIn +"."+s;

      std::ifstream inp(tStringIn);
      mfem::ParGridFunction tLoadGF(pmesh, inp);

      designVarVec = tLoadGF;

      for ( int Ik=0; Ik<designVarVec.Size(); Ik++)
      {
         if (designVarVec[Ik] < 0.1) {designVarVec[Ik]=0.1;}
      }
   }

   double max_ch=0.01; //max design change

   double ThermalCompliance; //energy dissipation  // cpl
   double vol; //volume

   double max_vol = 1.0;
   double maxVolAllowed = max_vol*0.5;

   mfem::ParaViewDataCollection paraview_dc("TopOpt", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);

   solver->SetLinearSolver(1e-10, 1e-12, 1000);

   if (true)
   {
      int nVar = trueDesVar.Size();
      int nCon = 1;
      // for (int li=0; li<nVar; li++)
      // {
      //    xxmin[li]=1e-3;
      //    xxmax[li]=1.0;
      // }
      mfem::ParGridFunction xxmin(&desFESpace_scalar_H1); xxmin=1e-3;
      mfem::ParGridFunction xxmax(&desFESpace_scalar_H1); xxmax=1.0;

      mfem::Vector & truexxmin = xxmin.GetTrueVector();
      mfem::Vector & truexxmax = xxmax.GetTrueVector();
      //--------------------------------------------------------------------------------
      // Set Up MMA
      MMA MMAmain(nVar, nCon, trueDesVar.GetData(), truexxmin.GetData(),
                  truexxmax.GetData());
      //--------------------------------------------------------------------------------
      std::cout<<"opt iter"<<std::endl;

      for (int i=1; i<max_it; i++)
      {
         solverFiltered->SetDesignGF(
            &designVarVec); // <--------------------------------------------------------- FILTER
         solverFiltered->FSolve();
         solverFiltered->GetSol(designVarVecFiltered);
         //designVarVecFiltered.Print();
         //solve design variable and solve
         solver->SetDesignGF(&designVarVecFiltered);

         solver->FSolve();

         mfem::ParGridFunction TempGF;
         solver->GetSol(TempGF);

         // evaluate obj and volume
         mfem::ThermalComplianceQoI tObj;
         tObj.SetFESpaceAndField(fes, &designVarVecFiltered,  &TempGF);

         std::cout<<"obj"<<std::endl;
         mfem::VolumeQoI tContraint;
         tContraint.SetDesignFES( &desFESpace_scalar_H1 );
         tContraint.SetDesField( &designVarVecFiltered );

         std::cout<<"val"<<std::endl;
         ThermalCompliance =tObj.Eval();
         vol               =tContraint.Eval();

         if (myrank==0)
         {
            std::cout<<"it: "<<i<<" | obj= "<<ThermalCompliance<<" | vol= "<<vol<<" | Constraint: "<<
                     vol-maxVolAllowed << std::endl;
         }

         tObj      .Grad(objgrad);     //dQ/dT   explicit
         tContraint.Grad(volgrad);     //dV/dT   explicit

         std::cout<<"EVAL: "<<std::endl;


         mfem::ThermalComplianceIntegrator_1 * ImpdQdTIntegrator = new
         mfem::ThermalComplianceIntegrator_1;
         ImpdQdTIntegrator->SetDesignAndTempGF(&designVarVecFiltered, &TempGF);
         mfem::ParLinearForm ParLinerFormDQDT(fes);
         ParLinerFormDQDT.AddDomainIntegrator(ImpdQdTIntegrator);

         ParLinerFormDQDT.Assemble();                             // dQ/dT       adjoint load

         //ParLinerFormDQDT.Print();
         solver->ASolve(
            ParLinerFormDQDT);                       // adjoint solve    dR/dT^(-T) * dQ/dT = lambda

         mfem::ParGridFunction tAdjointGF;
         solver->GetAdj(tAdjointGF);

         std::cout<<"Adjoint Solve done"<<std::endl;

         // ---- post-multiply -----

         mfem::DiffusionAdjointPostIntegrator * AdjointPostIntegrator = new
         mfem::DiffusionAdjointPostIntegrator;
         AdjointPostIntegrator->SetAdjoint(&tAdjointGF);
         AdjointPostIntegrator->SetDesignAndTempGF(&designVarVecFiltered, &TempGF);
         mfem::ParLinearForm ParLinerFormPostAdjoint(&desFESpace_scalar_H1);
         ParLinerFormPostAdjoint.AddDomainIntegrator(AdjointPostIntegrator);

         //ParLinerFormPostAdjoint.Assemble();                    // lambda * dR/ds

         //mfem::Vector & linFormPostAdjoint =
         //mfem::Vector & linFormPostAdjoint = ParLinerFormPostAdjoint.ParallelAssemble(); // maybe make pointer: mfem::Vector *
         ParLinerFormPostAdjoint.ParallelAssemble();
         objgrad -= ParLinerFormPostAdjoint;                     // dQ/ds(explicit) - labda * dR/ds

         if (TF)
         {
            mfem::ParGridFunction tFD_diff(&desFESpace_scalar_H1); tFD_diff = 0.0;
            tFD_diff = TempGF;
            tFD_diff -=tAdjointGF;
            tFD_diff.Print();
            std::cout<<"ConstSensFD norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if (ConstSensFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(&desFESpace_scalar_H1); tFD_sens = 0.0;
            for ( int Ia = 0; Ia<designVarVec.Size(); Ia++)
            {
               designVarVec[Ia] +=epsilon;

               mfem::VolumeQoI tContraintFD_1;
               tContraintFD_1.SetDesignFES( &desFESpace_scalar_H1 );
               tContraintFD_1.SetDesField( &designVarVec );
               double volFD_1  =tContraint.Eval();

               designVarVec[Ia] -=2.0*epsilon;
               mfem::VolumeQoI tContraintFD_2;
               tContraintFD_2.SetDesignFES( &desFESpace_scalar_H1 );
               tContraintFD_2.SetDesField( &designVarVec );
               double volFD_2  =tContraint.Eval();

               designVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (volFD_1-volFD_2)/(2.0*epsilon);
            }//
            //volgrad.Print();
            std::cout<<"  ----------  FD Diff ------------"<<std::endl;
            // tFD_sens.Print();

            std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
            mfem::ParGridFunction tFD_diff(&desFESpace_scalar_H1); tFD_diff = 0.0;
            tFD_diff = volgrad;
            tFD_diff -=tFD_sens;
            tFD_diff.Print();

            std::cout<<"ConstSensFD norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if (ObjSensFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(&desFESpace_scalar_H1); tFD_sens = 0.0;
            for ( int Ia = 0; Ia<designVarVec.Size(); Ia++)
            {
               designVarVec[Ia] +=epsilon;

               mfem::Diffusion_Solver* solverObj=new mfem::Diffusion_Solver(pmesh,1);

               //add boundary conditions
               solverObj->AddDirichletBC(1,0.0);

               solverObj->SetLinearSolver(1e-10, 1e-12, 1000);
               //solve design variable and solve
               solverObj->SetDesignGF(&designVarVec);

               solverObj->FSolve();

               mfem::ParGridFunction TempGF2;
               solverObj->GetSol(TempGF2);


               // mfem::ParGridFunction tPreassureGF_FD;
               // solver->GetSol(tPreassureGF_FD);

               mfem::ThermalComplianceQoI tObj_FD1;
               tObj_FD1.SetFESpaceAndField(fes, &designVarVec,  &TempGF2);

               double energyDissipFD1 =tObj_FD1.Eval();



               designVarVec[Ia] -=2.0*epsilon;


               mfem::Diffusion_Solver* solverObj1=new mfem::Diffusion_Solver(pmesh,1);

               //add boundary conditions
               solverObj1->AddDirichletBC(1,0.0);

               solverObj1->SetLinearSolver(1e-10, 1e-12, 1000);
               //solve design variable and solve
               solverObj1->SetDesignGF(&designVarVec);

               solverObj1->FSolve();

               mfem::ParGridFunction TempGF3;
               solverObj1->GetSol(TempGF3);


               // mfem::ParGridFunction tPreassureGF_FD;
               // solver->GetSol(tPreassureGF_FD);

               mfem::ThermalComplianceQoI tObj_FD2;
               tObj_FD2.SetFESpaceAndField(fes, &designVarVec,  &TempGF3);

               double energyDissipFD2 =tObj_FD2.Eval();

               designVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
               std::cout<<"Var number: "<< Ia<< " Analytic: "<<objgrad[Ia] << " FD: "<<
                        tFD_sens[Ia]<<std::endl;
            }

            // objgrad.Print();
            std::cout<<"  ---------- FD obj ------------"<<std::endl;
            // tFD_sens.Print();
            std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
            mfem::ParGridFunction tFD_diff(&desFESpace_scalar_H1); tFD_diff = 0.0;
            tFD_diff = objgrad;
            tFD_diff -=tFD_sens;
            tFD_diff.Print();

            std::cout<<"norml2: "<<tFD_diff.Norml2()<<"normllinf: "<<tFD_diff.Normlinf()
                     <<"max/min: "<<tFD_diff.Max()<<" / "<<tFD_diff.Min()<<std::endl;
         }

         if (dQdTFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for ( int Ia = 0; Ia<TempGF.Size(); Ia++)
            {
               TempGF[Ia] +=epsilon;

               mfem::ThermalComplianceQoI tObj_FD1;
               tObj_FD1.SetFESpaceAndField(fes, &designVarVec,  &TempGF);

               double energyDissipFD1 =tObj_FD1.Eval();

               TempGF[Ia] -=2.0*epsilon;

               mfem::ThermalComplianceQoI tObj_FD2;
               tObj_FD2.SetFESpaceAndField(fes, &designVarVec,  &TempGF);

               double energyDissipFD2 =tObj_FD2.Eval();

               TempGF[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
            }
            ParLinerFormDQDT.Print();
            std::cout<<"  ----------  FD Diff ------------"<<std::endl;
            tFD_sens.Print();

            std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
            mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
            tFD_diff = ParLinerFormDQDT;
            tFD_diff -=tFD_sens;
            tFD_diff.Print();
            std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if (dQdsFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for ( int Ia = 0; Ia<designVarVec.Size(); Ia++)
            {
               designVarVec[Ia] +=epsilon;

               mfem::ThermalComplianceQoI tObj_FD1;
               tObj_FD1.SetFESpaceAndField(fes, &designVarVec,  &TempGF);

               double energyDissipFD1 =tObj_FD1.Eval();

               designVarVec[Ia] -=2.0*epsilon;

               mfem::ThermalComplianceQoI tObj_FD2;
               tObj_FD2.SetFESpaceAndField(fes, &designVarVec,  &TempGF);

               double energyDissipFD2 =tObj_FD2.Eval();

               designVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
            }
            objgrad.Print();
            std::cout<<"  ----------  FD Diff ------------"<<std::endl;
            tFD_sens.Print();

            std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
            mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
            tFD_diff = objgrad;
            tFD_diff -=tFD_sens;
            tFD_diff.Print();
            std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         // if(dRdsFD)
         // {
         //    double epsilon = 1e-8;
         //    mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
         //    for( int Ia = 0; Ia<designVarVec.Size(); Ia++)
         //    {
         //       designVarVec[Ia] +=epsilon;

         //       mfem::EnergyDissipationObjective tObj_FD1;
         //       tObj_FD1.SetNLDiffusionSolver( solver );
         //       tObj_FD1.SetPreassure( &tPreassureGF);
         //       tObj_FD1.SetDesignFES( fes );
         //       tObj_FD1.SetDesField( designVarVec );
         //       tObj_FD1.SetNLDiffusionCoeff( tMatCoeff );

         //       double energyDissipFD1 =tObj_FD1.Eval();

         //       designVarVec[Ia] -=2.0*epsilon;

         //       mfem::EnergyDissipationObjective tObj_FD2;
         //       tObj_FD2.SetNLDiffusionSolver( solver );
         //       tObj_FD2.SetPreassure( &tPreassureGF);
         //       tObj_FD2.SetDesignFES( fes );
         //       tObj_FD2.SetDesField( designVarVec );
         //       tObj_FD2.SetNLDiffusionCoeff( tMatCoeff );

         //       double energyDissipFD2 =tObj_FD2.Eval();

         //       designVarVec[Ia] +=epsilon;

         //       tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
         //    }
         //       objgrad.Print();
         //       std::cout<<"  ----------  FD Diff ------------"<<std::endl;
         //       tFD_sens.Print();

         //       std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
         //       mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
         //       tFD_diff = objgrad;
         //       tFD_diff -=tFD_sens;
         //       tFD_diff.Print();
         //                               std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         // }

         if (BreakAfterFirstIt)
         {
            mfem::mfem_error("break before update");
         }
         mfem::ParGridFunction dThermCompds;
         solverFiltered->ASolve(objgrad);
         solverFiltered->GetAdj(dThermCompds);
         solverFiltered->ASolve(volgrad);
         solverFiltered->freeMemory();
         mfem::ParGridFunction DensGFFiltered;
         solverFiltered->GetAdj(DensGFFiltered);
         //DensGFFiltered *= -1.0;
         //dThermCompds *= -1.0;

         double con=vol/maxVolAllowed
                    -1;                                      // V/V_max -1
         volgrad /= maxVolAllowed;

         //////////////////////
         mfem::Vector & TrueThermalComp = dThermCompds.GetTrueVector();
         mfem::Vector & TrueDensGFFiltered = DensGFFiltered.GetTrueVector();

         // MMA Routine: Update(iteration, objective gradient, constraint(s), constraint gradients,  design variables)
         MMAmain.Update(i, TrueThermalComp.GetData(), &con,
                        TrueDensGFFiltered.GetData(), trueDesVar.GetData());

         designVarVec.SetFromTrueVector();
         //std::cout << "Con = " << con << ", Obj = " << ThermalCompliance << std::endl;
         {


            paraview_dc.SetCycle(i);
            paraview_dc.SetTime(i*1.0);
            paraview_dc.RegisterField("design",&designVarVec);
            paraview_dc.RegisterField("FilteredDesign",&designVarVecFiltered);
            mfem::ParGridFunction objGradGF(&desFESpace_scalar_H1); objGradGF = objgrad;
            paraview_dc.RegisterField("ObjGrad",&objGradGF);
            mfem::ParGridFunction ConstraintGF(&desFESpace_scalar_H1);
            ConstraintGF = volgrad;
            paraview_dc.RegisterField("Constraint",&ConstraintGF);
            paraview_dc.RegisterField("Temp",&TempGF);
            paraview_dc.RegisterField("FilteredObjGrad",&dThermCompds);
            paraview_dc.RegisterField("FilteredConGrad",&DensGFFiltered);
            paraview_dc.Save();



            mfem::ParaViewDataCollection paraview_dc2("topOptAft", pmesh);
            paraview_dc2.SetPrefixPath("ParaView");
            paraview_dc2.SetLevelsOfDetail(order);
            paraview_dc2.SetDataFormat(mfem::VTKFormat::BINARY);
            paraview_dc2.SetHighOrderOutput(true);

            paraview_dc2.SetCycle(i);
            paraview_dc2.SetTime(i*1.0);
            paraview_dc2.RegisterField("design",&designVarVec);
            paraview_dc2.RegisterField("filteredDesign",&designVarVecFiltered);
            paraview_dc2.Save();


         }
         std::string tDesignName = "designVarVec";
         designVarVec.Save( tDesignName.c_str() );

         std::string tFieldName = "FieldVec";
         TempGF.Save( tFieldName.c_str() );

      }

   }

   //delete DesignCoeff;
   delete solver;
   delete solverFiltered;
   delete pmesh;
   // Flush output before MPI_finalize
#ifdef MFEM_USE_CALIPER
   mgr.flush();
#endif
   MPI_Finalize();

   return 0;
}

