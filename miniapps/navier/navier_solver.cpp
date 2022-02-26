// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "navier_solver.hpp"
#include "../../general/forall.hpp"

#include "../../general/nvtx.hpp"

#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace navier;

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst)
{
   Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   {
      dst->AddDomainIntegrator((*bffis)[i]);
   }
}

NavierSolver::NavierSolver(ParMesh *mesh, int order, double kin_vis)
   : pmesh(mesh), order(order), kin_vis(kin_vis),
     gll_rules(0, Quadrature1D::GaussLobatto)
{
   vfec = new H1_FECollection(order, pmesh->Dimension());
   pfec = new H1_FECollection(order);
   vfes = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
   pfes = new ParFiniteElementSpace(pmesh, pfec);

   // Check if fully periodic mesh
   if (!(pmesh->bdr_attributes.Size() == 0))
   {
      vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr = 0;

      pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      pres_ess_attr = 0;
   }

   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();

   un.SetSize(vfes_truevsize);
   un = 0.0;
   un_next.SetSize(vfes_truevsize);
   un_next = 0.0;
   unm1.SetSize(vfes_truevsize);
   unm1 = 0.0;
   unm2.SetSize(vfes_truevsize);
   unm2 = 0.0;
   fn.SetSize(vfes_truevsize);
   Nun.SetSize(vfes_truevsize);
   Nun = 0.0;
   Nunm1.SetSize(vfes_truevsize);
   Nunm1 = 0.0;
   Nunm2.SetSize(vfes_truevsize);
   Nunm2 = 0.0;
   Fext.SetSize(vfes_truevsize);
   FText.SetSize(vfes_truevsize);
   Lext.SetSize(vfes_truevsize);
   curlcurlu.SetSize(vfes_truevsize);
   resu.SetSize(vfes_truevsize);

   tmp1.SetSize(vfes_truevsize);

   pn.SetSize(pfes_truevsize);
   pn = 0.0;
   resp.SetSize(pfes_truevsize);
   resp = 0.0;
   FText_bdr.SetSize(pfes_truevsize);
   g_bdr.SetSize(pfes_truevsize);

   un_gf.SetSpace(vfes);
   un_gf = 0.0;
   un_next_gf.SetSpace(vfes);
   un_next_gf = 0.0;

   FText_gf.SetSpace(vfes);
   resu_gf.SetSpace(vfes);

   pn_gf.SetSpace(pfes);
   pn_gf = 0.0;
   resp_gf.SetSpace(pfes);

   diag_pa.SetSize(vfes_truevsize);

   diag_pa.UseDevice(true);
   fn.UseDevice(true);
   un.UseDevice(true);
   un_next.UseDevice(true);
   unm1.UseDevice(true);
   unm2.UseDevice(true);
   Nun.UseDevice(true);
   Nunm1.UseDevice(true);
   Nunm2.UseDevice(true);
   Fext.UseDevice(true);
   FText.UseDevice(true);
   Lext.UseDevice(true);
   resu.UseDevice(true);
   curlcurlu.UseDevice(true);
   tmp1.UseDevice(true);
   B1.UseDevice(true);
   B2.UseDevice(true);
   pn.UseDevice(true);
   resp.UseDevice(true);
   FText_bdr.UseDevice(true);
   g_bdr.UseDevice(true);

   cur_step = 0;

   PrintInfo();
}

void NavierSolver::Setup(double dt)
{
   MFEM_NVTX;

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
      if (partial_assembly)
      {
         mfem::out << "Using Partial Assembly" << std::endl;
      }
      else
      {
         mfem::out << "Using Full Assembly" << std::endl;
      }
   }

   sw_setup.Start();

   curl_evaluator = new CurlEvaluator(*vfes);
   curl_evaluator->EnablePA(partial_assembly);

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);

   Array<int> empty;

   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni =
      gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order - 1);

   const IntegrationRule &ir =
      gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order - 1);

   const IntegrationRule &ir_face =
      gll_rules.Get(vfes->GetMesh()->GetFaceGeometry(0), 2 * order - 1);

   mean_evaluator = new MeanEvaluator(*pfes, ir);
   bdr_nor_evaluator = new BoundaryNormalEvaluator(*vfes, *pfes, ir_face);

   nlcoeff.constant = -1.0;
   N = new ParNonlinearForm(vfes);
   auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff);
   if (numerical_integ)
   {
      nlc_nlfi->SetIntRule(&ir);
   }
   N->AddDomainIntegrator(nlc_nlfi);
   if (partial_assembly)
   {
      N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      N->Setup();
   }

   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   if (numerical_integ)
   {
      mv_blfi->SetIntRule(&ir_ni);
   }
   Mv_form->AddDomainIntegrator(mv_blfi);
   if (partial_assembly)
   {
      Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);

   Sp_form = new ParBilinearForm(pfes);
   auto *sp_blfi = new DiffusionIntegrator;
   if (numerical_integ)
   {
      sp_blfi->SetIntRule(&ir_ni);
   }
   Sp_form->AddDomainIntegrator(sp_blfi);
   if (partial_assembly)
   {
      Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);

   D_form = new ParMixedBilinearForm(vfes, pfes);
   auto *vd_mblfi = new VectorDivergenceIntegrator();
   if (numerical_integ)
   {
      vd_mblfi->SetIntRule(&ir_ni);
   }
   D_form->AddDomainIntegrator(vd_mblfi);
   if (partial_assembly)
   {
      D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, D);

   G_form = new ParMixedBilinearForm(pfes, vfes);
   auto *g_mblfi = new GradientIntegrator();
   if (numerical_integ)
   {
      g_mblfi->SetIntRule(&ir_ni);
   }
   G_form->AddDomainIntegrator(g_mblfi);
   if (partial_assembly)
   {
      G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, G);

   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);
   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff);
   auto *hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);
   if (numerical_integ)
   {
      hmv_blfi->SetIntRule(&ir_ni);
      hdv_blfi->SetIntRule(&ir_ni);
   }
   H_form->AddDomainIntegrator(hmv_blfi);
   H_form->AddDomainIntegrator(hdv_blfi);
   if (partial_assembly)
   {
      H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   g_bdr_form = new ParLinearForm(pfes);
   for (auto &vel_dbc : vel_dbcs)
   {
      auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
      if (numerical_integ)
      {
         gbdr_bnlfi->SetIntRule(&ir_ni);
      }
      g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
   }

   f_form = new ParLinearForm(vfes);
   for (auto &accel_term : accel_terms)
   {
      auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
      // @TODO: This order should always be the same as the nonlinear forms one!
      // const IntegrationRule &ir = IntRules.Get(vfes->GetFE(0)->GetGeomType(),
      //                                          4 * order);
      // vdlfi->SetIntRule(&ir);
      if (numerical_integ)
      {
         vdlfi->SetIntRule(&ir_ni);
      }
      f_form->AddDomainIntegrator(vdlfi);
   }

   if (partial_assembly)
   {
      Mv_form->AssembleDiagonal(diag_pa);
      MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MvInv = new CGSolver(vfes->GetComm());
   MvInv->iterative_mode = false;
   MvInv->SetOperator(*Mv);
   MvInv->SetPreconditioner(*MvInvPC);
   MvInv->SetPrintLevel(pl_mvsolve);
   MvInv->SetRelTol(1e-12);
   MvInv->SetMaxIter(200);

   if (partial_assembly)
   {
      lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
      SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      SpInvPC->SetPrintLevel(pl_amg);
      SpInvPC->Mult(resp, pn);
      SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC->SetOperator(*SpInvPC);
   }
   else
   {
      SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(0);
      SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC->SetOperator(*SpInvPC);
   }
   SpInv = new CGSolver(vfes->GetComm());
   SpInv->iterative_mode = true;
   SpInv->SetOperator(*Sp);
   if (pres_dbcs.empty())
   {
      SpInv->SetPreconditioner(*SpInvOrthoPC);
   }
   else
   {
      SpInv->SetPreconditioner(*SpInvPC);
   }
   SpInv->SetPrintLevel(pl_spsolve);
   SpInv->SetRelTol(rtol_spsolve);
   SpInv->SetMaxIter(200);

   if (partial_assembly)
   {
      H_form->AssembleDiagonal(diag_pa);
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HInv = new CGSolver(vfes->GetComm());
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(200);

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   un_gf.GetTrueDofs(un);
   un_next = un;
   un_next_gf.SetFromTrueDofs(un_next);

   // Set initial time step in the history array
   dthist[0] = dt;

   if (filter_alpha != 0.0)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes,
                                        pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh,
                                              vfec_filter,
                                              pmesh->Dimension());

      un_NM1_gf.SetSpace(vfes_filter);
      un_NM1_gf = 0.0;

      un_filtered_gf.SetSpace(vfes);
      un_filtered_gf = 0.0;
   }

   sw_setup.Stop();
}

void NavierSolver::UpdateTimestepHistory(double dt)
{
   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;

   // Rotate values in nonlinear extrapolation history
   Nunm2 = Nunm1;
   Nunm1 = Nun;

   // Rotate values in solution history
   unm2 = unm1;
   unm1 = un;

   // Update the current solution and corresponding GridFunction
   un_next_gf.GetTrueDofs(un_next);
   un = un_next;
   un_gf.SetFromTrueDofs(un);
}

void NavierSolver::Step(double &time, double dt, int current_step,
                        bool provisional)
{
   MFEM_NVTX;
   sw_step.Start();

   SetTimeIntegrationCoefficients(current_step);

   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs)
   {
      vel_dbc.coeff->SetTime(time + dt);
   }
   // Set current time for pressure Dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs)
   {
      pres_dbc.coeff->SetTime(time + dt);
   }

   if (H_bdfcoeff.constant != bd0 / dt)
   {
      NVTX("H_form");
      {
         H_bdfcoeff.constant = bd0 / dt;
         H_form->Update();
         H_form->Assemble();
         H_form->FormSystemMatrix(vel_ess_tdof, H);
      }
      {
         NVTX("HInv");
         HInv->SetOperator(*H);
         if (partial_assembly)
         {
            auto *HInvJac = static_cast<OperatorJacobiSmoother*>(HInvPC);
            H_form->AssembleDiagonal(diag_pa);
            HInvJac->Setup(diag_pa);
         }
      }
   }

   {
      NVTX("f_form");
      // Set time for forcing function f^{n+1}.
      for (auto &accel_term : accel_terms)
      {
         accel_term.coeff->SetTime(time + dt);
      }

      f_form->Assemble();
      f_form->ParallelAssemble(fn);
   }

   // Nonlinear extrapolated terms.
   {
      NVTX("Extrapolation");
      sw_extrap.Start();

      N->Mult(un, Nun);
      N->Mult(unm1, Nunm1);
      N->Mult(unm2, Nunm2);

      {
         const auto d_Nun = Nun.Read();
         const auto d_Nunm1 = Nunm1.Read();
         const auto d_Nunm2 = Nunm2.Read();
         auto d_Fext = Fext.Write();
         const auto ab1_ = ab1;
         const auto ab2_ = ab2;
         const auto ab3_ = ab3;
         MFEM_FORALL(i, Fext.Size(),
                     d_Fext[i] = ab1_ * d_Nun[i] +
                                 ab2_ * d_Nunm1[i] +
                                 ab3_ * d_Nunm2[i];);
      }

      fn.Add(1.0, Fext);

      // Fext = M^{-1} (F(u^{n}) + f^{n+1})
      MvInv->Mult(fn, Fext);
      iter_mvsolve = MvInv->GetNumIterations();
      res_mvsolve = MvInv->GetFinalNorm();

      // Compute BDF terms.
      {
         const double bd1idt = -bd1 / dt;
         const double bd2idt = -bd2 / dt;
         const double bd3idt = -bd3 / dt;
         const auto d_un = un.Read();
         const auto d_unm1 = unm1.Read();
         const auto d_unm2 = unm2.Read();
         auto d_Fext = Fext.ReadWrite();
         MFEM_FORALL(i, Fext.Size(),
                     d_Fext[i] += bd1idt * d_un[i] +
                                  bd2idt * d_unm1[i] +
                                  bd3idt * d_unm2[i];);
      }
   }

   sw_extrap.Stop();

   // Pressure Poisson.
   {
      NVTX("Curl curl");
      sw_curlcurl.Start();
      {
         const auto d_un = un.Read();
         const auto d_unm1 = unm1.Read();
         const auto d_unm2 = unm2.Read();
         auto d_Lext = Lext.Write();
         const auto ab1_ = ab1;
         const auto ab2_ = ab2;
         const auto ab3_ = ab3;
         MFEM_FORALL(i, Lext.Size(),
                     d_Lext[i] = ab1_ * d_un[i] +
                                 ab2_ * d_unm1[i] +
                                 ab3_ * d_unm2[i];);
      }

      curl_evaluator->ComputeCurlCurl(Lext, curlcurlu);
      Lext.Set(kin_vis, curlcurlu);

      sw_curlcurl.Stop();
   }

   {
      NVTX("Boundary");
      // \tilde{F} = F - \nu CurlCurl(u)
      subtract(Fext, Lext, FText);

      // p_r = \nabla \cdot FText
      D->Mult(FText, resp);
      resp.Neg();

      // Add boundary terms.
      bdr_nor_evaluator->Mult(FText, FText_bdr);

      g_bdr_form->Assemble();
      g_bdr_form->ParallelAssemble(g_bdr);
      resp.Add(1.0, FText_bdr);
      resp.Add(-bd0 / dt, g_bdr);
   }

   {
      NVTX("Poisson");
      if (pres_dbcs.empty())
      {
         SpInvOrthoPC->Orthogonalize(resp);
      }

      for (auto &pres_dbc : pres_dbcs)
      {
         pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
      }

      pfes->GetRestrictionOperator()->MultTranspose(resp, resp_gf);

      if (partial_assembly)
      {
         auto *SpC = Sp.As<ConstrainedOperator>();
         EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, pn, B1, 1);
      }
      else
      {
         Sp_form->FormLinearSystem(pres_ess_tdof, pn_gf, resp_gf, Sp, pn, B1, 1);
      }
      sw_spsolve.Start();
      SpInv->Mult(B1, pn);
      sw_spsolve.Stop();
      iter_spsolve = SpInv->GetNumIterations();
      res_spsolve = SpInv->GetFinalNorm();
      Sp_form->RecoverFEMSolution(pn, resp_gf, pn_gf);
   }

   {
      NVTX("Project");
      // If the boundary conditions on the pressure are pure Neumann remove the
      // nullspace by removing the mean of the pressure solution. This is also
      // ensured by the OrthoSolver wrapper for the preconditioner which removes
      // the nullspace after every application.
      mean_evaluator->MakeMeanZero(pn);

      // Project velocity.
      G->Mult(pn, resu);
      Mv->Mult(Fext, tmp1);
      // resu = tmp1 - resu
      subtract(tmp1, resu, resu);

      for (auto &vel_dbc : vel_dbcs)
      {
         un_next_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
      }

      vfes->GetRestrictionOperator()->MultTranspose(resu, resu_gf);

      if (partial_assembly)
      {
         auto *HC = H.As<ConstrainedOperator>();
         EliminateRHS(*H_form, *HC, vel_ess_tdof, un_next_gf, resu_gf, un_next, B2, 1);
      }
      else
      {
         H_form->FormLinearSystem(vel_ess_tdof, un_next_gf, resu_gf, H, un_next, B2, 1);
      }
      sw_hsolve.Start();
      HInv->Mult(B2, un_next);
      sw_hsolve.Stop();
      iter_hsolve = HInv->GetNumIterations();
      res_hsolve = HInv->GetFinalNorm();
      H_form->RecoverFEMSolution(un_next, resu_gf, un_next_gf);
   }

   un_next_gf.GetTrueDofs(un_next);

   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   if (filter_alpha != 0.0)
   {
      un_NM1_gf.ProjectGridFunction(un_gf);
      un_filtered_gf.ProjectGridFunction(un_NM1_gf);
      const auto d_un_filtered_gf = un_filtered_gf.Read();
      auto d_un_gf = un_gf.ReadWrite();
      const auto filter_alpha_ = filter_alpha;
      MFEM_FORALL(i,
                  un_gf.Size(),
                  d_un_gf[i] = (1.0 - filter_alpha_) * d_un_gf[i]
                               + filter_alpha_ * d_un_filtered_gf[i];);
   }

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      // If numerical integration is active, there is no solve (thus no
      // iterations), on the inverse velocity mass application.
      if (!numerical_integ)
      {
         mfem::out << std::setw(5) << "MVIN " << std::setw(5) << std::fixed
                   << iter_mvsolve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_mvsolve
                   << "   " << 1e-12 << "\n";
      }
      mfem::out << std::setw(5) << "PRES " << std::setw(5) << std::fixed
                << iter_spsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_spsolve << "   " << rtol_spsolve
                << "\n";
      mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed
                << iter_hsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_hsolve << "   " << rtol_hsolve
                << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }
}

void NavierSolver::EliminateRHS(Operator &A,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A.GetOutputProlongation();
   const Operator *Pi = A.GetProlongation();
   const Operator *Ri = A.GetRestriction();
   A.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_tdof_list, 0.0);
   }
   constrainedA.EliminateRHS(X, B);
}

void NavierSolver::ComputeCurl(ParGridFunction &u, ParGridFunction &cu) const
{
   curl_evaluator->ComputeCurl(u, cu);
}

double NavierSolver::ComputeCFL(ParGridFunction &u, double dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}

void NavierSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((vel_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition deteceted.");
      if (attr[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void NavierSolver::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr)
{
   pres_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Pressure Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((pres_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition deteceted.");
      if (attr[i] == 1)
      {
         pres_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(f), attr);
}

void NavierSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierSolver::AddAccelTerm(VecFuncT *f, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void NavierSolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step + 1, max_bdf_order);

   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   double rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   double rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];

   if (bdf_order == 3)
   {
      rho2 = dthist[1] / dthist[2];
   }

   if (step == 0 && bdf_order == 1)
   {
      bd0 = 1.0;
      bd1 = -1.0;
      bd2 = 0.0;
      bd3 = 0.0;
      ab1 = 1.0;
      ab2 = 0.0;
      ab3 = 0.0;
   }
   else if (step >= 1 && bdf_order == 2)
   {
      bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
      bd1 = -(1.0 + rho1);
      bd2 = pow(rho1, 2.0) / (1.0 + rho1);
      bd3 = 0.0;
      ab1 = 1.0 + rho1;
      ab2 = -rho1;
      ab3 = 0.0;
   }
   else if (step >= 2 && bdf_order == 3)
   {
      bd0 = 1.0 + rho1 / (1.0 + rho1)
            + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
      bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
      bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
      ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
      ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
      ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
   }
}

void NavierSolver::PrintTimingData()
{
   double my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_step.RealTime();
   my_rt[2] = sw_extrap.RealTime();
   my_rt[3] = sw_curlcurl.RealTime();
   my_rt[4] = sw_spsolve.RealTime();
   my_rt[5] = sw_hsolve.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP"
                << std::setw(10) << "EXTRAP" << std::setw(10) << "CURLCURL"
                << std::setw(10) << "PSOLVE" << std::setw(10) << "HSOLVE"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4]
                << std::setw(10) << my_rt[5] << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10)
                << my_rt[1] / my_rt[1] << std::setw(10) << my_rt[2] / my_rt[1]
                << std::setw(10) << my_rt[3] / my_rt[1] << std::setw(10)
                << my_rt[4] / my_rt[1] << std::setw(10) << my_rt[5] / my_rt[1]
                << "\n";

      mfem::out << std::setprecision(8);
   }
}

void NavierSolver::PrintInfo()
{
   int fes_size0 = vfes->GlobalVSize();
   int fes_size1 = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}

void NavierSolver::MeanZero(ParGridFunction &v)
{
   Vector tvec;
   v.GetTrueDofs(tvec);
   mean_evaluator->MakeMeanZero(tvec);
   v.Distribute(tvec);
}

NavierSolver::~NavierSolver()
{
   delete bdr_nor_evaluator;
   delete curl_evaluator;
   delete mean_evaluator;
   delete g_bdr_form;
   delete Mv_form;
   delete N;
   delete Sp_form;
   delete D_form;
   delete G_form;
   delete HInvPC;
   delete HInv;
   delete H_form;
   delete SpInv;
   delete MvInvPC;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete lor;
   delete f_form;
   delete MvInv;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete vfec_filter;
   delete vfes_filter;
}
