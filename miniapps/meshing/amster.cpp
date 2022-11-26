// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//    ---------------------------------------------------------------------
//    Amsterdam 2023 code -- AMSTER (Automatic Mesh SmooThER)
//    ---------------------------------------------------------------------
//
//
// Compile with: make amster
//
// Sample runs:
//
//    2D untangling:
//      mpirun -np 4 amster -m jagged.mesh -o 2 -qo 4 -no-wc -no-fit
//    2D untangling + worst-case:
//      mpirun -np 4 amster -m amster_q4warp.mesh -o 2 -qo 6 -no-fit
//    2D fitting:
//      mpirun -np 6 amster -m amster_q4warp.mesh -rs 1 -o 3 -no-wc -amr 7
//
//    3D untangling:
//      mpirun -np 6 amster -m ../../../mfem_data/cube-holes-inv.mesh -o 3 -qo 4 -no-wc -no-fit

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"
#include "amster.hpp"

using namespace mfem;
using namespace std;

void Untangle(ParGridFunction &x, double min_detA, int quad_order);
void WorstCaseOptimize(ParGridFunction &x, int quad_order);

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "jagged.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   bool worst_case       = true;
   bool fit_optimize     = true;
   int solver_iter       = 50;
   int quad_order        = 8;
   int bg_amr_steps      = 6;
   double surface_fit_const = 10.0;
   double surface_fit_adapt = 10.0;
   double surface_fit_threshold = 1e-7;
   int metric_id         = 2;
   int target_id         = 1;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&worst_case, "-wc", "--worst-case",
                               "-no-wc", "--no-worst-case",
                  "Enable worst case optimization step.");
   args.AddOption(&fit_optimize, "-fit", "--fit_optimize",
                                 "-no-fit", "--no-fit-optimize",
                  "Enable optimization with tangential relaxation.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&bg_amr_steps, "-amr", "--amr-bg-steps",
                  "Number of AMR steps on the background mesh.");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();
   const int NE = pmesh->GetNE();

   delete mesh;

   // Define a finite element space on the mesh.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(pmesh, &fec, dim);
   pmesh->SetNodalFESpace(&pfes);

   // Get the mesh nodes as a finite element grid function in fespace.
   ParGridFunction x(&pfes);
   pmesh->SetNodalGridFunction(&x);

   // Save the starting (prior to the optimization) mesh to a file.
   ostringstream mesh_name;
   mesh_name << "amster_in.mesh";
   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->PrintAsOne(mesh_ofs);

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;

   // Compute the minimum det(A) of the starting mesh.
   double min_detA = infinity();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         min_detA = fmin(min_detA, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detA, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detA << endl; }

   // Metric.
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2)
   {
      switch (metric_id)
      {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.1); break;
      }
   }
   else { metric = new TMOP_Metric_302; }

   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
   case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
   case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
   case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
   }
   auto target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x0);

   // Visualize the starting mesh and metric values.
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   // If needed, untangle with fixed boundary.
   if (min_detA < 0.0) { Untangle(x, min_detA, quad_order); }

   // If needed, perform worst-case optimization with fixed boundary.
   if (worst_case) { WorstCaseOptimize(x, quad_order); }

   // Visualize the starting mesh and metric values.
   {
      char title[] = "After Untangl / WC";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   if (fit_optimize == false) { return 0; }

   // Average quality and worst-quality for the mesh.
   double integral_mu = 0.0, volume = 0.0, max_mu = -1.0;
   for (int i = 0; i < NE; i++)
   {
      const FiniteElement &fe_pos = *x.FESpace()->GetFE(i);
      const IntegrationRule &ir = IntRulesLo.Get(fe_pos.GetGeomType(), 10);
      const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

      DenseMatrix dshape(dof, dim);
      DenseMatrix pos(dof, dim);
      pos.SetSize(dof, dim);
      Vector posV(pos.Data(), dof * dim);

      Array<int> pos_dofs;
      x.FESpace()->GetElementVDofs(i, pos_dofs);
      x.GetSubVector(pos_dofs, posV);

      DenseTensor W(dim, dim, nsp);
      DenseMatrix Winv(dim), T(dim), A(dim);
      target_c->ComputeElementTargets(i, fe_pos, ir, posV, W);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         metric->SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         const double mu = metric->EvalW(T);
         max_mu = fmax(mu, max_mu);
         integral_mu += mu * ip.weight * A.Det();
         volume += ip.weight * A.Det();
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &max_mu, 1, MPI_DOUBLE, MPI_MAX, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &integral_mu, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());
   if (myid == 0)
   {
      cout << "Max mu: " << max_mu << endl
           << "Avg mu: " << integral_mu / volume << endl;
   }

   // Detect boundary nodes.
   Array<int> vdofs;
   ParFiniteElementSpace pfes_s(pmesh, &fec);
   ParGridFunction domain(&pfes_s);
   domain = 1.0;
   for (int i = 0; i < pfes_s.GetNBE(); i++)
   {
      pfes_s.GetBdrElementDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { domain(vdofs[j]) = 0.0; }
   }

   // Compute size field.
   ParGridFunction size_gf(&pfes_s);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *pfes.GetFE(e);
      const IntegrationRule &ir_nodes = fe.GetNodes();
      const int nqp = ir_nodes.GetNPoints();
      ElementTransformation &Tr = *pmesh->GetElementTransformation(e);
      auto n_fe = dynamic_cast<const NodalFiniteElement *>(&fe);
      const Array<int> &lex_order = n_fe->GetLexicographicOrdering();
      Vector loc_size(nqp);
      Array<int> dofs;
      pfes.GetElementDofs(e, dofs);
      for (int q = 0; q < nqp; q++)
      {
         Tr.SetIntPoint(&ir_nodes.IntPoint(q));
         loc_size(lex_order[q]) = Tr.Weight();
      }
      size_gf.SetSubVector(dofs, loc_size);
   }
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, size_gf,
                             "Size", 0, 0, 300, 300, "Rj");
   }
   DiffuseH1(size_gf, 2.0);
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, size_gf,
                             "Size", 300, 0, 300, 300, "Rj");
   }

   // Distance to the boundary, on the original mesh.
   GridFunctionCoefficient coeff(&domain);
   ParGridFunction dist(&pfes_s);
   ComputeScalarDistanceFromLevelSet(*pmesh, coeff, dist, false);
   dist *= -1.0;
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, dist,
                             "Dist to Boundary", 0, 700, 300, 300, "Rj");

      VisItDataCollection visit_dc("amster_in", pmesh);
      visit_dc.RegisterField("distance", &dist);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }

   // Create the background mesh.
   ParMesh *pmesh_bg = NULL;
   Mesh *mesh_bg = NULL;
   if (dim == 2)
   {
      mesh_bg = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true));
   }
   else if (dim == 3)
   {
      mesh_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON, true));
   }
   mesh_bg->EnsureNCMesh();
   pmesh_bg = new ParMesh(MPI_COMM_WORLD, *mesh_bg);
   delete mesh_bg;
   // Set curvature to linear because we use it with FindPoints for
   // interpolating a linear function later.
   int mesh_bg_curv = mesh_poly_deg;
   pmesh_bg->SetCurvature(mesh_bg_curv, false, -1, 0);

   // Make the background mesh big enough to cover the original domain.
   Vector p_min(dim), p_max(dim);
   pmesh->GetBoundingBox(p_min, p_max);
   GridFunction &x_bg = *pmesh_bg->GetNodes();
   const int num_nodes = x_bg.Size() / dim;
   for (int i = 0; i < num_nodes; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         double length_d = p_max(d) - p_min(d),
                extra_d = 0.2 * length_d;
         x_bg(i + d*num_nodes) = p_min(d) - extra_d +
                                 x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
      }
   }

#ifndef MFEM_USE_GSLIB
   MFEM_ABORT("GSLIB needed for this functionality.");
#endif

   // The background level set function is always linear to avoid oscillations.
   H1_FECollection bg_fec(mesh_bg_curv, dim);
   ParFiniteElementSpace bg_pfes(pmesh_bg, &bg_fec);
   ParGridFunction bg_domain(&bg_pfes);

   // Refine the background mesh around the boundary.
   OptimizeMeshWithAMRForAnotherMesh(*pmesh_bg, dist, bg_amr_steps, bg_domain);
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, bg_domain,
                             "Dist on Background", 300, 700, 300, 300, "Rj");
   }
   // Rebalance par mesh because of AMR
   pmesh_bg->Rebalance();
   bg_pfes.Update();
   bg_domain.Update();

   {
       VisItDataCollection visit_dc("amster_bg", pmesh_bg);
       visit_dc.RegisterField("distance", &bg_domain);
       visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
       visit_dc.Save();
   }

   // Compute min element size.
   double min_dx = std::numeric_limits<double>::infinity();
   for (int e = 0; e < pmesh_bg->GetNE(); e++)
   {
      min_dx = fmin(min_dx, pmesh_bg->GetElementSize(e));
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_dx, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

   // Shift the zero level set by ~ one element inside.
   const double alpha = 0.75*min_dx;
   bg_domain -= alpha;

   // Compute a distance function on the background.
   GridFunctionCoefficient ls_filt_coeff(&bg_domain);
   ComputeScalarDistanceFromLevelSet(*pmesh_bg, ls_filt_coeff, bg_domain, true);
   bg_domain *= -1.0;

   // Offset back to the original position of the boundary.
   bg_domain += alpha;
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, bg_domain,
                             "Final LS", 600, 700, 300, 300, "Rjmm");

      VisItDataCollection visit_dc("amster_bg", pmesh_bg);
      visit_dc.RegisterField("distance", &bg_domain);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // Surface fitting.
   Array<bool> surf_fit_marker(domain.Size());
   ParGridFunction surf_fit_mat_gf(&pfes_s);
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   // Background mesh FECollection, FESpace, and GridFunction
   ParFiniteElementSpace *bg_grad_fes = NULL;
   ParGridFunction *bg_grad = NULL;
   ParFiniteElementSpace *bg_hess_fes = NULL;
   ParGridFunction *bg_hess = NULL;

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   ParFiniteElementSpace *grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;

   if (surface_fit_const > 0.0)
   {
      bg_grad_fes = new ParFiniteElementSpace(pmesh_bg, &bg_fec, dim);
      bg_grad = new ParGridFunction(bg_grad_fes);

      int n_hessian_bg = dim * dim;
      bg_hess_fes = new ParFiniteElementSpace(pmesh_bg, &bg_fec, n_hessian_bg);
      bg_hess = new ParGridFunction(bg_hess_fes);

      // Setup gradient of the background mesh.
      bg_grad->ReorderByNodes();
      for (int d = 0; d < pmesh_bg->Dimension(); d++)
      {
         ParGridFunction bg_grad_comp(&bg_pfes, bg_grad->GetData()+d*bg_domain.Size());
         bg_domain.GetDerivative(1, d, bg_grad_comp);
      }

      // Setup Hessian on background mesh.
      bg_hess->ReorderByNodes();
      int id = 0;
      for (int d = 0; d < dim; d++)
      {
         for (int idir = 0; idir < dim; idir++)
         {
            ParGridFunction bg_grad_comp(&bg_pfes, bg_grad->GetData()+d*bg_domain.Size());
            ParGridFunction bg_hess_comp(&bg_pfes, bg_hess->GetData()+id*bg_domain.Size());
            bg_grad_comp.GetDerivative(1, idir, bg_hess_comp);
            id++;
         }
      }

      // Setup functions on the mesh being optimized.
      grad_fes = new ParFiniteElementSpace(pmesh, &fec, dim);
      surf_fit_grad = new ParGridFunction(grad_fes);

      surf_fit_hess_fes = new ParFiniteElementSpace(pmesh, &fec, dim * dim);
      surf_fit_hess = new ParGridFunction(surf_fit_hess_fes);

      for (int i = 0; i < surf_fit_marker.Size(); i++)
      {
         surf_fit_marker[i] = false;
         surf_fit_mat_gf[i] = 0.0;
      }

      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         pfes_s.GetBdrElementVDofs(i, vdofs);
         for (int j = 0; j < vdofs.Size(); j++)
         {
            surf_fit_marker[vdofs[j]] = true;
            surf_fit_mat_gf(vdofs[j]) = 1.0;
         }
      }

      adapt_surface = new InterpolatorFP;
      adapt_grad_surface = new InterpolatorFP;
      adapt_hess_surface = new InterpolatorFP;

      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_mat_gf,
                                "Boundary DOFs to Fit",
                                900, 600, 300, 300);
      }
   }

   MeshOptimizer mesh_opt;
   mesh_opt.Setup(pfes, metric_id, quad_order);
   mesh_opt.GetIntegrator()->
      EnableSurfaceFittingFromSource(bg_domain, domain,
                                     surf_fit_marker, surf_fit_coeff,
                                     *adapt_surface,
                                     *bg_grad, *surf_fit_grad, *adapt_grad_surface,
                                     *bg_hess, *surf_fit_hess, *adapt_hess_surface);
   mesh_opt.GetSolver()->SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
   mesh_opt.GetSolver()->SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   if (surface_fit_const > 0.0)
   {
      double err_avg, err_max;
      mesh_opt.GetIntegrator()->GetSurfaceFittingErrors(err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Initial Avg fitting error: " << err_avg << std::endl
                   << "Initial Max fitting error: " << err_max << std::endl;
      }
   }
   mesh_opt.OptimizeNodes(x);

   // Save the optimized mesh to files.
   {
      ostringstream mesh_name;
      mesh_name << "amster-out.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);

      VisItDataCollection visit_dc("amster_opt", pmesh);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }

   {
      socketstream vis;
      common::VisualizeMesh(vis, "localhost", 19916, *pmesh,
                            "Final mesh", 800, 0, 400, 400, "me");
   }

   if (surface_fit_const > 0.0)
   {
      double err_avg, err_max;
      mesh_opt.GetIntegrator()->GetSurfaceFittingErrors(err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;
      }
   }

   // Visualize the mesh displacement.
   {
      socketstream vis;
      x0 -= x;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Displacements", 1200, 0, 400, 400, "jRmclA");
   }

   // 20. Free the used memory.
   //   delete metric_coeff1;
   //   delete adapt_lim_eval;
   //   delete adapt_surface;
   delete target_c;
   //   delete adapt_coeff;
   delete metric;
   //   delete untangler_metric;
   delete adapt_hess_surface;
   delete adapt_grad_surface;
   delete adapt_surface;
   delete surf_fit_hess;
   delete surf_fit_hess_fes;
   delete surf_fit_grad;
   delete grad_fes;
   delete bg_hess;
   delete bg_hess_fes;
   delete bg_grad;
   delete bg_grad_fes;
   delete pmesh_bg;
   delete pmesh;

   return 0;
}

void Untangle(ParGridFunction &x, double min_detA, int quad_order)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nUntangle Phase\n***\n"; }

   // The metrics work in terms of det(T).
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(pfes.GetFE(0)->GetGeomType());
   // Slightly below the minimum to avoid division by 0.
   double min_detT = min_detA / Wideal.Det();

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::None;
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2) { metric = new TMOP_Metric_004; }
   else          { metric = new TMOP_Metric_360; }
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 2, 1.5,
                                                    0.001, 0.001,
                                                    btype, wctype);
   TargetConstructor::TargetType target =
         TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor target_c(target, pfes.GetComm());
   auto tmop_integ = new TMOP_Integrator(&u_metric, &target_c, nullptr);
   tmop_integ->EnableFiniteDifferences(x);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);


   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Linear solver.
   MINRESSolver minres(pfes.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres.SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes.GetComm(), ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetOperator(nlf);
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_detT);
   solver.SetMaxIter(200);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   delete metric;

   return;
}

void WorstCaseOptimize(ParGridFunction &x, int quad_order)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nWorst Quality Phase\n***\n"; }

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::None;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_304; }
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 2, 1.5,
                                                    0.001, 0.001,
                                                    btype, wctype);
   TargetConstructor::TargetType target =
         TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor target_c(target, pfes.GetComm());
   auto tmop_integ = new TMOP_Integrator(&u_metric, &target_c, nullptr);
   tmop_integ->EnableFiniteDifferences(x);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);

   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Linear solver.
   MINRESSolver minres(pfes.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres.SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes.GetComm(), ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetOperator(nlf);
   solver.EnableWorstCaseOptimization();
   solver.SetPreconditioner(minres);
   solver.SetMaxIter(1000);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   delete metric;

   return;
}
