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

#include "mesh_extras.hpp"

using namespace std;

namespace mfem
{

namespace common
{

ElementMeshStream::ElementMeshStream(Element::Type e)
{
   *this << "MFEM mesh v1.0" << endl;
   switch (e)
   {
      case Element::SEGMENT:
         *this << "dimension" << endl << 1 << endl
               << "elements" << endl << 1 << endl
               << "1 1 0 1" << endl
               << "boundary" << endl << 2 << endl
               << "1 0 0" << endl
               << "1 0 1" << endl
               << "vertices" << endl
               << 2 << endl
               << 1 << endl
               << 0 << endl
               << 1 << endl;
         break;
      case Element::TRIANGLE:
         *this << "dimension" << endl << 2 << endl
               << "elements" << endl << 1 << endl
               << "1 2 0 1 2" << endl
               << "boundary" << endl << 3 << endl
               << "1 1 0 1" << endl
               << "1 1 1 2" << endl
               << "1 1 2 0" << endl
               << "vertices" << endl
               << "3" << endl
               << "2" << endl
               << "0 0" << endl
               << "1 0" << endl
               << "0 1" << endl;
         break;
      case Element::QUADRILATERAL:
         *this << "dimension" << endl << 2 << endl
               << "elements" << endl << 1 << endl
               << "1 3 0 1 2 3" << endl
               << "boundary" << endl << 4 << endl
               << "1 1 0 1" << endl
               << "1 1 1 2" << endl
               << "1 1 2 3" << endl
               << "1 1 3 0" << endl
               << "vertices" << endl
               << "4" << endl
               << "2" << endl
               << "0 0" << endl
               << "1 0" << endl
               << "1 1" << endl
               << "0 1" << endl;
         break;
      case Element::TETRAHEDRON:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 4 0 1 2 3" << endl
               << "boundary" << endl << 4 << endl
               << "1 2 0 2 1" << endl
               << "1 2 1 2 3" << endl
               << "1 2 2 0 3" << endl
               << "1 2 0 1 3" << endl
               << "vertices" << endl
               << "4" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl;
         break;
      case Element::HEXAHEDRON:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 5 0 1 2 3 4 5 6 7" << endl
               << "boundary" << endl << 6 << endl
               << "1 3 0 3 2 1" << endl
               << "1 3 4 5 6 7" << endl
               << "1 3 0 1 5 4" << endl
               << "1 3 1 2 6 5" << endl
               << "1 3 2 3 7 6" << endl
               << "1 3 3 0 4 7" << endl
               << "vertices" << endl
               << "8" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "1 1 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl
               << "1 0 1" << endl
               << "1 1 1" << endl
               << "0 1 1" << endl;
         break;
      case Element::WEDGE:
         *this << "dimension" << endl << 3 << endl
               << "elements" << endl << 1 << endl
               << "1 6 0 1 2 3 4 5" << endl
               << "boundary" << endl << 5 << endl
               << "1 2 2 1 0" << endl
               << "1 2 3 4 5" << endl
               << "1 3 0 1 4 3" << endl
               << "1 3 1 2 5 4" << endl
               << "1 3 2 0 3 5" << endl
               << "vertices" << endl
               << "6" << endl
               << "3" << endl
               << "0 0 0" << endl
               << "1 0 0" << endl
               << "0 1 0" << endl
               << "0 0 1" << endl
               << "1 0 1" << endl
               << "0 1 1" << endl;
         break;
      default:
         mfem_error("Invalid element type!");
         break;
   }

}

double ComputeVolume(const Mesh &mesh, int ir_order)
{
   double vol = 0.0;

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight;

         vol += w;
      }
   }
   return vol;
}

double ComputeVolume(const Mesh &mesh, const Array<int> &attr_marker,
		     int ir_order)
{
   double vol = 0.0;

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      int attr = mesh.GetAttribute(i);
      if (attr_marker[attr-1] == 0) { continue; }
     
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight;

         vol += w;
      }
   }
   return vol;
}

double ComputeSurfaceArea(const Mesh &mesh, int ir_order)
{
   double area = 0.0;

   IsoparametricTransformation T;

   for (int i=0; i<mesh.GetNBE(); i++)
   {
     // ElementTransformation *T = mesh.GetBdrElementTransformation(i);
      const_cast<Mesh&>(mesh).GetBdrElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetBdrElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight;

         area += w;
      }
   }
   return area;
}

double ComputeSurfaceArea(const Mesh &mesh, const Array<int> &bdr_attr_marker,
			  int ir_order)
{
   double area = 0.0;

   IsoparametricTransformation T;

   for (int i=0; i<mesh.GetNBE(); i++)
   {
      int attr = mesh.GetBdrAttribute(i);
      if (bdr_attr_marker[attr-1] == 0) { continue; }
     
      const_cast<Mesh&>(mesh).GetBdrElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetBdrElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight;

         area += w;
      }
   }
   return area;
}

double ComputeZerothMoment(const Mesh &mesh, Coefficient &rho,
                           int ir_order)
{
   double mom = 0.0;

   IsoparametricTransformation T;
      
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight * rho.Eval(T, ip);

         mom += w;
      }
   }

   return mom;
}

void ComputeElementZerothMoments(const Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &m)
{
   MFEM_ASSERT(m.Size() == mesh.GetNE(), "Invalid GridFunction.  "
               "Must have a length equal to the number of mesh elements.");

   FiniteElementSpace * fes = m.FESpace();

   Array<int> vdofs;

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      fes->GetElementVDofs(i, vdofs);

      double mom = 0.0;

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);

         double w = T.Weight() * ip.weight * rho.Eval(T, ip);

         mom += w;
      }

      m[vdofs[0]] = mom;
   }
}

double ComputeFirstMoment(const Mesh &mesh, Coefficient &rho,
                          int ir_order, Vector &mom)
{
   double mom0 = 0.0;

   int sdim = mesh.SpaceDimension();

   mom.SetSize(sdim);
   mom = 0.0;

   double x_data[3];
   Vector x(x_data, sdim);

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);
         T.Transform(ip, x);

         double w = T.Weight() * ip.weight * rho.Eval(T, ip);

         mom0 += w;
         mom.Add(w, x);
      }
   }

   return mom0;
}

double ComputeSecondMoment(const Mesh &mesh, Coefficient &rho,
                           const Vector &center,
                           int ir_order, DenseMatrix &mom)
{
   double mom0 = 0.0;

   int sdim = mesh.SpaceDimension();

   mom.SetSize(sdim);
   mom = 0.0;

   double x_data[3];
   Vector x(x_data, sdim);

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);
         T.Transform(ip, x);
         x.Add(-1.0, center);

         double r2 = x * x;

         double w = T.Weight() * ip.weight * rho.Eval(T, ip);

         mom0 += w;

         for (int k=0; k<sdim; k++)
         {
            mom(k,k) += w * r2;
            for (int l=k; l<sdim; l++)
            {
               mom(k,l) -= w * x[k] * x[l];
            }
         }
      }
   }

   for (int k=0; k<sdim; k++)
   {
      for (int l=0; l<k; l++)
      {
         mom(k,l) = mom(l,k);
      }
   }

   return mom0;
}

void ComputeElementCentersOfMass(const Mesh &mesh, Coefficient &rho,
                                 int ir_order, GridFunction &c)
{
   int sdim = mesh.SpaceDimension();

   MFEM_ASSERT(c.Size() == mesh.GetNE() * sdim,
               "Invalid GridFunction.  Must have a length equal to the "
               "number of mesh elements times the spatial dimension.");

   FiniteElementSpace * fes = c.FESpace();

   Array<int> vdofs;

   double x_data[3];
   Vector x(x_data, sdim);

   c = 0.0;

   IsoparametricTransformation T;
   
   for (int i=0; i<mesh.GetNE(); i++)
   {
      const_cast<Mesh&>(mesh).GetElementTransformation(i, &T);
      Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      fes->GetElementVDofs(i, vdofs);

      double mom0 = 0.0;

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T.SetIntPoint(&ip);
         T.Transform(ip, x);

         double w = T.Weight() * ip.weight * rho.Eval(T, ip);

         mom0 += w;

         for (int k=0; k<sdim; k++)
         {
            c[vdofs[k]] += w * x[k];
         }
      }
      for (int k=0; k<sdim; k++)
      {
         c[vdofs[k]] /= mom0;
      }
   }
}

void
MergeMeshNodes(Mesh * mesh, int logging)
{
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   double h_min, h_max, k_min, k_max;
   mesh->GetCharacteristics(h_min, h_max, k_min, k_max);

   // Set tolerance for merging vertices
   double tol = 1.0e-8 * h_min;

   if ( logging > 0 )
      cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber() :
               ((dim==2)?mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE())) << endl;

   vector<int> v2v(mesh->GetNV());

   Vector vd(sdim);

   for (int i = 0; i < mesh->GetNV(); i++)
   {
      Vector vi(mesh->GetVertex(i), sdim);

      v2v[i] = -1;

      for (int j = 0; j < i; j++)
      {
         Vector vj(mesh->GetVertex(j), sdim);
         add(vi, -1.0, vj, vd);

         if ( vd.Norml2() < tol )
         {
            v2v[i] = j;
            break;
         }
      }
      if ( v2v[i] < 0 ) { v2v[i] = i; }
   }

   // renumber elements
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Element *el = mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   mesh->RemoveUnusedVertices();

   if ( logging > 0 )
   {
      cout << "Euler Number of Final Mesh:    "
           << ((dim==3) ? mesh->EulerNumber() :
               ((dim==2) ? mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE()))
           << endl;
   }
}

void AttrToMarker(int max_attr, const Array<int> &attrs, Array<int> &marker)
{
   MFEM_ASSERT(attrs.Max() <= max_attr, "Invalid attribute number present.");

   marker.SetSize(max_attr);
   if (attrs.Size() == 1 && attrs[0] == -1)
   {
      marker = 1;
   }
   else
   {
      marker = 0;
      for (int j=0; j<attrs.Size(); j++)
      {
         int attr = attrs[j];
         MFEM_VERIFY(attr > 0, "Attribute number less than one!");
         marker[attr-1] = 1;
      }
   }
}

} // namespace common

} // namespace mfem
