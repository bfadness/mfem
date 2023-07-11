//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "nodepair.hpp"

using namespace std;
using namespace mfem;

int get_rank(int tdof, std::vector<int> & tdof_offsets)
{
   int size = tdof_offsets.size();
   if (size == 1) { return 0; }
   std::vector<int>::iterator up;
   up=std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(),tdof); //
   return std::distance(tdof_offsets.begin(),up)-1;
}

void ComputeTdofOffsets(const ParFiniteElementSpace * pfes,
                        std::vector<int> & tdof_offsets)
{
   MPI_Comm comm = pfes->GetComm();
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   tdof_offsets.resize(num_procs);
   int mytoffset = pfes->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void ComputeTdofOffsets(MPI_Comm comm, int mytoffset, std::vector<int> & tdof_offsets)
{
   int num_procs;
   MPI_Comm_size(comm,&num_procs);
   tdof_offsets.resize(num_procs);
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void PrintElementVertices(Mesh * mesh, int elem,  int printid)
{
   int myid = Mpi::WorldRank();
   Array<int> vertices;
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " <<  "elem: " << elem <<
                ". Vertices = \n" ;
      mesh->GetElementVertices(elem,vertices);
      for (int i = 0; i<vertices.Size(); i++)
      {
         double * coords = mesh->GetVertex(vertices[i]);
         mfem::out << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")"
                   << endl;
      }
      mfem::out << endl;
   }
}

void PrintFaceVertices(Mesh * mesh, int face,  int printid)
{
   int myid = Mpi::WorldRank();
   Array<int> vertices;
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " <<  "face: " << face <<
                ". Vertices = \n" ;
      mesh->GetFaceVertices(face,vertices);
      for (int i = 0; i<vertices.Size(); i++)
      {
         double *coords = mesh->GetVertex(vertices[i]);
         mfem::out << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")"
                   << endl;
      }
      mfem::out << endl;
   }
}

template <class T>
void PrintArray(const Array<T> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      int sz = a.Size();
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (int i = 0; i<sz; i++)
      {
         mfem::out << a[i] << "  ";
      }
      mfem::out << endl;
   }
}

void PrintSet(const std::set<int> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (std::set<int>::iterator it = a.begin(); it!= a.end(); it++)
      {
         mfem::out << *it << "  ";
      }
      mfem::out << endl;
   }
}

void PrintVector(const Vector & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      int sz = a.Size();
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (int i = 0; i<sz; i++)
      {
         mfem::out << a[i] << "  ";
      }
      mfem::out << endl;
   }
}

void PrintSparseMatrix(const SparseMatrix & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      a.PrintMatlab(mfem::out);
   }
   mfem::out << endl;
}

void FindSurfaceToProject(Mesh& mesh, const int elem, int& cbdrface)
{
   Array<int> attr;
   attr.Append(2);
   Array<int> faces;
   Array<int> ori;
   std::vector<Array<int> > facesVertices;
   std::vector<int > faceid;
   mesh.GetElementFaces(elem, faces, ori);
   int face = -1;
   for (int i=0; i<faces.Size(); i++)
   {
      face = faces[i];
      Array<int> faceVert;
      if (!mesh.FaceIsInterior(face)) // if on the boundary
      {
         mesh.GetFaceVertices(face, faceVert);
         faceVert.Sort();
         facesVertices.push_back(faceVert);
         faceid.push_back(face);
      }
   }
   int bdrface = facesVertices.size();

   Array<int> bdryFaces;
   // This shoulnd't need to be rebuilt
   std::vector<Array<int> > bdryVerts;
   for (int b=0; b<mesh.GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh.GetBdrAttribute(b)) >= 0)  // found the contact surface
      {
         bdryFaces.Append(b);
         Array<int> vert;
         mesh.GetBdrElementVertices(b, vert);
         vert.Sort();
         bdryVerts.push_back(vert);
      }
   }

   int bdrvert = bdryVerts.size();
   cbdrface = -1;  // the face number of the contact surface element
   int count_cbdrface = 0;  // the number of matching surfaces, used for checks

   for (int i=0; i<bdrface; i++)
   {
      for (int j=0; j<bdrvert; j++)
      {
         if (facesVertices[i] == bdryVerts[j])
         {
            cbdrface = faceid[i];
            count_cbdrface += 1;
         }
      }
   }
   MFEM_VERIFY(count_cbdrface == 1,"projection surface not found");

};

Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref,
                       int & refFace, int & refNormal, bool & interior)
{

   ElementTransformation *trans = mesh.GetElementTransformation(elem);
   const int dim = mesh.Dimension();
   const int spaceDim = trans->GetSpaceDim();

   MFEM_VERIFY(spaceDim == 3, "");

   Vector n(spaceDim);

   IntegrationPoint ip;
   ip.Set(ref, dim);

   trans->SetIntPoint(&ip);
   //CalcOrtho(trans->Jacobian(), n);  // Works only for face transformations
   const DenseMatrix jac = trans->Jacobian();

   int dimNormal = -1;
   int normalSide = -1;

   const double tol = 1.0e-8;
   for (int i=0; i<dim; ++i)
   {
      const double d0 = std::abs(ref[i]);
      const double d1 = std::abs(ref[i] - 1.0);

      const double d = std::min(d0, d1);
      // TODO: this works only for hexahedral meshes!

      if (d < tol)
      {
         MFEM_VERIFY(dimNormal == -1, "");
         dimNormal = i;

         if (d0 < tol)
         {
            normalSide = 0;
         }
         else
         {
            normalSide = 1;
         }
      }
   }
   // closest point on the boundary
   if (dimNormal < 0 || normalSide < 0) // node is inside the element
   {
      interior = 1;
      Vector n(3);
      n = 0.0;
      return n;
   }

   MFEM_VERIFY(dimNormal >= 0 && normalSide >= 0, "");
   refNormal = dimNormal;

   MFEM_VERIFY(dim == 3, "");

   {
      // Find the reference face
      if (dimNormal == 0)
      {
         refFace = (normalSide == 1) ? 2 : 4;
      }
      else if (dimNormal == 1)
      {
         refFace = (normalSide == 1) ? 3 : 1;
      }
      else
      {
         refFace = (normalSide == 1) ? 5 : 0;
      }
   }

   std::vector<Vector> tang(2);

   int tangDir[2] = {-1, -1};
   {
      int t = 0;
      for (int i=0; i<dim; ++i)
      {
         if (i != dimNormal)
         {
            tangDir[t] = i;
            t++;
         }
      }

      MFEM_VERIFY(t == 2, "");
   }

   for (int i=0; i<2; ++i)
   {
      tang[i].SetSize(3);

      Vector tangRef(3);
      tangRef = 0.0;
      tangRef[tangDir[i]] = 1.0;

      jac.Mult(tangRef, tang[i]);
   }

   Vector c(3);  // Cross product

   c[0] = (tang[0][1] * tang[1][2]) - (tang[0][2] * tang[1][1]);
   c[1] = (tang[0][2] * tang[1][0]) - (tang[0][0] * tang[1][2]);
   c[2] = (tang[0][0] * tang[1][1]) - (tang[0][1] * tang[1][0]);

   c /= c.Norml2();

   Vector nref(3);
   nref = 0.0;
   nref[dimNormal] = 1.0;

   Vector ndir(3);
   jac.Mult(nref, ndir);

   ndir /= ndir.Norml2();

   const double dp = ndir * c;

   // TODO: eliminate c?
   n = c;
   if (dp < 0.0)
   {
      n *= -1.0;
   }
   interior = 0;
   return n;
}

// WARNING: global variable, just for this little example.
std::array<std::array<int, 3>, 8> HEX_VERT =
{
   {  {0,0,0},
      {1,0,0},
      {1,1,0},
      {0,1,0},
      {0,0,1},
      {1,0,1},
      {1,1,1},
      {0,1,1}
   }
};

int GetHexVertex(int cdim, int c, int fa, int fb, Vector & refCrd)
{
   int ref[3];
   ref[cdim] = c;
   ref[cdim == 0 ? 1 : 0] = fa;
   ref[cdim == 2 ? 1 : 2] = fb;

   for (int i=0; i<3; ++i) { refCrd[i] = ref[i]; }

   int refv = -1;

   for (int i=0; i<8; ++i)
   {
      bool match = true;
      for (int j=0; j<3; ++j)
      {
         if (ref[j] != HEX_VERT[i][j]) { match = false; }
      }

      if (match) { refv = i; }
   }

   MFEM_VERIFY(refv >= 0, "");

   return refv;
}

// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, const Vector & xyz, Array<int>& conn,
                      Vector & xyz2, Vector& xi, DenseMatrix & coords)
{
   const int dim = mesh.Dimension();
   const int np = xyz.Size() / dim;

   MFEM_VERIFY(np * dim == xyz.Size(), "");

   mesh.EnsureNodes();

   FindPointsGSLIB finder(MPI_COMM_WORLD);

   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.5);

   const double bb_t = 0.5;
   finder.Setup(mesh, bb_t);

   finder.FindPoints(xyz);

   Array<unsigned int> procs = finder.GetProc();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   Array<unsigned int> codes = finder.GetCode();

   /// Return element number for each point found by FindPoints.
   Array<unsigned int> elems = finder.GetElem();

   /// Return reference coordinates for each point found by FindPoints.
   Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   Vector dist = finder.GetDist();

   MFEM_VERIFY(dist.Size() == np, "");
   MFEM_VERIFY(refcrd.Size() == np * dim, "");
   MFEM_VERIFY(elems.Size() == np, "");
   MFEM_VERIFY(codes.Size() == np, "");

   bool allfound = true;
   for (auto code : codes)
      if (code == 2) { allfound = false; }

   MFEM_VERIFY(allfound, "A point was not found");

   // cout << "Maximum distance of projected points: " << dist.Max() << endl;

   // extract information
   int myid = Mpi::WorldRank();
   GSLIBCommunicator gslcomm(MPI_COMM_WORLD);

   Array<unsigned int> index_recv, elems_recv, proc_recv;
   Vector ref_recv;

   Vector xyz_recv;
   gslcomm.SendData(dim,procs,elems,refcrd,xyz, proc_recv,index_recv,elems_recv,
                    ref_recv, xyz_recv);

   int np_loc = elems_recv.Size();
   Array<int> conn_loc(np_loc*4);
   Vector xi_send(np_loc*(dim-1));
   for (int i=0; i<np_loc; ++i)
   {
      int refFace, refNormal, refNormalSide;
      bool is_interior = -1;

      Vector normal = GetNormalVector(mesh, elems_recv[i],
                                      ref_recv.GetData() + (i*dim),
                                      refFace, refNormal, is_interior);

      // continue;
      int phyFace;
      if (is_interior)
      {
         phyFace = -1; // the id of the face that has the closest point
         FindSurfaceToProject(mesh, elems_recv[i], phyFace); // seems that this works

         Array<int> cbdrVert;
         mesh.GetFaceVertices(phyFace, cbdrVert);
         Vector xs(dim);
         xs[0] = xyz_recv[i + 0*np_loc];
         xs[1] = xyz_recv[i + 1*np_loc];
         xs[2] = xyz_recv[i + 2*np_loc];

         Vector xi_tmp(dim-1);
         // get nodes!

         GridFunction *nodes = mesh.GetNodes();
         DenseMatrix coords(4,3);
         for (int i=0; i<4; i++)
         {
            for (int j=0; j<3; j++)
            {
               coords(i,j) = (*nodes)[cbdrVert[i]*3+j];
            }
         }
         SlaveToMaster(coords, xs, xi_tmp);

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = xi_tmp[j];
         }
         // now get get the projection to the surface
      }
      else
      {
         Vector faceRefCrd(dim-1);
         {
            int fd = 0;
            for (int j=0; j<dim; ++j)
            {
               if (j == refNormal)
               {
                  refNormalSide = (ref_recv[(i*dim) + j] > 0.5);
               }
               else
               {
                  faceRefCrd[fd] = ref_recv[(i*dim) + j];
                  fd++;
               }
            }
            MFEM_VERIFY(fd == dim-1, "");
         }

         for (int j=0; j<dim-1; ++j)
         {
            xi_send[i*(dim-1)+j] = faceRefCrd[j]*2.0 - 1.0;
         }
      }
      // Get the element face
      Array<int> faces;
      Array<int> ori;
      int face;

      if (is_interior)
      {
         face = phyFace;
      }
      else
      {
         mesh.GetElementFaces(elems_recv[i], faces, ori);
         face = faces[refFace];
      }

      Array<int> faceVert;
      mesh.GetFaceVertices(face, faceVert);

      for (int p=0; p<4; p++)
      {
         conn_loc[4*i+p] = faceVert[p];
      }
   }

   if (0) // for debugging
   {
      int sz = xi_send.Size()/2;

      for (int i = 0; i<sz; i++)
      {
         mfem::out << "("<<xi_send[i*(dim-1)]<<","<<xi_send[i*(dim-1)+1]<<"): -> ";
         for (int j = 0; j<4; j++)
         {
            double * vc = mesh.GetVertex(conn_loc[4*i+j]);
            if (j<3)
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<"), ";
            }
            else
            {
               mfem::out << "("<<vc[0]<<","<<vc[1]<<","<<vc[2]<<") \n " << endl;
            }
         }
      }
   }

   int sz = xi_send.Size()/2;
   DenseMatrix coordsm(sz*4, dim);
   for (int i = 0; i<sz; i++)
   {
      for (int j = 0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh.GetVertex(conn_loc[i*4+j])[k];
         }
      }
   }

   // pass global indices for conn_loc
   for (int i = 0; i<conn_loc.Size(); i++)
   {
      conn_loc[i] = gvert[conn_loc[i]];
   }

   // need to send data (xi, conn and coords of xyz) back to the owning processor
   gslcomm.SendData2(dim,proc_recv,xyz_recv,xi_send, conn_loc, coordsm,xyz2,xi,conn,coords);

}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 1. Parse command-line options.
   const char *mesh_file1 = "block1.mesh";
   const char *mesh_file2 = "block2.mesh";

   Array<int> attr;
   Array<int> m_attr;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh1(mesh_file1, 1, 1);
   Mesh mesh2(mesh_file2, 1, 1);

   const int dim = mesh1.Dimension();
   MFEM_VERIFY(dim == mesh2.Dimension(), "");

   // boundary attribute 2 is the potential contact surface of nodes
   attr.Append(2);
   // boundary attribute 2 is the potential contact surface for master surface
   m_attr.Append(2);

   ParMesh pmesh1(MPI_COMM_WORLD, mesh1); mesh1.Clear();
   ParMesh pmesh2(MPI_COMM_WORLD, mesh2); mesh2.Clear();

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream mesh1_sock(vishost, visport);
   mesh1_sock << "parallel " << num_procs << " " << myid << "\n";
   mesh1_sock.precision(8);
   mesh1_sock << "mesh\n" << pmesh1 << flush;

   socketstream mesh2_sock(vishost, visport);
   mesh2_sock << "parallel " << num_procs << " " << myid << "\n";
   mesh2_sock.precision(8);
   mesh2_sock << "mesh\n" << pmesh2 << flush;

   FiniteElementCollection *fec1;
   ParFiniteElementSpace *fespace1;
   fec1 = new H1_FECollection(1, dim);
   fespace1 = new ParFiniteElementSpace(&pmesh1, fec1, dim, Ordering::byVDIM);
   HYPRE_BigInt size1 = fespace1->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns for mesh1: " << size1 << endl;
   }
   pmesh1.SetNodalFESpace(fespace1);


   GridFunction nodes0 =
      *pmesh1.GetNodes(); // undeformed mesh1 nodal grid function
   GridFunction *nodes1 = pmesh1.GetNodes();

   FiniteElementCollection *fec2 = new H1_FECollection(1, dim);
   ParFiniteElementSpace *fespace2 = new ParFiniteElementSpace(&pmesh2, fec2, dim,
                                                               Ordering::byVDIM);
   HYPRE_BigInt size2 = fespace2->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns for mesh2: " << size2 << endl;
   }

   // degrees of freedom of both meshes
   int ndof_1 = fespace1->GetTrueVSize();
   int ndof_2 = fespace2->GetTrueVSize();
   int ndofs = ndof_1 + ndof_2;
   int gndof_1 = fespace1->GlobalTrueVSize();
   int gndof_2 = fespace2->GlobalTrueVSize();
   int gndofs = gndof_1 + gndof_2;
   // number of nodes for each mesh
   // int nnd_1 = pmesh1.GetNV();
   // int nnd_2 = pmesh2.GetNV();

   // find the total number of vertices owned
   ParFiniteElementSpace *vertexfes1 = new ParFiniteElementSpace(&pmesh1, fec1);
   ParFiniteElementSpace *vertexfes2 = new ParFiniteElementSpace(&pmesh2, fec2);
   int gnnd_1 = vertexfes1->GlobalTrueVSize();
   int gnnd_2 = vertexfes2->GlobalTrueVSize();

   int nnd_1 = vertexfes1->GetTrueVSize();
   int nnd_2 = vertexfes2->GetTrueVSize();


   int nnd = nnd_1 + nnd_2;
   int gnnd = gnnd_1 + gnnd_2;

   Array<int> ess_tdof_list1, ess_bdr1(pmesh1.bdr_attributes.Max());
   ess_bdr1 = 0;

   Array<int> ess_tdof_list2, ess_bdr2(pmesh2.bdr_attributes.Max());
   ess_bdr2 = 0;

   // Define the displacement vector x as a finite element grid function
   // corresponding to fespace. GridFunction is a derived class of Vector.
   ParGridFunction x1(fespace1); x1 = 0.0;
   ParGridFunction x2(fespace2); x2 = 0.0;

   // Generate force
   ParLinearForm *b1 = new ParLinearForm(fespace1);
   b1->Assemble();

   ParLinearForm *b2 = new ParLinearForm(fespace2);
   b2->Assemble();

   Vector lambda1(pmesh1.attributes.Max()); lambda1 = 57.6923076923;
   PWConstCoefficient lambda1_func(lambda1);
   Vector mu1(pmesh1.attributes.Max()); mu1 = 38.4615384615;
   PWConstCoefficient mu1_func(mu1);

   ParBilinearForm *a1 = new ParBilinearForm(fespace1);
   a1->AddDomainIntegrator(new ElasticityIntegrator(lambda1_func,mu1_func));
   a1->Assemble();

   Vector lambda2(pmesh2.attributes.Max()); lambda2 = 57.6923076923;
   PWConstCoefficient lambda2_func(lambda2);
   Vector mu2(pmesh2.attributes.Max());  mu2 = 38.4615384615;
   PWConstCoefficient mu2_func(mu2);

   ParBilinearForm *a2 = new ParBilinearForm(fespace2);
   a2->AddDomainIntegrator(new ElasticityIntegrator(lambda2_func,mu2_func));
   a2->Assemble();

   HypreParMatrix A1;
   Vector B1, X1;
   a1->FormLinearSystem(ess_tdof_list1, x1, *b1, A1, X1, B1);

   HypreParMatrix A2;
   Vector B2, X2;
   a2->FormLinearSystem(ess_tdof_list2, x2, *b2, A2, X2, B2);

   // Combine elasticity operator for two meshes into one.
   // Block Matrix
   Array2D<HypreParMatrix *> blkA(2,2);
   blkA(0,0) = &A1;
   blkA(1,1) = &A2;

   HypreParMatrix * K = HypreParMatrixFromBlocks(blkA);

   // Construct node to segment contact constraint.
   attr.Sort();
   // cout << "Boundary attributes for contact surface faces in mesh 2" << endl;
   // for (auto a : attr)  cout << a << endl;

   // unique numbering of vertices;
   Array<int> globalvertices1(pmesh1.GetNV());
   Array<int> globalvertices2(pmesh2.GetNV());

   for (int i = 0; i<pmesh1.GetNV(); i++)
   {
      globalvertices1[i] = i;
   }
   pmesh1.GetGlobalVertexIndices(globalvertices1);

   for (int i = 0; i<pmesh2.GetNV(); i++)
   {
      globalvertices2[i] = i;
   }
   pmesh2.GetGlobalVertexIndices(globalvertices2);


   std::set<int> bdryVerts2;
   std::vector<int> vertex_offsets;
   ComputeTdofOffsets(vertexfes2,vertex_offsets);
   for (int b=0; b<pmesh2.GetNBE(); ++b)
   {
      if (attr.FindSorted(pmesh2.GetBdrAttribute(b)) >= 0)
      {
         Array<int> vert;
         pmesh2.GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            // skip if the processor does not own the vertex
            if (myid != get_rank(globalvertices2[v],vertex_offsets)) { continue; }
            bdryVerts2.insert(v);
         }
      }
   }

   int npoints = bdryVerts2.size();

   Array<int> s_conn(npoints); // connectivity of the second/slave mesh
   Vector xyz(dim * npoints);
   xyz = 0.0;

   // cout << "Boundary vertices for contact surface vertices in mesh 2" << endl;

   // construct the nodal coordinates on mesh2 to be projected, including displacement
   int count = 0;
   for (auto v : bdryVerts2)
   {
      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = pmesh2.GetVertex(v)[i] + x2[v*dim+i];
      }

      // s_conn[count] = v + nnd_1; // dof1 is the master
      // s_conn[count] = globalvertices[v] - vertexfes2->GetMyTDofOffset() + nnd_1; // dof1 is the master
      
      // This is wrong ... need global enumeration of vertex dofs for both meshes
      // MFEM_ABORT("Need to compute a global numbering of vertex dofs for both meshes");
      // s_conn[count] = globalvertices2[v] + nnd_1; // dof1 is the master
      s_conn[count] = globalvertices2[v]; // dof1 is the master
      count++;
   }
   // mfem::out << "nnd1 = " << n
   // PrintArray(s_conn, "s_conn",0);
   // PrintArray(s_conn, "s_conn",1);
   // PrintArray(s_conn, "s_conn",2);
   // mfem::out << "myid, size of g = " << myid << ", " << npoints*dim << endl;
   MFEM_VERIFY(count == npoints, "");

   // gap function
   Vector g(npoints*dim);
   g = -1.0;
   // segment reference coordinates of the closest point
   Vector m_xi(npoints*(dim-1));
   m_xi = -1.0;

   Array<int> m_conn(npoints*4); // only works for linear elements that have 4 vertices!
   DenseMatrix coordsm(npoints*4, dim);

   // adding displacement to mesh1 using a fixed grid function from mesh1
   x1 = 0.0; // x1 order: [xyz xyz... xyz]
   add(nodes0, x1, *nodes1);

   Vector xyz_recv;
   FindPointsInMesh(pmesh1, globalvertices1, xyz, m_conn, xyz_recv, m_xi ,coordsm);

   // decode and print
   if (0) // for debugging
   {
      int sz = m_xi.Size()/2;

      for (int i = 0; i<sz; i++)
      {
         mfem::out << "("<<xyz_recv[i+0*sz]<<","<<xyz_recv[i+1*sz]<<","<<xyz_recv[i+2*sz]<<"): -> ";
         mfem::out << "("<<m_xi[i*(dim-1)]<<","<<m_xi[i*(dim-1)+1]<<"): -> ";
         for (int j = 0; j<4; j++)
         {
            if (j<3)
            {
               mfem::out << "("<<coordsm(i*4+j,0)<<","<<coordsm(i*4+j,1)<<","<<coordsm(i*4+j,
                                                                                       2)<<"), ";
            }
            else
            {
               mfem::out << "("<<coordsm(i*4+j,0)<<","<<coordsm(i*4+j,1)<<","<<coordsm(i*4+j,
                                                                                       2)<<") \n " << endl;
            }
         }
      }


      MPI_Barrier(MPI_COMM_WORLD);
      // debug m_conn
      std::vector<int> vertex_offsets1;
      ComputeTdofOffsets(vertexfes1,vertex_offsets1);

      for (int i = 0; i< pmesh1.GetNV(); i++)
      {
         int gv = globalvertices1[i];
         if (myid != get_rank(gv,vertex_offsets1)) continue;
         double *vcoords = pmesh1.GetVertex(gv-vertexfes1->GetMyTDofOffset()); 
         mfem::out << "vertex: " << gv << " = ("<<vcoords[0] <<","<<vcoords[1]<<","<<vcoords[2]<<")" << endl;
      }
   }


   Vector xs(dim*npoints);
   xs = 0.0;
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<dim; j++)
      {
         xs[i*dim+j] = xyz_recv[i + (j*npoints)];
      }
   }

   SparseMatrix M(gnnd,gndofs);
   std::vector<SparseMatrix> dM(gnnd, SparseMatrix(gndofs,gndofs)); 
   Assemble_Contact(gnnd, npoints, xs, m_xi, coordsm,
                       s_conn, m_conn, g, M, dM);

   // Note: s_conn and m_conn have global (unique) numbering corresponding 
   // to their pmesh

   // --------------------------------------------------------------------
   // Redistribute the M matrix
   // --------------------------------------------------------------------
   std::vector<int> Moffsets;
   int myMtoffset = vertexfes1->GetMyTDofOffset() + vertexfes2->GetMyTDofOffset();

   ComputeTdofOffsets(K->GetComm(), myMtoffset, Moffsets);

   Array<int> M_send_count(num_procs);
   Array<int> M_send_displ(num_procs);
   Array<int> M_recv_count(num_procs);
   Array<int> M_recv_displ(num_procs);

   M_send_count = 0; M_send_displ = 0;
   M_recv_count = 0; M_recv_displ = 0;
   for (int i = 0; i<M.NumRows(); i++)
   {
      int rsize = M.RowSize(i);
      if (rsize == 0) continue;
      int rank = get_rank(i,Moffsets);
      M_send_count[rank] += rsize+2;
   }

   // communicate so that M_recv_count is constructed
   MPI_Alltoall(&M_send_count[0],1,MPI_INT,&M_recv_count[0],1,MPI_INT,K->GetComm());
   for (int k=0; k<num_procs-1; k++)
   {
      M_send_displ[k+1] = M_send_displ[k] + M_send_count[k];
      M_recv_displ[k+1] = M_recv_displ[k] + M_recv_count[k];
   }
   int M_sbuff_size = M_send_count.Sum();
   int M_rbuff_size = M_recv_count.Sum();

   // now allocate space for the send buffer
   Array<double> M_sendvals(M_sbuff_size);  M_sendvals = 0.0;
   Array<int> M_sendcols(M_sbuff_size);  M_sendcols = 0;
   Array<int> M_sendoffs(num_procs); M_sendoffs = 0;
   for (int i = 0; i<M.NumRows(); i++)
   {
      int rsize = M.RowSize(i);
      if (rsize == 0) continue;
      int rank = get_rank(i,Moffsets);
      // mfem::out << "i, rank  = " << i << ", " << rank << endl;
      int j = M_send_displ[rank] + M_sendoffs[rank];
      Array<int> cols;
      Vector vals;
      M.GetRow(i,cols,vals);
      M_sendoffs[rank] += rsize+2;
      M_sendvals[j] = (double)i;
      M_sendvals[j+1] = (double)rsize;
      M_sendcols[j] = i;
      M_sendcols[j+1] = rsize;
      for (int l=0; l<rsize ; l++)
      {
         M_sendvals[j+l+2] = vals[l];
         M_sendcols[j+l+2] = cols[l];
      }
   }

   // communication
   Array<double> M_recvvals(M_rbuff_size);
   Array<int> M_recvcols(M_rbuff_size);

   double * M_sendvals_ptr = nullptr;
   double * M_recvvals_ptr = nullptr;
   int * M_sendcols_ptr = nullptr;
   int * M_recvcols_ptr = nullptr;
   if (M_sbuff_size !=0 ) 
   {
      M_sendvals_ptr = &M_sendvals[0]; 
      M_sendcols_ptr = &M_sendcols[0]; 
   }   
   if (M_rbuff_size !=0 ) 
   {
      M_recvvals_ptr = &M_recvvals[0]; 
      M_recvcols_ptr = &M_recvcols[0]; 
   }

   MPI_Alltoallv(M_sendvals_ptr, M_send_count, M_send_displ, MPI_DOUBLE, M_recvvals_ptr,
                 M_recv_count, M_recv_displ, MPI_DOUBLE, K->GetComm());

   MPI_Alltoallv(M_sendcols_ptr, M_send_count, M_send_displ, MPI_INT, M_recvcols_ptr,
                 M_recv_count, M_recv_displ, MPI_INT, K->GetComm());


   SparseMatrix localM(nnd,K->GetGlobalNumCols());

   int counter = 0;
   while (counter < M_rbuff_size)
   {
      int row = M_recvcols[counter] - myMtoffset;
      int size = M_recvcols[counter+1];
      Vector vals(size);
      Array<int> cols(size);
      for (int i = 0; i<size; i++)
      {
         vals[i] = M_recvvals[counter+2 + i];
         cols[i] = M_recvcols[counter+2 + i];
      }
      localM.AddRow(row,cols,vals);
      counter += size+2; 
   }
   MFEM_VERIFY(counter == M_rbuff_size, "inconsistent size");
   localM.Finalize();
   localM.SortColumnIndices();
   // --------------------------------------------------------------------


   // --------------------------------------------------------------------
   // Redistribute the dM_i matrices
   // --------------------------------------------------------------------
   std::vector<int> Koffsets;
   ComputeTdofOffsets(K->GetComm(), K->RowPart()[0], Koffsets);

   Array<int> dM_send_count(num_procs);
   Array<int> dM_send_displ(num_procs);
   Array<int> dM_recv_count(num_procs);
   Array<int> dM_recv_displ(num_procs);
   dM_send_count = 0; dM_send_displ = 0;
   dM_recv_count = 0; dM_recv_displ = 0;
   // loop through the dM matrices
   for (int k = 0; k<dM.size(); k++)
   {
      if (dM[k].NumNonZeroElems() == 0) continue;
      int nrows = dM[k].NumRows();
      for (int i = 0; i<nrows; i++)
      {
         int rsize = dM[k].RowSize(i);
         if (rsize == 0) continue;
         int rank = get_rank(i,Koffsets);
         dM_send_count[rank] += rsize+2;
      }
   }
   // comunicate so that dM_recv_count is constructed
   MPI_Alltoall(&dM_send_count[0],1,MPI_INT,&dM_recv_count[0],1,MPI_INT,K->GetComm());
   for (int k=0; k<num_procs-1; k++)
   {
      dM_send_displ[k+1] = dM_send_displ[k] + dM_send_count[k];
      dM_recv_displ[k+1] = dM_recv_displ[k] + dM_recv_count[k];
   }
   int dM_sbuff_size = dM_send_count.Sum();
   int dM_rbuff_size = dM_recv_count.Sum();

   // now allocate space for the send buffer
   Array<double> dM_sendvals(dM_sbuff_size);  dM_sendvals = 0.0;
   Array<int> dM_sendcols(dM_sbuff_size);  dM_sendcols = 0;
   Array<int> dM_sendoffs(num_procs); dM_sendoffs = 0;
   for (int k = 0; k<dM.size(); k++)
   {
      if (dM[k].NumNonZeroElems() == 0) continue;
      int nrows = dM[k].NumRows();
      for (int i = 0; i<nrows; i++)
      {
         int rsize = dM[k].RowSize(i);
         if (rsize == 0) continue;
         int rank = get_rank(i,Koffsets);
         int j = dM_send_displ[rank] + dM_sendoffs[rank];
         Array<int> cols;
         Vector vals;
         dM[k].GetRow(i,cols,vals);
         dM_sendoffs[rank] += rsize+2;
         dM_sendvals[j] = (double)i;
         dM_sendvals[j+1] = (double)rsize;
         dM_sendcols[j] = i;
         dM_sendcols[j+1] = rsize;
         for (int l=0; l<rsize ; l++)
         {
            dM_sendvals[j+l+2] = vals[l];
            dM_sendcols[j+l+2] = cols[l];
         }
      }
   }

   // communication
   Array<double> dM_recvvals(dM_rbuff_size);
   Array<int> dM_recvcols(dM_rbuff_size);
   double * dM_sendvals_ptr = nullptr;
   double * dM_recvvals_ptr = nullptr;
   int * dM_sendcols_ptr = nullptr;
   int * dM_recvcols_ptr = nullptr;
   if (dM_sbuff_size !=0 ) 
   {
      dM_sendvals_ptr = &dM_sendvals[0]; 
      dM_sendcols_ptr = &dM_sendcols[0]; 
   }   
   if (dM_rbuff_size !=0 ) 
   {
      dM_recvvals_ptr = &dM_recvvals[0]; 
      dM_recvcols_ptr = &dM_recvcols[0]; 
   }

   MPI_Alltoallv(dM_sendvals_ptr, dM_send_count, dM_send_displ, MPI_DOUBLE, dM_recvvals_ptr,
                 dM_recv_count, dM_recv_displ, MPI_DOUBLE, K->GetComm());

   MPI_Alltoallv(dM_sendcols_ptr, dM_send_count, dM_send_displ, MPI_INT, dM_recvcols_ptr,
                 dM_recv_count, dM_recv_displ, MPI_INT, K->GetComm());

   SparseMatrix localDM(K->NumRows(),K->GetGlobalNumCols());

   counter = 0;
   while (counter < dM_rbuff_size)
   {
      int row = dM_recvcols[counter] - K->GetRowStarts()[0];
      int size = dM_recvcols[counter+1];
      Vector vals(size);
      Array<int> cols(size);
      for (int i = 0; i<size; i++)
      {
         vals[i] = dM_recvvals[counter+2 + i];
         cols[i] = dM_recvcols[counter+2 + i];
      }
      localDM.AddRow(row,cols,vals);
      counter += size+2; 
   }
   localDM.Finalize();
   localDM.SortColumnIndices();
   MFEM_VERIFY(counter == dM_rbuff_size, "inconsistent size");
   // --------------------------------------------------------------------

   // Assume this is true
   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(), "Hypre_AssumedPartitionCheck is False");


   // Construct M and DM row and col starts to construct HypreParMatrix
   int Mrows[2]; Mrows[0] = myMtoffset; Mrows[1] = myMtoffset+nnd;
   int Mcols[2]; Mcols[0] = K->ColPart()[0]; Mcols[1] = K->ColPart()[1]; 

   int DMrows[2]; DMrows[0] = K->RowPart()[0]; DMrows[1] = K->RowPart()[1];
   int DMcols[2]; DMcols[0] = K->ColPart()[0]; DMcols[1] = K->ColPart()[1]; 

   HypreParMatrix hypreM(K->GetComm(),nnd,gnnd,gndofs,
                         localM.GetI(), localM.GetJ(),localM.GetData(),
                         Mrows,Mcols);

   HypreParMatrix hypreDM(K->GetComm(),ndofs,gndofs,gndofs,
                          localDM.GetI(), localDM.GetJ(),localDM.GetData(),
                          DMrows,DMcols);                      

   hypreM.Print("m.dat");


   // --------------------------------------------------------------------
   // std::set<int> dirbdryv2;
   // for (int b=0; b<mesh2.GetNBE(); ++b)
   // {
   //    if (mesh2.GetBdrAttribute(b) == 1)
   //    {
   //       Array<int> vert;
   //       mesh2.GetBdrElementVertices(b, vert);
   //       for (auto v : vert)
   //       {
   //          dirbdryv2.insert(v);
   //       }
   //    }
   // }
   // std::set<int> dirbdryv1;
   // for (int b=0; b<mesh1.GetNBE(); ++b)
   // {
   //    if (mesh1.GetBdrAttribute(b) == 1)
   //    {
   //       Array<int> vert;
   //       mesh1.GetBdrElementVertices(b, vert);
   //       for (auto v : vert)
   //       {
   //          dirbdryv1.insert(v);
   //       }
   //    }
   // }

   // Array<int> Dirichlet_dof;
   // Array<double> Dirichlet_val;

   // for (auto v : dirbdryv2)
   // {
   //    for (int i=0; i<dim; ++i)
   //    {
   //       Dirichlet_dof.Append(v*dim + i + ndof_1);
   //       Dirichlet_val.Append(0.);
   //    }
   // }
   // double delta = 0.1;
   // for (auto v : dirbdryv1)
   // {
   //    Dirichlet_dof.Append(v*dim + 0);
   //    Dirichlet_val.Append(delta);
   //    Dirichlet_dof.Append(v*dim + 1);
   //    Dirichlet_val.Append(0.);
   //    Dirichlet_dof.Append(v*dim + 2);
   //    Dirichlet_val.Append(0.);
   // }
   return 0;
}
