#include "mfem.hpp"

using namespace std;
using namespace mfem;

void uFun(const Vector& x, Vector& u);
real_t pFun(const Vector& x);
real_t fFun(const Vector& x);
const real_t pi(M_PI);

int main(int argc, char* argv[])
{
    int order = 1;
    int refine = 0;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Polynomial degree");
    args.AddOption(&refine, "-r", "--refine", "Number of uniform mesh refinements");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Mesh mesh(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE, 1));
    for (int i = 0; i < refine; ++i)
        mesh.UniformRefinement();

    int dim = mesh.Dimension();
    DG_FECollection element_collection(order, dim);
    DG_Interface_FECollection face_collection(order, dim);

    FiniteElementSpace velocity_space(&mesh, &element_collection, dim);
    FiniteElementSpace pressure_space(&mesh, &element_collection);
    FiniteElementSpace auxiliary_space(&mesh, &face_collection);

    Array<int> ess_bdr, ess_dof_marker;
    ess_bdr.SetSize(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    auxiliary_space.GetEssentialVDofs(ess_bdr, ess_dof_marker);

    LinearForm f(&pressure_space);
    FunctionCoefficient data(fFun);
    f.AddDomainIntegrator(new DomainLFIntegrator(data));
    f.Assemble();

    BilinearForm a(&velocity_space);
    ConstantCoefficient minus_one(-1.0);
    a.AddDomainIntegrator(new VectorMassIntegrator(minus_one));

    MixedBilinearForm b(&pressure_space, &velocity_space);
    b.AddDomainIntegrator(new VectorDivergenceIntegrator());

    Table element_to_edge_table(mesh.ElementToEdgeTable());
    const real_t tau(5.0);
    SparseMatrix H(auxiliary_space.GetNDofs());
    H = 0.0;

    DenseMatrix* saved_velocity_matrices = nullptr;
    DenseMatrix* saved_pressure_matrices = nullptr;
    int offset_index = 0;

    for (int element_index = 0; element_index < mesh.GetNE(); ++element_index)
    {
        cout << "Element index: " << element_index << endl;
        DenseMatrix A11;
        a.ComputeElementMatrix(element_index, A11);
        A11.PrintMatlab();
        cout << endl;

        DenseMatrix A21;
        b.ComputeElementMatrix(element_index, A21);
        A21.PrintMatlab();
        cout << endl;

        const FiniteElement* velocity_element = velocity_space.GetFE(element_index);
        const FiniteElement* pressure_element = pressure_space.GetFE(element_index);

        const int num_velocity_dofs(velocity_element->GetDof());
        const int num_pressure_dofs(pressure_element->GetDof());

        Vector velocity_element_shape(num_velocity_dofs);
        Vector pressure_element_shape(num_pressure_dofs);

        DenseMatrix A22(num_pressure_dofs);
        A22 = 0.0;

        Array<int> edge_indices_array;
        element_to_edge_table.GetRow(element_index, edge_indices_array);
        const int num_element_edges(edge_indices_array.Size());

        DenseMatrix* B1 = new DenseMatrix[num_element_edges];
        DenseMatrix* B2 = new DenseMatrix[num_element_edges];
        DenseMatrix* D = new DenseMatrix[num_element_edges];

        if (0 == element_index)
        {
            saved_velocity_matrices = new DenseMatrix[mesh.GetNE()*num_element_edges];
            saved_pressure_matrices = new DenseMatrix[mesh.GetNE()*num_element_edges];
        }

        Array<int> interior_indices, boundary_indices;
        Array<int>* edge_dofs = new Array<int>[num_element_edges];

        for (int local_index = 0; local_index < num_element_edges; ++local_index)
        {
            cout << "Local edge index: " << local_index;
            const int edge_index(edge_indices_array[local_index]);
            FaceElementTransformations* trans(
                mesh.GetFaceElementTransformations(edge_index));
            bool use_element_two(false);
            if (trans->Elem2No == element_index)
                use_element_two = true;

            const FiniteElement* edge = auxiliary_space.GetFaceElement(edge_index);
            const int num_edge_dofs(edge->GetDof());
            B1[local_index].SetSize(dim*num_velocity_dofs, num_edge_dofs);
            B1[local_index] = 0.0;

            B2[local_index].SetSize(num_pressure_dofs, num_edge_dofs);
            B2[local_index] = 0.0;

            D[local_index].SetSize(num_edge_dofs);
            D[local_index] = 0.0;

            const int quad_order = 5;
            const IntegrationRule ir(IntRules.Get(trans->FaceGeom, quad_order));

            for (int point_index = 0; point_index < ir.GetNPoints(); ++point_index)
            {
                const IntegrationPoint ip(ir.IntPoint(point_index));
                trans->SetIntPoint(&ip);

                IntegrationPoint eip;
                if (use_element_two)
                    trans->Loc2.Transform(ip, eip);
                else
                    trans->Loc1.Transform(ip, eip);

                pressure_element->CalcShape(eip, pressure_element_shape);
                velocity_element->CalcShape(eip, velocity_element_shape);

                // accumulate into A22
                real_t weight = ip.weight*trans->Weight()*tau;
                AddMult_a_VVt(weight, pressure_element_shape, A22);

                Vector edge_shape(num_edge_dofs);
                edge->CalcShape(ip, edge_shape);
                AddMult_a_VVt(weight, edge_shape, D[local_index]);

                weight *= -1.0;  // make negative for B2 matrices
                AddMult_a_VWt(weight, pressure_element_shape, edge_shape, B2[local_index]);

                // is the normal vector computation
                // independent of the integration point?
                Vector normal_vector(dim);  // note: non-unit normal
                CalcOrtho(trans->Jacobian(), normal_vector);
                if (!use_element_two)
                    normal_vector *= -1.0;
                normal_vector *= ip.weight;  // multiply by weight here for efficiency

                DenseMatrix mixed_shape_matrix(velocity_element_shape.Size(),
                                               edge_shape.Size());
                MultVWt(velocity_element_shape, edge_shape, mixed_shape_matrix);
                for (int d = 0; d < dim; ++d)
                {
                    const int row_offset = d*num_velocity_dofs;
                    const real_t normal_vector_component = normal_vector(d);
                    for (int i = 0; i < num_velocity_dofs; ++i)
                    {
                        for (int j = 0; j < num_edge_dofs; ++j)
                        {
                            B1[local_index](row_offset + i, j) +=
                                normal_vector_component*mixed_shape_matrix(i, j);
                        }
                    }
                }
            }
            auxiliary_space.GetFaceVDofs(edge_index, edge_dofs[local_index]);
            cout << " has dofs: ";
            edge_dofs[local_index].Print(out, edge_dofs[local_index].Size());
            if (ess_dof_marker[edge_dofs[local_index][0]])
                boundary_indices.Append(local_index);
            else
                interior_indices.Append(local_index);

            B1[local_index].PrintMatlab();
            cout << endl;
            B2[local_index].PrintMatlab();
            cout << endl;
            D[local_index].PrintMatlab();
            cout << endl;
        }
        boundary_indices.Print();
        interior_indices.Print();
        A22.PrintMatlab();
        cout << endl;

        A11.Invert();
        DenseMatrix W1(A21.Height(), A11.Width());
        Mult(A21, A11, W1);
        DenseMatrix W2(W1.Height(), A21.Height());
        MultABt(W1, A21, W2);

        // overwrite A22 with the Schur complement
        A22 -= W2;
        A22.Invert();
        DenseMatrix W3(A22.Height(), W1.Width());

        // overwrite A21 with A22*W1
        Mult(A22, W1, A21);
        DenseMatrix W4(W1.Width(), W3.Width());
        MultAtB(W1, A21, W4);
        A21.Neg();

        // overwrite A11 with the inverse (1, 1) block
        A11 += W4;

        A11.PrintMatlab();
        cout << endl;
        A21.PrintMatlab();
        cout << endl;
        A22.PrintMatlab();
        cout << endl;
        // compute and store A^{-1}B for each edge
        for (int local_index = 0; local_index < num_element_edges; ++local_index)
        {
            H.AddSubMatrix(edge_dofs[local_index], edge_dofs[local_index], D[local_index]);
            const int matrix_index = offset_index + local_index;
            saved_velocity_matrices[matrix_index].SetSize(
                A11.Height(), B1[local_index].Width());
            // A11B1
            Mult(A11, B1[local_index], saved_velocity_matrices[matrix_index]);
            DenseMatrix A12B2(A21.Width(), B2[local_index].Width());
            MultAtB(A21, B2[local_index], A12B2);
            saved_velocity_matrices[matrix_index] += A12B2;
            saved_velocity_matrices[matrix_index].Neg();  // so that we subtract in assembly

            saved_pressure_matrices[matrix_index].SetSize(
                A21.Height(), B1[local_index].Width());
            // A21B1
            Mult(A21, B1[local_index], saved_pressure_matrices[matrix_index]);
            DenseMatrix A22B2(A22.Height(), B2[local_index].Width());
            Mult(A22, B2[local_index], A22B2);
            saved_pressure_matrices[matrix_index] += A22B2;
            saved_pressure_matrices[matrix_index].Neg();  // so that we subtract in assembly
        }

        for (int column_interior_index : interior_indices)
        {
            const int matrix_index = offset_index + column_interior_index;
            for (int row_interior_index : interior_indices)
            {
                DenseMatrix first_matrix(
                    B1[row_interior_index].Width(),
                    saved_velocity_matrices[matrix_index].Width());
                MultAtB(B1[row_interior_index],
                        saved_velocity_matrices[matrix_index],
                        first_matrix);
                DenseMatrix second_matrix(
                    B2[row_interior_index].Width(),
                    saved_pressure_matrices[matrix_index].Width());
                MultAtB(B2[row_interior_index],
                        saved_pressure_matrices[matrix_index],
                        second_matrix);
                first_matrix += second_matrix;
                H.AddSubMatrix(edge_dofs[row_interior_index],
                               edge_dofs[column_interior_index],
                               first_matrix);
            }
        }

        delete[] D;
        delete[] B2;
        delete[] B1;
        delete[] edge_dofs;
        offset_index += num_element_edges;
    }
    H.Print();
    delete[] saved_velocity_matrices;
    delete[] saved_pressure_matrices;
    return 0;
}

void uFun(const Vector& x, Vector& u)
{
    const real_t freq(2.0*pi);
    const real_t freqxi(freq*x(0));
    const real_t freqxj(freq*x(1));

    if (2 == x.Size())
    {
        u(0) = -freq*cos(freqxi)*sin(freqxj) - 1.0; 
        u(1) = -freq*sin(freqxi)*cos(freqxj);
    }
    else
    {
        const real_t freqxk(freq*x(2));
        u(0) = -freq*cos(freqxi)*sin(freqxj)*sin(freqxk) - 1.0; 
        u(1) = -freq*sin(freqxi)*cos(freqxj)*sin(freqxk);
        u(2) = -freq*sin(freqxi)*sin(freqxj)*cos(freqxk);
    }
}

real_t pFun(const Vector& x)
{
    const real_t xi(x(0));
    const real_t xj(x(1));

    const real_t freq(2.0*pi);
    if (2 == x.Size())
        return 1.0 + xi + sin(freq*xi)*sin(freq*xj);
    else
    {
        const real_t xk(x(2));
        return xi + sin(freq*xi)*sin(freq*xj)*sin(freq*xk);
    }
}
real_t fFun(const Vector& x)
{
    const real_t xi(x(0));
    const real_t xj(x(1));

    const real_t freq(2.0*pi);
    if (2 == x.Size())
        return 4.0*freq*pi*sin(freq*xi)*sin(freq*xj);
    else
    {
        const real_t xk(x(2));
        return 6.0*freq*pi*sin(freq*xi)*sin(freq*xj)*sin(freq*xk);
    }
}
