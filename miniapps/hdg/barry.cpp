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

    LinearForm f(&pressure_space);
    FunctionCoefficient data(fFun);
    f.AddDomainIntegrator(new DomainLFIntegrator(data));
    f.Assemble();

    BilinearForm a(&velocity_space);
    ConstantCoefficient minus_one(-1.0);
    a.AddDomainIntegrator(new VectorMassIntegrator(minus_one));

    MixedBilinearForm b(&pressure_space, &velocity_space);
    b.AddDomainIntegrator(new VectorDivergenceIntegrator());

    for (int element_index = 0; element_index < mesh.GetNE(); ++element_index)
    {
        cout << "Element_index: " << element_index << endl;
        // Array<int> element_vdofs;
        // velocity_space.GetElementVDofs(element_index, element_vdofs);
        // element_vdofs.Print(out, element_vdofs.Size());
        DenseMatrix A11;
        a.ComputeElementMatrix(element_index, A11);
        A11.PrintMatlab();
        cout << endl;

        DenseMatrix A21;
        b.ComputeElementMatrix(element_index, A21);
        A21.PrintMatlab();
        cout << endl;

        cout << endl;
    }
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
