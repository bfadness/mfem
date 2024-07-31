#include "mfem.hpp"

using namespace std;
using namespace mfem;

void uFun(const Vector& x, Vector& u);
real_t pFun(const Vector& x);
real_t fFun(const Vector& x);
const int pi(M_PI);

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

    Mesh *mesh = new Mesh(Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE, 1));
    int dim = mesh->Dimension();

    return 0;
}

void uFun(const Vector& x, Vector& u)
{
    real_t freq(2.0*pi);
    real_t freqxi(freq*x(0));
    real_t freqxj(freq*x(1));

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

real_t pFun(const Vector& x);
{
    real_t xi(x(0));
    real_t xj(x(1));

    real_t freq(2.0*pi);
    if (2 == x.Size())
        return 1.0 + xi + sin(freq*xi)*sin(freq*xj);
    else
    {
        xk = x(2);
        return xi + sin(freq*xi)*sin(freq*xj)*sin(freq*xk);
    }
}
real_t fFun(const Vector& x);
{
    real_t xi(x(0));
    real_t xj(x(1));

    real_t freq(2.0*pi);
    if (2 == x.Size())
        return 4.0*freq*pi*sin(freq*xi)*sin(freq*xj);
    else
    {
        xk = x(2);
        return 6.0*freq*pi*sin(freq*xi)*sin(freq*xj)*sin(freq*xk);
    }

}
