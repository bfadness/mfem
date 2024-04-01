#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "MMAOptimizer.hpp"
#include "MMA.hpp"


double obj0(mfem::Vector& x)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i]*x[i];
    }
    return rez;
}

double dobj0(mfem::Vector& x, mfem::Vector& dx)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i]*x[i];
        dx[i]=2.0*x[i];
    }
    return rez;
}

double g0(mfem::Vector& x)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i];
    }

    rez=rez/n;
    return rez-2.0;
}

double dg0(mfem::Vector& x, mfem::Vector& dx)
{
    const int n=x.Size();
    double rez=0.0;
    for(int i=0;i<n;i++){
        rez=rez+x[i];
        dx[i]=1.0;
    }
    rez=rez/n;
    return rez-1.0;
}

int main(int argc, char *argv[])
{

    // Initialize MPI.
     int nprocs, myrank;
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int num_var=10;
    const char *petscrc_file = "";

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&num_var, "-n", "--nvar", "Number of design variables.");

    mfem::Vector x(num_var);
    mfem::Vector dx(num_var);
    mfem::Vector xmin(num_var); xmin=-1.0;
    mfem::Vector xmax(num_var); xmax=2.0;
    x=xmin; x+=0.5;

    mfem::MMAOpt mma(num_var,1,x);


    {
       args.PrintOptions(std::cout);
    }
    mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

    double a[4]={0.0,0.0,0.0,0.0};
    double c[4]={1000.0,1000.0,1000.0,1000.0};
    double d[4]={0.0,0.0,0.0,0.0};

    mfem::NativeMMA* nmma=new mfem::NativeMMA(MPI_COMM_SELF,1,x,a,c,d);

    mfem::Vector g(1); g=-1.0;
    mfem::Vector dg(num_var); dg=0.0;

    double o;
    for(int it=0;it<30;it++){
        o=dobj0(x,dx);
        g[0]=dg0(x,dg);

        std::cout<<"it="<<it<<" o="<<o<<" g="<<g[0]<<std::endl;

        for(int i=0;i<num_var;i++){
            std::cout<<" "<<x[i];
        }
        std::cout<<std::endl;
        for(int i=0;i<num_var;i++){
            std::cout<<" "<<dx[i];
        }
        std::cout<<std::endl;

        //mma.Update(it,dx,g,dg,xmin,xmax,x);


        nmma->Update(x,dx,g.GetData(),&dg,xmin,xmax);
        std::cout<<std::endl;

    }

    for(int i=0;i<num_var;i++){
        std::cout<<" "<<x[i];
    }
    std::cout<<std::endl;

    o=obj0(x);
    std::cout<<"Final o="<<o<<std::endl;

    delete nmma;

    mfem::MFEMFinalizePetsc();
    MPI_Finalize();
    return 0;



}
