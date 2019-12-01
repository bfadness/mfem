#include <fstream>
#include <iostream>
#include "lib/hypsys.hpp"

int main(int argc, char *argv[])
{
   int Problem = 1;
   int ConfigNum = 1;
   const char *MeshFile = "data/inline-quad.mesh";
   int ref = 2;
   int order = 3;
   double tFinal = 1.;
   double dt = 0.002;
   int odeSolver = 3;
   int VisSteps = 20;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&Problem, "-p", "--problem",
                  "Hyperbolic system of equations to solve.");
   args.AddOption(&ConfigNum, "-c", "--configuration",
                  "Problem setup to use.");
   args.AddOption(&MeshFile, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&tFinal, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&odeSolver, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&VisSteps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return -1;
   }

   Mesh *mesh;

   return 0;
}

