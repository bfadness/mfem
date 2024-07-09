#include "MMAOptimizer.hpp"

#include <fstream>
#include <math.h>

extern "C" void dgesv_(int* nLAP, int* nrhs, double* AA, int* lda, int* ipiv,
                       double* bb, int* ldb, int* info);

namespace optmma{



void MMA::MMASubSerial::AllocSubData(int nVar, int nCon)
{
    epsi = 1.0;
    ittt = itto = itera = 0;
    raa0 = 0.00001;
    move = 0.5;
    albefa = 0.1;
    xmamieps = 1e-5;
    ux1 = new double[nVar]; //ini
    xl1 = new double[nVar]; //ini
    plam = new double[nVar]; //ini
    qlam = new double[nVar]; //ini
    gvec = new double[nCon]; //ini
    residu = new double[3 * nVar + 4 * nCon + 2]; //ini
    GG = new double[nVar * nCon]; //ini
    delx = new double[nVar]; //init
    dely = new double[nCon]; //ini
    dellam = new double[nCon]; //ini
    dellamyi = new double[nCon];
    diagx = new double[nVar];//ini
    diagy = new double[nCon];//ini
    diaglamyi = new double[nCon]; //ini
    bb = new double[nVar + 1];
    bb1 = new double[nCon + 1];//ini
    Alam = new double[nCon * nCon];//ini
    AA = new double[(nVar + 1) * (nVar + 1)];
    AA1 = new double[(nCon + 1) * (nCon + 1)];//ini
    dlam = new double[nCon]; //ini
    dx = new double[nVar]; //ini
    dy = new double[nCon]; //ini
    dxsi = new double[nVar]; //ini
    deta = new double[nVar]; //ini
    dmu = new double[nCon]; //ini
    Axx = new double[nVar * nCon]; //ini
    axz = new double[nVar]; //ini
    ds = new double[nCon]; //ini
    xx = new double[4 * nCon + 2 * nVar + 2]; //ini
    dxx = new double[4 * nCon + 2 * nVar + 2]; //ini
    stepxx = new double[4 * nCon + 2 * nVar + 2]; //ini
    sum = 0;
    sum1 = new double[nVar];
    stepalfa = new double[nVar]; //ini
    stepbeta = new double[nVar]; //ini
    xold = new double[nVar]; //ini
    yold = new double[nCon]; //ini
    lamold = new double[nCon];//ini
    xsiold = new double[nVar];//ini
    etaold = new double[nVar];//ini
    muold = new double[nCon]; //ini
    sold = new double[nCon]; //ini
    q0 = new double[nVar]; //ini
    p0 = new double[nVar]; //ini
    P = new double[nCon * nVar]; //ini
    Q = new double[nCon * nVar]; //ini
    alfa = new double[nVar]; //ini
    beta = new double[nVar]; //ini
    xmami = new double[nVar];
    b = new double[nCon]; //ini


    for(int i=0; i<(3 * nVar + 4 * nCon + 2);i++){
        residu[i]=0.0;
    }
}

void MMA::MMASubSerial::FreeSubData()
{
    delete[] sum1;
    delete[] ux1;
    delete[] xl1;
    delete[] plam;
    delete[] qlam;
    delete[] gvec;
    delete[] residu;
    delete[] GG;
    delete[] delx;
    delete[] dely;
    delete[] dellam;
    delete[] dellamyi;
    delete[] diagx;
    delete[] diagy;
    delete[] diaglamyi;
    delete[] bb;
    delete[] bb1;
    delete[] Alam;
    delete[] AA;
    delete[] AA1;
    delete[] dlam;
    delete[] dx;
    delete[] dy;
    delete[] dxsi;
    delete[] deta;
    delete[] dmu;
    delete[] Axx;
    delete[] axz;
    delete[] ds;
    delete[] xx;
    delete[] dxx;
    delete[] stepxx;
    delete[] stepalfa;
    delete[] stepbeta;
    delete[] xold;
    delete[] yold;
    delete[] lamold;
    delete[] xsiold;
    delete[] etaold;
    delete[] muold;
    delete[] sold;
    delete[] xmami;
    delete[] q0;
    delete[] p0;
    delete[] P;
    delete[] Q;
    delete[] alfa;
    delete[] beta;
    delete[] b;
}


/// CHECK y and x - do we need them? Tim?
double MMA::MMASubSerial::KKTNorm(double* y,
                                  const double* dfdx,
                                  const double* gx,
                                  const double* dgdx,
                                  const double* xmin,
                                  const double* xmax,
                                  double* x)
{
    int nVar = mma_ptr->nVar;
    int nCon = mma_ptr->nCon;

    for (int i = 0; i < nVar; i++)
    {
       sum1[i] = 0.0;
       for (int j = 0; j < nCon; j++)
       {
          sum1[i] += dgdx[j * nVar + i] * mma_ptr->lam[j];
       }
    }

    for (int i = 0; i < nVar; i++)
    {
       residu[i] = dfdx[i] + sum1[i] - mma_ptr->xsi[i] + mma_ptr->eta[i]; //rex
       residu[nVar + nCon + 1 + nCon + i] = mma_ptr->xsi[i] *
                                    (mma_ptr->x[i] - xmin[i]);   //rexsi
       residu[nVar + nCon + 1 + nCon + nVar + i] = mma_ptr->eta[i] *
                                    (xmax[i] - mma_ptr->x[i]);   //reeta
    }

    residu[nVar + nCon] = mma_ptr->a0 - mma_ptr->zet; //rez
    for (int i = 0; i < nCon; i++)
    {
       residu[nVar + i] = mma_ptr->c[i] +
                            mma_ptr->d[i] * mma_ptr->y[i] -
                            mma_ptr->mu[i] - mma_ptr->lam[i]; //rey
       residu[nVar + nCon + 1 + i] = gx[i] -
                            mma_ptr->a[i] * mma_ptr->z -
                            mma_ptr->y[i]  + mma_ptr->s[i]; //relam
       residu[nVar + nCon + 1 + nCon + 2 * nVar + i] = mma_ptr->mu[i] *
                                                       mma_ptr->y[i]; //remu
       residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon + 1 + i] = mma_ptr->lam[i] *
                                                               mma_ptr->s[i]; //res
       residu[nVar + nCon] -= mma_ptr->a[i] * mma_ptr->lam[i]; //rez
    }
    residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon] = mma_ptr->zet *
                                                    mma_ptr->z; //rezet

    //Get vector product and maximum absolute value
    residunorm = 0.0;
    residumax = 0.0;
    for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
    {
       residunorm += residu[i] * residu[i];
       residumax = std::max(residumax, std::abs(residu[i]));
    }
    // Norm of the residual
    residunorm = std::sqrt(residunorm);
    mma_ptr->kktnorm = residunorm;

    return mma_ptr->kktnorm;
}

void MMA::MMASubSerial::Update(const double* dfdx,
                               const double* gx,
                               const double* dgdx,
                               const double* xmin,
                               const double* xmax,
                               const double* xval)
{
    MMA* mma = this->mma_ptr;
    int nCon = mma->nCon;
    int nVar = mma->nVar;

    ittt = 0;
    itto = 0;
    epsi = 1.0;
    itera = 0;
    mma->z = 1.0;
    mma->zet = 1.0;

    for (int i = 0; i < nCon; i++)
    {
       b[i] = 0.0;
    }

    for (int i = 0; i < nVar; i++)
    {
       // Calculation of bounds alfa and beta according to:
       // alfa = max{xmin, low + 0.1(xval-low), xval-0.5(xmax-xmin)}
       // beta = min{xmax, upp - 0.1(upp-xval), xval+0.5(xmax-xmin)}

       alfa[i] = std::max(std::max(mma->low[i] + albefa * (xval[i] - mma->low[i]),
                                   xval[i] - move * (xmax[i] - xmin[i])), xmin[i]);
       beta[i] = std::min(std::min(mma->upp[i] - albefa * (mma->upp[i] - xval[i]),
                                   xval[i] + move * (xmax[i] - xmin[i])), xmax[i]);
       xmami[i] = std::max(xmax[i] - xmin[i], xmamieps);

       // Calculations of p0, q0, P, Q, and b
       ux1[i] = mma->upp[i] - xval[i];
       if(std::fabs(ux1[i]) <= mma->machineEpsilon)
       {
          ux1[i] = mma->machineEpsilon;
       }
       xl1[i] = xval[i] - mma->low[i];
       if (std::fabs(xl1[i]) <= mma->machineEpsilon)
       {
          xl1[i] = mma->machineEpsilon;
       }
       p0[i] = ( std::max(dfdx[i], 0.0) + 0.001 * (std::max(dfdx[i], 0.0) + std::max(-dfdx[i], 0.0)) + raa0 / xmami[i]) * ux1[i] * ux1[i];
       q0[i] = ( std::max(-dfdx[i], 0.0) + 0.001 * (std::max(dfdx[i], 0.0) + std::max(-dfdx[i], 0.0)) + raa0 / xmami[i]) * xl1[i] * xl1[i];
    }

    // P = max(dgdx,0)
    // Q = max(-dgdx,0)
    // P = P + 0.001(P+Q) + raa0/xmami
    // Q = Q + 0.001(P+Q) + raa0/xmami
    for (int i = 0; i < nCon; i++)
    {
       for (int j = 0; j < nVar; j++)
       {
          // P = P * spdiags(ux2,0,n,n)
          // Q = Q * spdiags(xl2,0,n,n)
          P[i * nVar + j] = (std::max(dgdx[i * nVar + j], 0.0) + 0.001 * (std::max(dgdx[i * nVar + j], 0.0) + std::max(-1*dgdx[i * nVar + j], 0.0)) + raa0 / xmami[j]) * ux1[j] * ux1[j];
          Q[i * nVar + j] = (std::max(-1*dgdx[i * nVar + j], 0.0) + 0.001 * (std::max(dgdx[i * nVar + j], 0.0) + std::max(-1*dgdx[i * nVar + j], 0.0)) + raa0 / xmami[j]) * xl1[j] * xl1[j];
          // b = P/ux1 + Q/xl1 - gx
          b[i] = b[i] + P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
       }
       b[i] = b[i] - gx[i];
    }

    for (int i = 0; i < nVar; i++)
    {
       mma->x[i] = 0.5 * (alfa[i] + beta[i]);
       mma->xsi[i] = 1.0/(mma->x[i] - alfa[i]);
       mma->xsi[i] = std::max(mma->xsi[i], 1.0);
       mma->eta[i] = 1.0/(beta[i] - mma->x[i]);
       mma->eta[i] = std::max(mma->eta[i], 1.0);
       ux1[i] = 0.0;
       xl1[i] = 0.0;
    }

    for (int i = 0; i < nCon; i++)
    {
       mma->y[i] = 1.0;
       mma->lam[i] = 1.0;
       mma->mu[i] = std::max(1.0, 0.5 * mma->c[i]);
       mma->s[i] = 1.0;
    }

    while (epsi > mma->epsimin)
    {
       residu[nVar + nCon] = mma->a0 - mma->zet; //rez
       for (int i = 0; i < nVar; i++)
       {
          ux1[i] = mma->upp[i] - mma->x[i];
          if(std::fabs(ux1[i]) < mma->machineEpsilon){
              ux1[i] = mma->machineEpsilon;
          }


          xl1[i] = mma->x[i] - mma->low[i];
          if(std::fabs(xl1[i]) < mma->machineEpsilon)
          {
              xl1[i] = mma->machineEpsilon;
          }
          // plam = P' * lam, qlam = Q' * lam
          plam[i] = p0[i];
          qlam[i] = q0[i];
          for (int j = 0; j < nCon; j++)
          {
             plam[i] += P[j * nVar + i] * mma->lam[j];
             qlam[i] += Q[j * nVar + i] * mma->lam[j];
             residu[nVar + nCon] -= mma->a[j] * mma->lam[j]; //rez
          }
          residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                      mma->xsi[i] + mma->eta[i]; //rex
          //residu[nVar + nCon] -= mma->a[i] * mma->lam[i]; //rez
          residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * (mma->x[i] - alfa[i]) -
                                               epsi; //rexsi
          if (std::fabs(mma->x[i]-alfa[i]) < mma->machineEpsilon)
          {
             residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * mma->machineEpsilon - epsi;
          }
          residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] *
                                                      (beta[i] - mma->x[i]) - epsi; //reeta
          if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
          {
             residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] * mma->machineEpsilon -
                                                         epsi;;
          }
       }
       for (int i = 0; i < nCon; i++)
       {
          gvec[i] = 0.0;
          // gvec = P/ux + Q/xl
          for (int j = 0; j < nVar; j++)
          {
             gvec[i] += P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
          }
          residu[nVar + i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->mu[i] -
                             mma->lam[i]; //rey
          residu[nVar + nCon + 1 + i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] +
                                        mma->s[i] - b[i]; //relam
          residu[nVar + nCon + 1 + nCon + 2 * nVar + i] = mma->mu[i] * mma->y[i] -
                                                          epsi; //remu
          residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon + 1 + i] = mma->lam[i] * mma->s[i]
                                                                  - epsi; //res
       }
       residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon] = mma->zet * mma->z - epsi;

       //Get vector product and maximum absolute value
       residunorm = 0.0;
       residumax = 0.0;
       for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
       {
          residunorm += residu[i] * residu[i];
          residumax = std::max(residumax, std::abs(residu[i]));
       }
       // Norm of the residual
       residunorm = std::sqrt(residunorm);

       ittt = 0;

       while (residumax > 0.9 * epsi && ittt < 200)
       {
          ittt++;
          for (int i = 0; i < nVar; i++)
          {
             ux1[i] = mma->upp[i] - mma->x[i];
             if(std::fabs(ux1[i]) < mma->machineEpsilon)
             {
                 ux1[i] = mma->machineEpsilon;
             }

             xl1[i] = mma->x[i] - mma->low[i];
             if(std::fabs(xl1[i]) <= mma->machineEpsilon)
             {
                 xl1[i] = mma->machineEpsilon;
             }
             // plam = P' * lam, qlam = Q' * lam
             plam[i] = p0[i];
             qlam[i] = q0[i];
             for (int j = 0; j < nCon; j++)
             {
                plam[i] += P[j * nVar + i] * mma->lam[j];
                qlam[i] += Q[j * nVar + i] * mma->lam[j];
             }
             // NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]);
                   diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                   (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / mma->machineEpsilon + mma->eta[i] /
                              mma->machineEpsilon;
                }
                else
                {
                   delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                             mma->machineEpsilon + epsi / (beta[i] - mma->x[i]);
                   diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                   (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                              mma->eta[i] / (beta[i] - mma->x[i]);
                }
             }
             else if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                          (mma->x[i] - alfa[i]) + epsi / mma->machineEpsilon;
                diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                           mma->eta[i] / mma->machineEpsilon;
             }
             else
             {
                delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                          (mma->x[i] - alfa[i]) + epsi / (beta[i] - mma->x[i]);
                diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                           mma->eta[i] / (beta[i] - mma->x[i]);
             }
          }

          delz = mma->a0 - epsi / mma->z;
          for (int i = 0; i < nCon; i++)
          {
             gvec[i] = 0.0;
             // gvec = P/ux + Q/xl
             for (int j = 0; j < nVar; j++)
             {
                gvec[i] += P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
                GG[i * nVar + j] = P[i * nVar + j] / (ux1[j] * ux1[j]) - Q[i * nVar + j] / (xl1[j] * xl1[j]);
             }

             dely[i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->lam[i] - epsi / mma->y[i];
             delz -= mma->a[i] * mma->lam[i];
             dellam[i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] - b[i] + epsi /
                         mma->lam[i];
             diagy[i] = mma->d[i] + mma->mu[i] / mma->y[i];
             diaglamyi[i] = mma->s[i] / mma->lam[i] + 1.0 / diagy[i];
          }

          if (nCon < nVar)
          {

             // bb1 = dellam + dely./diagy - GG*(delx./diagx);
             // bb1 = [bb1; delz];
             for (int j = 0; j < nCon; j++)
             {
                sum = 0.0;
                for (int i = 0; i < nVar; i++)
                {
                   sum += GG[j * nVar + i] * (delx[i] / diagx[i]);
                }
                bb1[j] = dellam[j] + dely[j] / diagy[j] - sum;
             }
             bb1[nCon] = delz;

             // Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
             for (int i = 0; i < nCon; i++)
             {
                // Axx = GG*spdiags(diagxinv,0,n,n);
                for (int k = 0; k < nVar; k++)
                {
                   Axx[i * nVar + k] = GG[k * nCon + i] / diagx[k];
                }
             }
             // Alam = spdiags(diaglamyi,0,m,m) + Axx*GG';
             for (int i = 0; i < nCon; i++)
             {
                for (int j = 0; j < nCon; j++)
                {
                   Alam[i * nCon + j] = 0.0;
                   for (int k = 0; k < nVar; k++)
                   {
                      Alam[i * nCon + j] += Axx[i * nVar + k] * GG[j * nVar + k];
                   }
                   if (i == j)
                   {
                      Alam[i * nCon + j] += diaglamyi[i];
                   }
                }
             }
             // AA1 = [Alam     a
             //       a'    -zet/z];
             for (int i = 0; i < nCon; i++)
             {
                for (int j = 0; j < nCon; j++)
                {
                   AA1[i * (nCon + 1) + j] = Alam[i * nCon + j];
                }
                AA1[i * (nCon + 1) + nCon] = mma->a[i];
             }
             for (int i = 0; i < nCon; i++)
             {
                AA1[nCon * (nCon + 1) + i] = mma->a[i];
             }
             AA1[(nCon + 1) * (nCon + 1) - 1] = -mma->zet / mma->z;


             // ----------------------------------------------------------------------------
             //bb1 = AA1\bb1 --> solve linear system of equations using LAPACK
             int info;
             int nLAP = nCon + 1;
             int nrhs = 1;
             int lda = nLAP;
             int ldb = nLAP;
             int* ipiv = new int[nLAP];
             dgesv_(&nLAP, &nrhs, AA1, &lda, ipiv, bb1, &ldb, &info);
             if (info == 0) {
                delete[] ipiv;
                for (int i = 0; i < nCon; i++)
                {
                   dlam[i] = bb1[i];
                }
                dz = bb1[nCon];
             } else if (info > 0) {
                std::cerr << "Error: matrix is singular." << std::endl;
             } else {
                std::cerr << "Error: Argument " << info << " has illegal value." << std::endl;
             }
             // ----------------------------------------------------------------------------
             //dx = -(GG'*dlam)./diagx - delx./diagx;
             for (int i = 0; i < nVar; i++)
             {
                sum = 0.0;
                for (int j = 0; j < nCon; j++)
                {
                   sum += GG[j * nVar + i] * dlam[j];
                }
                dx[i] = -sum / diagx[i] - delx[i] / diagx[i];
             }
          }
          else
          {
             azz = mma->zet / mma->z;
             for (int i = 0; i < nCon; i++)
             {
                dellamyi[i] = dellam[i] + dely[i] / diagy[i];
                // azz = zet/z + a'*(a./diaglamyi)
                azz += mma->a[i] * (mma->a[i] / diaglamyi[i]);
             }

             // Axx = spdiags(diagx,0,nVar,nVar) + GG'*spdiags(diaglamyiinv,0,nCon,nCon)*GG;
             // AA = [Axx      axz
             //       axz'     azz];
             for (int i = 0; i < nVar; i++)
             {
                // Axx =  GG'*spdiags(diaglamyiinv,0,nCon,nCon);
                for (int k = 0; k < nCon; k++)
                {
                   Axx[i * nCon + k] = GG[k * nVar + i] / diaglamyi[k];
                }
                axz[i] = 0.0;
                // axz = -GG'*(a./diaglamyi)
                for (int j = 0; j < nCon; j++)
                {
                   axz[i] -= GG[j * nVar + i] * (mma->a[j] / diaglamyi[j]);
                }
             }
             //Assemble matrix AA
             for (int i = 0; i < (nVar + 1); i++)
             {
                for (int j = 0; j < (nVar + 1); j++)
                {
                   // AA = [Axx  .
                   //       .    .]
                   AA[i * (nVar + 1) + j] = 0.0;
                   if (i < nVar && j < nVar)
                   {
                      // Axx = Axx*GG
                      for (int k = 0; k < nCon; k++)
                      {
                         AA[i * (nVar + 1) + j] += Axx[i * nCon + k] * GG[k * nVar + j];
                      }
                      // Axx = Axx + spdiags(diagx,0,nVar,nVar)
                      if (i == j)
                      {
                         AA[i * (nVar + 1) + j] += diagx[j];
                      }
                   }
                   // AA = [Axx  axz
                   //       axz' azz]
                   else if (i < nVar && j == nVar)
                   {
                      AA[i * (nVar + 1) + j] = axz[i];
                   }
                   else if (i == nVar && j < nVar)
                   {
                      AA[i * (nVar + 1) + j] = axz[j];
                   }
                   else
                   {
                      AA[i * (nVar + 1) + j] = azz;
                   }
                }
             }
             // bb = [-bx'; -bz]
             // bx = delx - GG'*(dellamyi./diaglamyi)
             // bz = delz - a'*(dellamyi./diaglamyi)
             for (int i = 0; i < nVar; i++)
             {
                bb[i] = -delx[i];
                for (int j = 0; j < nCon; j++)
                {
                   bb[i] -= GG[j * nVar + i] * (dellamyi[j] / diaglamyi[j]);
                }
             }
             bb[nVar] = -delz;
             for (int i = 0; i < nCon; i++)
             {
                bb[nVar] += mma->a[i] * (dellamyi[i] / diaglamyi[i]);
             }
             // ----------------------------------------------------------------------------
             //bb = AA\bb --> solve linear system of equations using LAPACK
             int info;
             int nLAP = nVar + 1;
             int nrhs = 1;
             int lda = nLAP;
             int ldb = nLAP;
             int* ipiv = new int[nLAP];
             dgesv_(&nLAP, &nrhs, AA, &lda, ipiv, bb, &ldb, &info);
             delete[] ipiv;
             for (int i = 0; i < nVar; i++)
             {
                dx[i] = bb[i];
             }
             dz = bb[nVar];
             // ----------------------------------------------------------------------------
             //dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi;
             for (int i = 0; i < nCon; i++)
             {
                sum = 0.0;
                for (int j = 0; j < nVar; j++)
                {
                   sum += GG[i * nVar + j] * dx[j];
                }
                dlam[i] = sum / diaglamyi[i] - dz * (mma->a[i] / diaglamyi[i]) + dellamyi[i] /
                          diaglamyi[i];
             }
          }

          for (int i = 0; i < nCon; i++)
          {
             dy[i] = -dely[i] / diagy[i] + dlam[i] / diagy[i];
             dmu[i] = -mma->mu[i] + epsi / mma->y[i] - (mma->mu[i] * dy[i]) / mma->y[i];
             ds[i] = -mma->s[i] + epsi / mma->lam[i] - (mma->s[i] * dlam[i]) / mma->lam[i];
             // xx = [y z lam xsi eta mu zet s]
             // dxx = [dy dz dlam dxsi deta dmu dzet ds]
             xx[i] = mma->y[i];
             xx[nCon + 1 + i] = mma->lam[i];
             xx[2 * nCon + 1 + 2 * nVar + i] = mma->mu[i];
             xx[3 * nCon + 2 * nVar + 2 + i] = mma->s[i];

             dxx[i] = dy[i];
             dxx[nCon + 1 + i] = dlam[i];
             dxx[2 * nCon + 1 + 2 * nVar + i] = dmu[i];
             dxx[3 * nCon + 2 * nVar + 2 + i] = ds[i];
          }
          xx[nCon] = mma->z;
          xx[3 * nCon + 2 * nVar + 1] = mma->zet;
          dxx[nCon] = dz;
          for (int i = 0; i < nVar; i++)
          {
             // NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   dxsi[i] = -mma->xsi[i] + epsi / mma->machineEpsilon - (mma->xsi[i] * dx[i]) /
                             mma->machineEpsilon;
                   deta[i] = -mma->eta[i] + epsi / mma->machineEpsilon + (mma->eta[i] * dx[i]) /
                             mma->machineEpsilon;
                }
                else
                {
                   dxsi[i] = -mma->xsi[i] + epsi / mma->machineEpsilon - (mma->xsi[i] * dx[i]) /
                             mma->machineEpsilon;
                   deta[i] = -mma->eta[i] + epsi / (beta[i] - mma->x[i]) +
                             (mma->eta[i] * dx[i]) / (beta[i] - mma->x[i]);
                }
             }
             else if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                dxsi[i] = -mma->xsi[i] + epsi / (mma->x[i] - alfa[i]) -
                          (mma->xsi[i] * dx[i]) / (mma->x[i] - alfa[i]);
                deta[i] = -mma->eta[i] + epsi / mma->machineEpsilon + (mma->eta[i] * dx[i]) /
                          mma->machineEpsilon;
             }
             else
             {
                dxsi[i] = -mma->xsi[i] + epsi / (mma->x[i] - alfa[i]) -
                          (mma->xsi[i] * dx[i]) / (mma->x[i] - alfa[i]);
                deta[i] = -mma->eta[i] + epsi / (beta[i] - mma->x[i]) +
                          (mma->eta[i] * dx[i]) / (beta[i] - mma->x[i]);
             }
             xx[nCon + 1 + nCon + i] = mma->xsi[i];
             xx[nCon + 1 + nCon + nVar + i] = mma->eta[i];
             dxx[nCon + 1 + nCon + i] = dxsi[i];
             dxx[nCon + 1 + nCon + nVar + i] = deta[i];
          }
          dzet = -mma->zet + epsi / mma->z - mma->zet * dz / mma->z;
          dxx[3 * nCon + 2 * nVar + 1] = dzet;

          stmxx = 0.0;
          for (int i = 0; i < (4 * nCon + 2 * nVar + 2); i++)
          {
             stepxx[i] = -1.01*dxx[i] /  xx[i];
             if (stepxx[i] > stmxx)
             {
                stmxx = stepxx[i];
             }
          }
          stmalfa = 0.0;
          stmbeta = 0.0;
          for (int i = 0; i < nVar; i++)
          {
             //NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                stepalfa[i] = -1.01*dx[i] / mma->machineEpsilon;
             }
             else
             {
                stepalfa[i] = -1.01*dx[i] / (mma->x[i] - alfa[i]);
             }
             if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                stepbeta[i] = 1.01*dx[i] / mma->machineEpsilon;
             }
             else
             {
                stepbeta[i] = 1.01*dx[i] / (beta[i] - mma->x[i]);
             }
             // --------------
             if (stepalfa[i] > stmalfa)
             {
                stmalfa = stepalfa[i];
             }
             if (stepbeta[i] > stmbeta)
             {
                stmbeta = stepbeta[i];
             }
          }
          stmalbe = std::max(stmalfa, stmbeta);
          stmalbexx = std::max(stmalbe, stmxx);
          stminv = std::max(stmalbexx, 1.0);
          steg = 1.0 / stminv;

          for (int i = 0; i < nVar; i++)
          {
             xold[i] = mma->x[i];
             xsiold[i] = mma->xsi[i];
             etaold[i] = mma->eta[i];
          }
          for (int i = 0; i < nCon; i++)
          {
             yold[i] = mma->y[i];
             lamold[i] = mma->lam[i];
             muold[i] = mma->mu[i];
             sold[i] = mma->s[i];
          }
          zold = mma->z;
          zetold = mma->zet;

          itto = 0;
          resinew = 2.0 * residunorm;
          while (resinew > residunorm && itto < 50)
          {
             itto++;

             for (int i = 0; i < nCon; ++i)
             {
                mma->y[i] = yold[i] + steg * dy[i];
                if (std::fabs(mma->y[i])< mma->machineEpsilon)
                {
                   mma->y[i] = mma->machineEpsilon;
                }

                mma->lam[i] = lamold[i] + steg * dlam[i];
                if (std::fabs(mma->lam[i])< mma->machineEpsilon )
                {
                   mma->lam[i] = mma->machineEpsilon;
                }
                mma->mu[i] = muold[i] + steg * dmu[i];
                mma->s[i] = sold[i] + steg * ds[i];
             }

             residu[nVar + nCon] = mma->a0 - mma->zet; //rez
             for (int i = 0; i < nVar; ++i)
             {
                mma->x[i] = xold[i] + steg * dx[i];
                mma->xsi[i] = xsiold[i] + steg * dxsi[i];
                mma->eta[i] = etaold[i] + steg * deta[i];

                ux1[i] = mma->upp[i] - mma->x[i];
                if (std::fabs(ux1[i]) < mma->machineEpsilon)
                {
                   ux1[i] = mma->machineEpsilon;
                }
                xl1[i] = mma->x[i] - mma->low[i];
                if (std::fabs(xl1[i]) < mma->machineEpsilon )
                {
                   xl1[i] = mma->machineEpsilon;
                }
                // plam & qlam
                plam[i] = p0[i];
                qlam[i] = q0[i];
                for (int j = 0; j < nCon; j++)
                {
                   plam[i] += P[j * nVar + i] * mma->lam[j];
                   qlam[i] += Q[j * nVar + i] * mma->lam[j];
                   residu[nVar + nCon] -= mma->a[j] * mma->lam[j]; //rez
                }

                // Assembly starts here

                residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                            mma->xsi[i] + mma->eta[i]; //rex
                //residu[nVar + nCon] -= mma->a[i] * mma->lam[i]; //rez
                residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * (mma->x[i] - alfa[i]) -
                                                     epsi; //rexsi
                if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
                {
                   residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * mma->machineEpsilon - epsi;
                }
                residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] *
                                                            (beta[i] - mma->x[i]) - epsi; //reeta
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] * mma->machineEpsilon -
                                                               epsi;
                }
             }
             mma->z = zold + steg * dz;
             if (std::fabs(mma->z) < mma->machineEpsilon)
             {
                mma->z = mma->machineEpsilon;
             }
             mma->zet = zetold + steg * dzet;

             // gvec = P/ux + Q/xl
             for (int i = 0; i < nCon; i++)
             {
                gvec[i] = 0.0;
                for (int j = 0; j < nVar; j++)
                {
                   gvec[i] += P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
                }
                residu[nVar + i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->mu[i] -
                                   mma->lam[i]; //rey
                residu[nVar + nCon + 1 + i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] +
                                              mma->s[i] - b[i]; //relam
                residu[nVar + nCon + 1 + nCon + 2 * nVar + i] = mma->mu[i] * mma->y[i] -
                                                                epsi; //remu
                residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon + 1 + i] = mma->lam[i] * mma->s[i]
                                                                        - epsi; //res
             }
             residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon] = mma->zet * mma->z -
                                                             epsi; //rezet

             // New residual
             sum = 0.0;
             for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
             {
                sum += residu[i] * residu[i];
             }
             // Norm of the residual
             resinew = std::sqrt(sum);
             steg = steg / 2.0;
          }
          residunorm = resinew;
          residumax = 0.0;
          for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
          {
             residumax = std::max(residumax, std::abs(residu[i]));
          }
          steg = steg * 2.0;
       }
       if (ittt > 198)
       {
          printf("Warning: Maximum number of iterations reached in subsolv.\n");
       }
       epsi = 0.1 * epsi;
    }

    // should return x, y, z, lam, xsi, eta, mu, zet, s
}


void MMA::MMASubParallel::AllocSubData(int nVar, int nCon)
{
    epsi = 1.0;
    ittt = itto = itera = 0;
    raa0 = 0.00001;
    move = 0.5;
    albefa = 0.1;
    xmamieps = 1e-5;
    ux1 = new double[nVar]; //ini
    xl1 = new double[nVar]; //ini
    plam = new double[nVar]; //ini
    qlam = new double[nVar]; //ini
    gvec = new double[nCon]; //ini
    residu = new double[3 * nVar + 4 * nCon + 2]; //ini
    GG = new double[nVar * nCon]; //ini
    delx = new double[nVar]; //init
    dely = new double[nCon]; //ini
    dellam = new double[nCon]; //ini
    dellamyi = new double[nCon];
    diagx = new double[nVar];//ini
    diagy = new double[nCon];//ini
    diaglamyi = new double[nCon]; //ini
    bb = new double[nVar + 1];
    bb1 = new double[nCon + 1];//ini
    Alam = new double[nCon * nCon];//ini
    AA = new double[(nVar + 1) * (nVar + 1)];
    AA1 = new double[(nCon + 1) * (nCon + 1)];//ini
    dlam = new double[nCon]; //ini
    dx = new double[nVar]; //ini
    dy = new double[nCon]; //ini
    dxsi = new double[nVar]; //ini
    deta = new double[nVar]; //ini
    dmu = new double[nCon]; //ini
    Axx = new double[nVar * nCon]; //ini
    axz = new double[nVar]; //ini
    ds = new double[nCon]; //ini
    xx = new double[4 * nCon + 2 * nVar + 2]; //ini
    dxx = new double[4 * nCon + 2 * nVar + 2]; //ini
    stepxx = new double[4 * nCon + 2 * nVar + 2]; //ini
    sum = 0;
    sum1 = new double[nVar];
    stepalfa = new double[nVar]; //ini
    stepbeta = new double[nVar]; //ini
    xold = new double[nVar]; //ini
    yold = new double[nCon]; //ini
    lamold = new double[nCon];//ini
    xsiold = new double[nVar];//ini
    etaold = new double[nVar];//ini
    muold = new double[nCon]; //ini
    sold = new double[nCon]; //ini
    q0 = new double[nVar]; //ini
    p0 = new double[nVar]; //ini
    P = new double[nCon * nVar]; //ini
    Q = new double[nCon * nVar]; //ini
    alfa = new double[nVar]; //ini
    beta = new double[nVar]; //ini
    xmami = new double[nVar];
    b = new double[nCon]; //ini

    b_local = new double[nCon];
    gvec_local = new double[nCon];
    Alam_local = new double[nCon * nCon];
    sum_local = new double[nCon];
    sum_global = new double[nCon];


    for(int i=0; i<(3 * nVar + 4 * nCon + 2);i++){
        residu[i]=0.0;
    }
}

void MMA::MMASubParallel::FreeSubData()
{
    delete[] sum1;
    delete[] ux1;
    delete[] xl1;
    delete[] plam;
    delete[] qlam;
    delete[] gvec;
    delete[] residu;
    delete[] GG;
    delete[] delx;
    delete[] dely;
    delete[] dellam;
    delete[] dellamyi;
    delete[] diagx;
    delete[] diagy;
    delete[] diaglamyi;
    delete[] bb;
    delete[] bb1;
    delete[] Alam;
    delete[] AA;
    delete[] AA1;
    delete[] dlam;
    delete[] dx;
    delete[] dy;
    delete[] dxsi;
    delete[] deta;
    delete[] dmu;
    delete[] Axx;
    delete[] axz;
    delete[] ds;
    delete[] xx;
    delete[] dxx;
    delete[] stepxx;
    delete[] stepalfa;
    delete[] stepbeta;
    delete[] xold;
    delete[] yold;
    delete[] lamold;
    delete[] xsiold;
    delete[] etaold;
    delete[] muold;
    delete[] sold;
    delete[] xmami;
    delete[] q0;
    delete[] p0;
    delete[] P;
    delete[] Q;
    delete[] alfa;
    delete[] beta;
    delete[] b;

    delete[] gvec_local;
    delete[] b_local;
    delete[] Alam_local;
    delete[] sum_local;
    delete[] sum_global;

}

void MMA::MMASubParallel::Update(const double* dfdx,
                                 const double* gx,
                                 const double* dgdx,
                                 const double* xmin,
                                 const double* xmax,
                                 const double* xval)
{
    MMA* mma = this->mma_ptr;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(mma->comm, &rank);
    MPI_Comm_size(mma->comm, &size);
     
    int nCon = mma->nCon;
    int nVar = mma->nVar;

    ittt = 0;
    itto = 0;
    epsi = 1.0;
    itera = 0;
    mma->z = 1.0;
    mma->zet = 1.0;

    for (int i = 0; i < nCon; i++)
    {
       b[i] = 0.0;
       b_local[i] = 0.0;
    }

    for (int i = 0; i < nVar; i++)
    {
       // Calculation of bounds alfa and beta according to:
       // alfa = max{xmin, low + 0.1(xval-low), xval-0.5(xmax-xmin)}
       // beta = min{xmax, upp - 0.1(upp-xval), xval+0.5(xmax-xmin)}

       alfa[i] = std::max(std::max(mma->low[i] + albefa * (xval[i] - mma->low[i]),
                                   xval[i] - move * (xmax[i] - xmin[i])), xmin[i]);
       beta[i] = std::min(std::min(mma->upp[i] - albefa * (mma->upp[i] - xval[i]),
                                   xval[i] + move * (xmax[i] - xmin[i])), xmax[i]);
       xmami[i] = std::max(xmax[i] - xmin[i], xmamieps);

       // Calculations of p0, q0, P, Q, and b
       ux1[i] = mma->upp[i] - xval[i];
       if(std::fabs(ux1[i]) <= mma->machineEpsilon)
       {
          ux1[i] = mma->machineEpsilon;
       }
       xl1[i] = xval[i] - mma->low[i];
       if (std::fabs(xl1[i]) <= mma->machineEpsilon)
       {
          xl1[i] = mma->machineEpsilon;
       }
       p0[i] = ( std::max(dfdx[i], 0.0) + 0.001 * (std::max(dfdx[i], 0.0) + std::max(-dfdx[i], 0.0)) + raa0 / xmami[i]) * ux1[i] * ux1[i];
       q0[i] = ( std::max(-dfdx[i], 0.0) + 0.001 * (std::max(dfdx[i], 0.0) + std::max(-dfdx[i], 0.0)) + raa0 / xmami[i]) * xl1[i] * xl1[i];
    }

    // P = max(dgdx,0)
    // Q = max(-dgdx,0)
    // P = P + 0.001(P+Q) + raa0/xmami
    // Q = Q + 0.001(P+Q) + raa0/xmami
    for (int i = 0; i < nCon; i++)
    {
       for (int j = 0; j < nVar; j++)
       {
          // P = P * spdiags(ux2,0,n,n)
          // Q = Q * spdiags(xl2,0,n,n)
          P[i * nVar + j] = (std::max(dgdx[i * nVar + j], 0.0) + 0.001 * (std::max(dgdx[i * nVar + j], 0.0) + std::max(-1*dgdx[i * nVar + j], 0.0)) + raa0 / xmami[j]) * ux1[j] * ux1[j];
          Q[i * nVar + j] = (std::max(-1*dgdx[i * nVar + j], 0.0) + 0.001 * (std::max(dgdx[i * nVar + j], 0.0) + std::max(-1*dgdx[i * nVar + j], 0.0)) + raa0 / xmami[j]) * xl1[j] * xl1[j];
          // b = P/ux1 + Q/xl1 - gx
          b_local[i] = b_local[i] + P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
       }
    }
    MPI_Allreduce(b_local, b, nCon, MPI_DOUBLE, MPI_SUM, mma->comm);

    for (int i = 0; i < nCon; i++)
    {
        b[i] = b[i] - gx[i];      
    }


    for (int i = 0; i < nVar; i++)
    {
       mma->x[i] = 0.5 * (alfa[i] + beta[i]);
       mma->xsi[i] = 1.0/(mma->x[i] - alfa[i]);
       mma->xsi[i] = std::max(mma->xsi[i], 1.0);
       mma->eta[i] = 1.0/(beta[i] - mma->x[i]);
       mma->eta[i] = std::max(mma->eta[i], 1.0);
       ux1[i] = 0.0;
       xl1[i] = 0.0;
    }

    for (int i = 0; i < nCon; i++)
    {
       mma->y[i] = 1.0;
       mma->lam[i] = 1.0;
       mma->mu[i] = std::max(1.0, 0.5 * mma->c[i]);
       mma->s[i] = 1.0;
    }

    while (epsi > mma->epsimin)
    {
       residu[nVar + nCon] = mma->a0 - mma->zet; //rez
       for (int i = 0; i < nVar; i++)
       {
          ux1[i] = mma->upp[i] - mma->x[i];
          if(std::fabs(ux1[i]) < mma->machineEpsilon){
              ux1[i] = mma->machineEpsilon;
          }

          xl1[i] = mma->x[i] - mma->low[i];
          if(std::fabs(xl1[i]) < mma->machineEpsilon)
          {
              xl1[i] = mma->machineEpsilon;
          }

          // plam = P' * lam, qlam = Q' * lam
          plam[i] = p0[i];
          qlam[i] = q0[i];
          for (int j = 0; j < nCon; j++)
          {
             plam[i] += P[j * nVar + i] * mma->lam[j];
             qlam[i] += Q[j * nVar + i] * mma->lam[j];
             residu[nVar + nCon] -= mma->a[j] * mma->lam[j]; //rez
          }
          residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                      mma->xsi[i] + mma->eta[i]; //rex
          //residu[nVar + nCon] -= mma->a[i] * mma->lam[i]; //rez
          residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * (mma->x[i] - alfa[i]) -
                                               epsi; //rexsi
          if (std::fabs(mma->x[i]-alfa[i]) < mma->machineEpsilon)
          {
             residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * mma->machineEpsilon - epsi;
          }
          residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] *
                                                      (beta[i] - mma->x[i]) - epsi; //reeta
          if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
          {
             residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] * mma->machineEpsilon -
                                                         epsi;
          }
       }
       for (int i = 0; i < nCon; i++)
       {
          gvec_local[i] = 0.0;
          // gvec = P/ux + Q/xl
          for (int j = 0; j < nVar; j++)
          {
             gvec_local[i] = gvec_local[i] + P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
          }
       }
       MPI_Allreduce(gvec_local, gvec, nCon, MPI_DOUBLE, MPI_SUM, mma->comm);

if( rank == 0)   //Mathias
{
       for (int i = 0; i < nCon; i++)
       {
          residu[nVar + i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->mu[i] -
                             mma->lam[i]; //rey
          residu[nVar + nCon + 1 + i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] +
                                        mma->s[i] - b[i]; //relam
          residu[nVar + nCon + 1 + nCon + 2 * nVar + i] = mma->mu[i] * mma->y[i] -
                                                          epsi; //remu
          residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon + 1 + i] = mma->lam[i] * mma->s[i]
                                                                  - epsi; //res
       }       
       residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon] = mma->zet * mma->z - epsi;
}

       //Get vector product and maximum absolute value
       residunorm = 0.0;
       residumax = 0.0;
       for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
       {
          residunorm += residu[i] * residu[i];
          residumax = std::max(residumax, std::abs(residu[i]));
       }

       global_norm = 0.0;
       global_max = 0.0;  
       MPI_Allreduce(&residunorm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, mma->comm);
       MPI_Allreduce(&residumax, &global_max, 1, MPI_DOUBLE, MPI_MAX, mma->comm);
       // Norm of the residual
       residunorm = std::sqrt(global_norm);
       residumax = global_max;

       ittt = 0;

       while (residumax > 0.9 * epsi && ittt < 200)
       {
          ittt++;
          for (int i = 0; i < nVar; i++)
          {
             ux1[i] = mma->upp[i] - mma->x[i];
             if(std::fabs(ux1[i]) < mma->machineEpsilon)
             {
                 ux1[i] = mma->machineEpsilon;
             }

             xl1[i] = mma->x[i] - mma->low[i];
             if(std::fabs(xl1[i]) <= mma->machineEpsilon)
             {
                 xl1[i] = mma->machineEpsilon;
             }
             // plam = P' * lam, qlam = Q' * lam
             plam[i] = p0[i];
             qlam[i] = q0[i];
             for (int j = 0; j < nCon; j++)
             {
                plam[i] += P[j * nVar + i] * mma->lam[j];
                qlam[i] += Q[j * nVar + i] * mma->lam[j];
             }
             // NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]);
                   diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                   (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / mma->machineEpsilon + mma->eta[i] /
                              mma->machineEpsilon;
                }
                else
                {
                   delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                             mma->machineEpsilon + epsi / (beta[i] - mma->x[i]);
                   diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                   (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                              mma->eta[i] / (beta[i] - mma->x[i]);
                }
             }
             else if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                          (mma->x[i] - alfa[i]) + epsi / mma->machineEpsilon;
                diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                           mma->eta[i] / mma->machineEpsilon;
             }
             else
             {
                delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                          (mma->x[i] - alfa[i]) + epsi / (beta[i] - mma->x[i]);
                diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                (xl1[i] * xl1[i] * xl1[i])) + mma->xsi[i] / (mma->x[i] - alfa[i]) +
                           mma->eta[i] / (beta[i] - mma->x[i]);
             }
          }

          for (int i = 0; i < nCon; i++)
          {
             gvec_local[i] = 0.0;
             // gvec = P/ux + Q/xl
             for (int j = 0; j < nVar; j++)
             {
                gvec_local[i] = gvec_local[i] + P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
                GG[i * nVar + j] = P[i * nVar + j] / (ux1[j] * ux1[j]) - Q[i * nVar + j] / (xl1[j] * xl1[j]);
             }
          }
          MPI_Allreduce(gvec_local, gvec, nCon, MPI_DOUBLE, MPI_SUM, mma->comm);

          delz = mma->a0 - epsi / mma->z;
          for (int i = 0; i < nCon; i++)
          {
             dely[i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->lam[i] - epsi / mma->y[i];
             delz -= mma->a[i] * mma->lam[i];
             dellam[i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] - b[i] + epsi /
                         mma->lam[i];
             diagy[i] = mma->d[i] + mma->mu[i] / mma->y[i];
             diaglamyi[i] = mma->s[i] / mma->lam[i] + 1.0 / diagy[i];
          }

          if (nCon < nVar)
          {
             // bb1 = dellam + dely./diagy - GG*(delx./diagx);
             // bb1 = [bb1; delz];
             for (int j = 0; j < nCon; j++)
             {
                sum_local[j] = 0.0;
                for (int i = 0; i < nVar; i++)
                {
                   sum_local[j] = sum_local[j] + GG[j * nVar + i] * (delx[i] / diagx[i]);
                }
             }
             MPI_Allreduce(sum_local, sum_global, nCon, MPI_DOUBLE, MPI_SUM, mma->comm);
             
             for (int j = 0; j < nCon; j++)
             {
                bb1[j] = - sum_global[j] + dellam[j] + dely[j] / diagy[j];
             }
             bb1[nCon] = delz;

             // Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
             for (int i = 0; i < nCon; i++)
             {
                // Axx = GG*spdiags(diagxinv,0,n,n);
                for (int k = 0; k < nVar; k++)
                {
                   Axx[i * nVar + k] = GG[k * nCon + i] / diagx[k];
                }
             }
             // Alam = spdiags(diaglamyi,0,m,m) + Axx*GG';
             for (int i = 0; i < nCon; i++)
             {
                for (int j = 0; j < nCon; j++)
                {
                   Alam_local[i * nCon + j] = 0.0;
                   for (int k = 0; k < nVar; k++)
                   {
                      Alam_local[i * nCon + j] += Axx[i * nVar + k] * GG[j * nVar + k];
                   }
                }
             }
             MPI_Reduce(Alam_local, Alam, nCon * nCon, MPI_DOUBLE, MPI_SUM, 0, mma->comm);

             if (0 == rank)   
             {
               for (int i = 0; i < nCon; i++)
               {
                  for (int j = 0; j < nCon; j++)
                  {
                     if (i == j)
                     {
                        Alam[i * nCon + j] += diaglamyi[i];
                     }
                  }
               }
               // AA1 = [Alam     a
               //       a'    -zet/z];
               for (int i = 0; i < nCon; i++)
               {
                  for (int j = 0; j < nCon; j++)
                  {
                     AA1[i * (nCon + 1) + j] = Alam[i * nCon + j];
                  }
                  AA1[i * (nCon + 1) + nCon] = mma->a[i];
               }
               for (int i = 0; i < nCon; i++)
               {
                  AA1[nCon * (nCon + 1) + i] = mma->a[i];
               }
               AA1[(nCon + 1) * (nCon + 1) - 1] = -mma->zet / mma->z;
            
               // -------------------------------------------------------------------
               //bb1 = AA1\bb1 --> solve linear system of equations using LAPACK
               int info;
               int nLAP = nCon + 1;
               int nrhs = 1;
               int lda = nLAP;
               int ldb = nLAP;
               int* ipiv = new int[nLAP];
               dgesv_(&nLAP, &nrhs, AA1, &lda, ipiv, bb1, &ldb, &info);
               if (info == 0) {
                  delete[] ipiv;
               } else if (info > 0) {
                  std::cerr << "Error: matrix is singular." << std::endl;
               } else {
                  std::cerr << "Error: Argument " << info << " has illegal value." << std::endl;
               }
            }

            MPI_Bcast(bb1, nCon + 1, MPI_DOUBLE, 0, mma->comm); 

            for (int i = 0; i < nCon; i++)
            {
               dlam[i] = bb1[i];
            }
            dz = bb1[nCon];

             // ----------------------------------------------------------------------------
             //dx = -(GG'*dlam)./diagx - delx./diagx;
             for (int i = 0; i < nVar; i++)
             {
                sum = 0.0;
                for (int j = 0; j < nCon; j++)
                {
                   sum = sum + GG[j * nVar + i] * dlam[j];
                }
                dx[i] = -sum / diagx[i] - delx[i] / diagx[i];
             }
          }
          else
          {
            /*
             azz = mma->zet / mma->z;
             for (int i = 0; i < nCon; i++)
             {
                dellamyi[i] = dellam[i] + dely[i] / diagy[i];
                // azz = zet/z + a'*(a./diaglamyi)
                azz += mma->a[i] * (mma->a[i] / diaglamyi[i]);
             }

             // Axx = spdiags(diagx,0,nVar,nVar) + GG'*spdiags(diaglamyiinv,0,nCon,nCon)*GG;
             // AA = [Axx      axz
             //       axz'     azz];
             for (int i = 0; i < nVar; i++)
             {
                // Axx =  GG'*spdiags(diaglamyiinv,0,nCon,nCon);
                for (int k = 0; k < nCon; k++)
                {
                   Axx[i * nCon + k] = GG[k * nVar + i] / diaglamyi[k];
                }
                axz[i] = 0.0;
                // axz = -GG'*(a./diaglamyi)
                for (int j = 0; j < nCon; j++)
                {
                   axz[i] -= GG[j * nVar + i] * (mma->a[j] / diaglamyi[j]);
                }
             }
             //Assemble matrix AA
             for (int i = 0; i < (nVar + 1); i++)
             {
                for (int j = 0; j < (nVar + 1); j++)
                {
                   // AA = [Axx  .
                   //       .    .]
                   AA[i * (nVar + 1) + j] = 0.0;
                   if (i < nVar && j < nVar)
                   {
                      // Axx = Axx*GG
                      for (int k = 0; k < nCon; k++)
                      {
                         AA[i * (nVar + 1) + j] += Axx[i * nCon + k] * GG[k * nVar + j];
                      }
                      // Axx = Axx + spdiags(diagx,0,nVar,nVar)
                      if (i == j)
                      {
                         AA[i * (nVar + 1) + j] += diagx[j];
                      }
                   }
                   // AA = [Axx  axz
                   //       axz' azz]
                   else if (i < nVar && j == nVar)
                   {
                      AA[i * (nVar + 1) + j] = axz[i];
                   }
                   else if (i == nVar && j < nVar)
                   {
                      AA[i * (nVar + 1) + j] = axz[j];
                   }
                   else
                   {
                      AA[i * (nVar + 1) + j] = azz;
                   }
                }
             }
             // bb = [-bx'; -bz]
             // bx = delx - GG'*(dellamyi./diaglamyi)
             // bz = delz - a'*(dellamyi./diaglamyi)
             for (int i = 0; i < nVar; i++)
             {
                bb[i] = -delx[i];
                for (int j = 0; j < nCon; j++)
                {
                   bb[i] -= GG[j * nVar + i] * (dellamyi[j] / diaglamyi[j]);
                }
             }
             bb[nVar] = -delz;
             for (int i = 0; i < nCon; i++)
             {
                bb[nVar] += mma->a[i] * (dellamyi[i] / diaglamyi[i]);
             }
             // ----------------------------------------------------------------------------
             //bb = AA\bb --> solve linear system of equations using LAPACK
             int info;
             int nLAP = nVar + 1;
             int nrhs = 1;
             int lda = nLAP;
             int ldb = nLAP;
             int* ipiv = new int[nLAP];
             dgesv_(&nLAP, &nrhs, AA, &lda, ipiv, bb, &ldb, &info);
             delete[] ipiv;
             for (int i = 0; i < nVar; i++)
             {
                dx[i] = bb[i];
             }
             dz = bb[nVar];
             // ----------------------------------------------------------------------------
             //dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi;
             for (int i = 0; i < nCon; i++)
             {
                sum = 0.0;
                for (int j = 0; j < nVar; j++)
                {
                   sum += GG[i * nVar + j] * dx[j];
                }
                dlam[i] = sum / diaglamyi[i] - dz * (mma->a[i] / diaglamyi[i]) + dellamyi[i] /
                          diaglamyi[i];
             }
             */
          }

          for (int i = 0; i < nCon; i++)
          {
             dy[i] = -dely[i] / diagy[i] + dlam[i] / diagy[i];
             dmu[i] = -mma->mu[i] + epsi / mma->y[i] - (mma->mu[i] * dy[i]) / mma->y[i];
             ds[i] = -mma->s[i] + epsi / mma->lam[i] - (mma->s[i] * dlam[i]) / mma->lam[i];
             // xx = [y z lam xsi eta mu zet s]
             // dxx = [dy dz dlam dxsi deta dmu dzet ds]
             xx[i] = mma->y[i];
             xx[nCon + 1 + i] = mma->lam[i];
             xx[2 * nCon + 1 + 2 * nVar + i] = mma->mu[i];
             xx[3 * nCon + 2 * nVar + 2 + i] = mma->s[i];

             dxx[i] = dy[i];
             dxx[nCon + 1 + i] = dlam[i];
             dxx[2 * nCon + 1 + 2 * nVar + i] = dmu[i];
             dxx[3 * nCon + 2 * nVar + 2 + i] = ds[i];
          }
          xx[nCon] = mma->z;
          xx[3 * nCon + 2 * nVar + 1] = mma->zet;
          dxx[nCon] = dz;
          for (int i = 0; i < nVar; i++)
          {
             // NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   dxsi[i] = -mma->xsi[i] + epsi / mma->machineEpsilon - (mma->xsi[i] * dx[i]) /
                             mma->machineEpsilon;
                   deta[i] = -mma->eta[i] + epsi / mma->machineEpsilon + (mma->eta[i] * dx[i]) /
                             mma->machineEpsilon;
                }
                else
                {
                   dxsi[i] = -mma->xsi[i] + epsi / mma->machineEpsilon - (mma->xsi[i] * dx[i]) /
                             mma->machineEpsilon;
                   deta[i] = -mma->eta[i] + epsi / (beta[i] - mma->x[i]) +
                             (mma->eta[i] * dx[i]) / (beta[i] - mma->x[i]);
                }
             }
             else if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                dxsi[i] = -mma->xsi[i] + epsi / (mma->x[i] - alfa[i]) -
                          (mma->xsi[i] * dx[i]) / (mma->x[i] - alfa[i]);
                deta[i] = -mma->eta[i] + epsi / mma->machineEpsilon + (mma->eta[i] * dx[i]) /
                          mma->machineEpsilon;
             }
             else
             {
                dxsi[i] = -mma->xsi[i] + epsi / (mma->x[i] - alfa[i]) -
                          (mma->xsi[i] * dx[i]) / (mma->x[i] - alfa[i]);
                deta[i] = -mma->eta[i] + epsi / (beta[i] - mma->x[i]) +
                          (mma->eta[i] * dx[i]) / (beta[i] - mma->x[i]);
             }
             xx[nCon + 1 + nCon + i] = mma->xsi[i];
             xx[nCon + 1 + nCon + nVar + i] = mma->eta[i];
             dxx[nCon + 1 + nCon + i] = dxsi[i];
             dxx[nCon + 1 + nCon + nVar + i] = deta[i];
          }
          dzet = -mma->zet + epsi / mma->z - mma->zet * dz / mma->z;
          dxx[3 * nCon + 2 * nVar + 1] = dzet;

          stmxx = 0.0;
          for (int i = 0; i < (4 * nCon + 2 * nVar + 2); i++)
          {
             stepxx[i] = -1.01*dxx[i] /  xx[i];
             stmxx = std::max(stepxx[i], stmxx);
          }

          MPI_Allreduce(&stmxx, &stmxx_global, 1, MPI_DOUBLE, MPI_MAX, mma->comm);

          stmalfa = 0.0;
          stmbeta = 0.0;
          for (int i = 0; i < nVar; i++)
          {
             //NaN-Avoidance
             if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
             {
                stepalfa[i] = -1.01*dx[i] / mma->machineEpsilon;
             }
             else
             {
                stepalfa[i] = -1.01*dx[i] / (mma->x[i] - alfa[i]);
             }
             if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
             {
                stepbeta[i] = 1.01*dx[i] / mma->machineEpsilon;
             }
             else
             {
                stepbeta[i] = 1.01*dx[i] / (beta[i] - mma->x[i]);
             }
             // --------------
             stmalfa = std::max(stepalfa[i], stmalfa);
             stmbeta = std::max(stepbeta[i], stmbeta);   
          }

          MPI_Allreduce(&stmalfa, &stmalfa_global, 1, MPI_DOUBLE, MPI_MAX, mma->comm);
          MPI_Allreduce(&stmbeta, &stmbeta_global, 1, MPI_DOUBLE, MPI_MAX, mma->comm);
          stminv = std::max(std::max(std::max(stmalfa_global, stmbeta_global), stmxx_global), 1.0);
          steg = 1.0 / stminv;

          for (int i = 0; i < nVar; i++)
          {
             xold[i] = mma->x[i];
             xsiold[i] = mma->xsi[i];
             etaold[i] = mma->eta[i];
          }
          for (int i = 0; i < nCon; i++)
          {
             yold[i] = mma->y[i];
             lamold[i] = mma->lam[i];
             muold[i] = mma->mu[i];
             sold[i] = mma->s[i];
          }
          zold = mma->z;
          zetold = mma->zet;

          itto = 0;
          resinew = 2.0 * residunorm;
          while (resinew > residunorm && itto < 50)
          {
             itto++;

             for (int i = 0; i < nCon; ++i)
             {
                mma->y[i] = yold[i] + steg * dy[i];
                if (std::fabs(mma->y[i])< mma->machineEpsilon)
                {
                   mma->y[i] = mma->machineEpsilon;
                }

                mma->lam[i] = lamold[i] + steg * dlam[i];
                if (std::fabs(mma->lam[i])< mma->machineEpsilon )
                {
                   mma->lam[i] = mma->machineEpsilon;
                }
                mma->mu[i] = muold[i] + steg * dmu[i];
                mma->s[i] = sold[i] + steg * ds[i];
             }

             residu[nVar + nCon] = mma->a0 - mma->zet; //rez
             for (int i = 0; i < nVar; ++i)
             {
                mma->x[i] = xold[i] + steg * dx[i];
                mma->xsi[i] = xsiold[i] + steg * dxsi[i];
                mma->eta[i] = etaold[i] + steg * deta[i];

                ux1[i] = mma->upp[i] - mma->x[i];
                if (std::fabs(ux1[i]) < mma->machineEpsilon)
                {
                   ux1[i] = mma->machineEpsilon;
                }
                xl1[i] = mma->x[i] - mma->low[i];
                if (std::fabs(xl1[i]) < mma->machineEpsilon )
                {
                   xl1[i] = mma->machineEpsilon;
                }
                // plam & qlam
                plam[i] = p0[i];
                qlam[i] = q0[i];
                for (int j = 0; j < nCon; j++)
                {
                   plam[i] += P[j * nVar + i] * mma->lam[j];
                   qlam[i] += Q[j * nVar + i] * mma->lam[j];
                   residu[nVar + nCon] -= mma->a[j] * mma->lam[j]; //rez
                }

                // Assembly starts here

                residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                            mma->xsi[i] + mma->eta[i]; //rex
                //residu[nVar + nCon] -= mma->a[i] * mma->lam[i]; //rez
                residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * (mma->x[i] - alfa[i]) -
                                                     epsi; //rexsi
                if (std::fabs(mma->x[i] - alfa[i]) < mma->machineEpsilon)
                {
                   residu[nVar + nCon + 1 + nCon + i] = mma->xsi[i] * mma->machineEpsilon - epsi;
                }
                residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] *
                                                            (beta[i] - mma->x[i]) - epsi; //reeta
                if (std::fabs(beta[i] - mma->x[i]) < mma->machineEpsilon)
                {
                   residu[nVar + nCon + 1 + nCon + nVar + i] = mma->eta[i] * mma->machineEpsilon -
                                                               epsi;
                }
             }
             mma->z = zold + steg * dz;
             if (std::fabs(mma->z) < mma->machineEpsilon)
             {
                mma->z = mma->machineEpsilon;
             }
             mma->zet = zetold + steg * dzet;

             // gvec = P/ux + Q/xl
             for (int i = 0; i < nCon; i++)
             {
                gvec_local[i] = 0.0;
                for (int j = 0; j < nVar; j++)
                {
                   gvec_local[i] = gvec_local[i] + P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
                }
             }
             MPI_Allreduce(gvec_local, gvec, nCon, MPI_DOUBLE, MPI_SUM, mma->comm); 


            if (rank == 0)
            {
             for (int i = 0; i < nCon; i++)
             {
                residu[nVar + i] = mma->c[i] + mma->d[i] * mma->y[i] - mma->mu[i] -
                                   mma->lam[i]; //rey
                residu[nVar + nCon + 1 + i] = gvec[i] - mma->a[i] * mma->z - mma->y[i] +
                                              mma->s[i] - b[i]; //relam
                residu[nVar + nCon + 1 + nCon + 2 * nVar + i] = mma->mu[i] * mma->y[i] -
                                                                epsi; //remu
                residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon + 1 + i] = mma->lam[i] * mma->s[i]
                                                                        - epsi; //res
             }
             residu[nVar + nCon + 1 + 2 * nVar + 2 * nCon] = mma->zet * mma->z -
                                                             epsi; //rezet
            }

             //Get vector product and maximum absolute value
             resinew = 0.0;
             for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
             {
                resinew = resinew + residu[i] * residu[i];
             }

             global_norm = 0.0;
             MPI_Allreduce(&resinew, &global_norm, 1, MPI_DOUBLE, MPI_SUM, mma->comm);

             // Norm of the residual
             resinew = std::sqrt(global_norm);

             steg = steg / 2.0;
          }                                                         // checkl this 

          residunorm = resinew;
          residumax = 0.0;
          for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
          {
             residumax = std::max(residumax, std::abs(residu[i]));
          }
          global_max = 0.0;
          MPI_Allreduce(&residumax, &global_max, 1, MPI_DOUBLE, MPI_MAX, mma->comm);
          residumax = global_max;
          steg = steg * 2.0;

       }
       if (ittt > 198)
       {
          printf("Warning: Maximum number of iterations reached in subsolv.\n");
       }
       epsi = 0.1 * epsi;
    }

    // should return x, y, z, lam, xsi, eta, mu, zet, s

}

void MMA::InitData(double *xval)
{

    for (int i = 0; i < nVar; i++)
    {
       x[i]=xval[i];
       xo1[i] = 0.0;
       xo2[i] = 0.0;
    }


    for (int i = 0; i < nCon; i++)
    {
       a[i] = 0.0;
       c[i] = 1000.0;
       d[i] = 1.0;
    }
    a0 = 1.0;

}

/// Serial MMA
MMA::MMA(int nVar, int nCon, double *xval)
{
#ifdef MFEM_USE_MPI
   comm=MPI_COMM_SELF;
#endif

   AllocData(nVar,nCon);
   InitData(xval);
   // allocate the serial subproblem
   mSubProblem = new MMA::MMASubSerial(this, nVar,nCon);
}


MMA::MMA(MPI_Comm comm_, int nVar, int nCon, double *xval)
{
   comm=comm_;

   AllocData(nVar,nCon);
   InitData(xval);
   // allocate the serial subproblem
   mSubProblem = new MMA::MMASubParallel(this, nVar,nCon);
}


MMA::~MMA()
{
    delete mSubProblem;
    FreeData();
}

void MMA::AllocData(int nVariables,int nConstr)
{
    //accessed by the subproblems
    nVar = nVariables;
    nCon = nConstr;

    x= new double[nVar]; //ini
    xo1 = new double[nVar]; //ini
    xo2 = new double[nVar]; //ini

    y = new double[nCon]; //ini
    c = new double[nCon]; //ini
    d = new double[nCon]; //ini
    a = new double[nCon]; //ini

    lam = new double[nCon]; //ini

    xsi = new double[nVar];//ini
    eta = new double[nVar];//ini

    mu = new double[nCon]; //ini
    s = new double[nCon]; //ini

    z = zet = 1.0;
    kktnorm = 10;
    machineEpsilon = 1e-10;


    //accessed by MMA
    epsimin = 1e-7;
    asyinit = 0.5;
    asyincr = 1.1;
    asydecr = 0.7;
    low = new double[nVar]; //ini
    upp = new double[nVar]; //ini
    factor = new double[nVar]; //ini
    lowmin = lowmax = uppmin = uppmax = zz = 0.0;

}

void MMA::FreeData()
{

    //accessed from the subproblems
    delete[] x;
    delete[] xo1;
    delete[] xo2;

    delete[] y;
    delete[] c;
    delete[] d;
    delete[] a;

    delete[] lam;
    delete[] xsi;
    delete[] eta;
    delete[] mu;
    delete[] s;

    //accessed only from MMA
    delete[] factor;
    delete[] low;
    delete[] upp;

}

void MMA::Update(int iter, const double* dfdx,
                 const double* gx,const double* dgdx,
                 const double* xmin, const double* xmax,
                 double* xval)
{
    // Calculation of the asymptotes low and upp
    if (iter < 3)
    {
       for (int i = 0; i < nVar; i++)
       {
          low[i] = xval[i] - asyinit * (xmax[i] - xmin[i]);
          upp[i] = xval[i] + asyinit * (xmax[i] - xmin[i]);
       }
    }
    else
    {
       for (int i = 0; i < nVar; i++)
       {
          //Determine sign
          zz = (xval[i] - xo1[i]) * (xo1[i] - xo2[i]);
          if ( zz > 0.0)
          {
             factor[i] =  asyincr;
          }
          else if ( zz < 0.0)
          {
             factor[i] =  asydecr;
          }
          else
          {
             factor[i] =  1.0;
          }


          //Find new asymptote
          low[i] = xval[i] - factor[i] * (xo1[i] - low[i]);
          upp[i] = xval[i] + factor[i] * (upp[i] - xo1[i]);

          lowmin = xval[i] - 10.0 * (xmax[i] - xmin[i]);
          lowmax = xval[i] - 0.01 * (xmax[i] - xmin[i]);
          uppmin = xval[i] + 0.01 * (xmax[i] - xmin[i]);
          uppmax = xval[i] + 10.0 * (xmax[i] - xmin[i]);

          low[i] = std::max(low[i], lowmin);
          low[i] = std::min(low[i], lowmax);
          upp[i] = std::max(upp[i], uppmin);
          upp[i] = std::min(upp[i], uppmax);
       }
    }

    for(int i=0;i<nVar;i++){
        std::cout<<" "<<low[i];
    }
    std::cout<<std::endl;
    for(int i=0;i<nVar;i++){
        std::cout<<" "<<upp[i];
    }
    std::cout<<std::endl;

    mSubProblem->Update(dfdx,gx,dgdx,xmin,xmax,xval);
    // Update design variables
    for (int i = 0; i < nVar; i++)
    {
       xo2[i] = xo1[i];
       xo1[i] = xval[i];
       xval[i] = x[i];
    }
}

}
