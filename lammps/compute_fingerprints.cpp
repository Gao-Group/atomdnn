/* ------------------------------------------------------------------------- */

#include "compute_fingerprints.h"
#include <cstring>
#include <iostream>
#include <cmath>
#include <numeric>
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeFingerprints::ComputeFingerprints(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), 
  cutsq(0.0), list(NULL), eta_G2(NULL), zeta(NULL), eta_G4(NULL), lambda(NULL), fingerprints(NULL)

{

  if (narg < 4) error->all(FLERR,"Illegal compute fingerprints command");

  // read ACSF parameters and compute the number of fingerprints
  double cut = atof(arg[3]);
  cutsq = cut*cut;

  int g4_1 = 0;
  int g4_2 = 0;
  int g4_3 = 0;

  for (int iarg = 4; iarg < narg; iarg++) {
    if (strcmp(arg[iarg],"etaG2") == 0) {
      g2_flag = 1;
      n_etaG2 = 0;
      while (strcmp(arg[iarg+n_etaG2+1],"zeta") && strcmp(arg[iarg+n_etaG2+1],"etaG4") && strcmp(arg[iarg+n_etaG2+1],"lambda")!= 0 && strcmp(arg[iarg+n_etaG2+1],"end")!= 0)
        n_etaG2++;
      memory->create(eta_G2,n_etaG2,"fingerprints:eta_G2");
      for(int c = 0; c < n_etaG2; c++)
        eta_G2[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"etaG4") == 0) {
      g4_1 = 1;
      n_etaG4 = 0;
      while (strcmp(arg[iarg+n_etaG4+1],"etaG2") && strcmp(arg[iarg+n_etaG4+1],"zeta") && strcmp(arg[iarg+n_etaG4+1],"lambda")!= 0 && strcmp(arg[iarg+n_etaG4+1],"end")!= 0)
        n_etaG4++;
      memory->create(eta_G4,n_etaG4,"fingerprints:eta_G4");
      for(int c = 0; c < n_etaG4; c++)
        eta_G4[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"zeta") == 0) {
      g4_2 = 1;
      n_zeta = 0;
      while (strcmp(arg[iarg+n_zeta+1],"etaG2") && strcmp(arg[iarg+n_zeta+1],"etaG4") && strcmp(arg[iarg+n_zeta+1],"lambda")!= 0 && strcmp(arg[iarg+n_zeta+1],"end")!= 0)
        n_zeta++;
      memory->create(zeta,n_zeta,"fingerprints:zeta");
      for(int c = 0; c < n_zeta; c++)
        zeta[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"lambda") == 0) {
      g4_3 = 1;
      n_lambda = 0;
      while (strcmp(arg[iarg+n_lambda+1],"zeta") && strcmp(arg[iarg+n_lambda+1],"etaG4") && strcmp(arg[iarg+n_lambda+1],"etaG2")!= 0 && strcmp(arg[iarg+n_lambda+1],"end")!= 0)
        n_lambda++;
      memory->create(lambda,n_lambda,"fingerprints:lambda");
      for(int c = 0; c < n_lambda; c++)
        lambda[c] = atof(arg[iarg+c+1]);
    }
  }

  if (g4_1 && g4_2 && g4_3 == 1)  g4_flag = 1;

  int ntypes = atom->ntypes;
  int ntypes_combinations = ntypes*(ntypes+1)/2;
  n_fingerprints = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  size_peratom_cols = n_fingerprints;

  nmax_atom = 0; 
  peratom_flag = 1;

}

/* ---------------------------------------------------------------------- */

ComputeFingerprints::~ComputeFingerprints()

{

  //  memory->destroy(array_atom);
  memory->destroy(fingerprints);
  memory->destroy(eta_G2);
  memory->destroy(eta_G4);
  memory->destroy(zeta);
  memory->destroy(lambda);

}

/* ---------------------------------------------------------------------- */

void ComputeFingerprints::init()

{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"fingerprints") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute fingerprints");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

}

/* ---------------------------------------------------------------------- */

void ComputeFingerprints::init_list(int /*id*/, NeighList *ptr)

{

  list = ptr;

}

/* ---------------------------------------------------------------------- */

void ComputeFingerprints::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  neighbor->build_one(list);

  // Get initial atoms data and neighborlists
  const int inum = list->inum;
  const int* const ilist = list->ilist;
  const int* const numneigh = list->numneigh;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  int const ntypes = atom->ntypes;
  double** const x = atom->x;
  int * const tag = atom->tag;
  const int* const mask = atom->mask;
  double pi = 3.14159265358979323846;

  // Initialize fingerprnts vector per atom
  double fingerprints_atom[size_peratom_cols];
  for (int i=0;i<size_peratom_cols;i++)
    fingerprints_atom[i] = 0;

  // when the atoms number in processor increases, re-allocate fingerprints 
  if (inum > nmax_atom) {
    memory->destroy(fingerprints);
    nmax_atom = inum;
    memory->create(fingerprints,nmax_atom,size_peratom_cols,
                     "ComputeFingerprints:fingerprints");
    array_atom = fingerprints;
  }

  int position[ntypes][ntypes];
  int pos = 0;
  for (int pos_1 = 0; pos_1 < ntypes; pos_1++)  {
    for (int pos_2 = pos_1; pos_2 < ntypes; pos_2++)  {
      position[pos_1][pos_2] = pos;
      position[pos_2][pos_1] = pos;
      pos++;  
    }
  }

  int j, jnum, jtype, type_comb, k;
  double Rx_ij, Ry_ij, Rz_ij, rsq, Rx_ik, Ry_ik, Rz_ik, rsq1;
  double Rx_jk, Ry_jk, Rz_jk, rsq2, cos_theta, aux, G4;
  double function, function1, function2;

  // The fingerprints are calculated for each atom i in the initial data

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    
    if (mask[i] & groupbit) {

      // First neighborlist for atom i
      const int* const jlist = firstneigh[i];
      jnum = numneigh[i];

      /* ------------------------------------------------------------------------------------------------------------------------------- */
      for (int jj = 0; jj < jnum; jj++) {            // Loop for the first neighbor j
        j = jlist[jj];
        j &= NEIGHMASK;
        // Element type of atom j. Rij calculation.
        Rx_ij = x[j][0] - x[i][0];
        Ry_ij = x[j][1] - x[i][1];
        Rz_ij = x[j][2] - x[i][2];
        rsq = Rx_ij*Rx_ij + Ry_ij*Ry_ij + Rz_ij*Rz_ij;
        jtype = type[j];

        // Cutoff function Fc(Rij) calculation
        if (rsq < cutsq && rsq>1e-20) {    
          function = 0.5*(cos(sqrt(rsq/cutsq)*pi)+1);

          // G1 fingerprints calculation: sum Fc(Rij)
          fingerprints_atom[jtype-1] += function;

          if (g2_flag == 1) {
            // The number of G2 fingerprints depend on the number of given eta_G2 parameters
            for (int m = 0; m < n_etaG2; m++)  {
              fingerprints_atom[ntypes+m*ntypes+jtype-1] += exp(-eta_G2[m]*rsq)*function; // G2 fingerprints calculation 
            }
          }

          /* ------------------------------------------------------------------------------------------------------------------------------- */
          if (g4_flag == 1) {
            for (int kk = 0; kk < jnum; kk++) {            // Loop for the second neighbor k
              k = jlist[kk];
              k &= NEIGHMASK;

              // Rik (rsq1) and Rjk (rsq2) calculation. G2 fingerprints and derivatives are only calculated if Rik<Rc and Rjk<Rc
              Rx_ik = x[k][0] - x[i][0];
              Ry_ik = x[k][1] - x[i][1];
              Rz_ik = x[k][2] - x[i][2];
              rsq1 = Rx_ik*Rx_ik + Ry_ik*Ry_ik + Rz_ik*Rz_ik;
              Rx_jk = x[k][0] - x[j][0];
              Ry_jk = x[k][1] - x[j][1];
              Rz_jk = x[k][2] - x[j][2];
              rsq2 = Rx_jk*Rx_jk + Ry_jk*Ry_jk + Rz_jk*Rz_jk;
              cos_theta = (rsq+rsq1-rsq2)/(2*sqrt(rsq*rsq1));        // cos(theta)
              type_comb = position[jtype-1][type[k]-1];

              if (rsq1 < cutsq && rsq1>1e-20 && rsq2 < cutsq && rsq2>1e-20) {
                function1 = 0.5*(cos(sqrt(rsq1/cutsq)*pi)+1);        // fc(Rik)
                function2 = 0.5*(cos(sqrt(rsq2/cutsq)*pi)+1);        // fc(Rjk)

                // The number of G4 fingerprints depend on the number of given parameters
                for (int h = 0; h < n_lambda; h++)  {
                  aux = 1+(lambda[h]*cos_theta);
		  if (aux < 0)
		    aux = 0;
		  
		  for (int l = 0; l < n_zeta; l++)  {
		    for (int q = 0; q < n_etaG4; q++) {
		      G4 = pow(2,1-zeta[l])*pow(aux,zeta[l])*exp(-eta_G4[q]*(rsq+rsq1+rsq2))*function*function1*function2;
		      if (kk > jj)   fingerprints_atom[ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))] += G4;
		    }
		  }
                  
                }
              }          
            }
          }
        }
      }
      
      // Writing the fingerprnts vector in the fingerprints matrix
      for(int n = 0; n < size_peratom_cols; n++) {
        fingerprints[i][n] = fingerprints_atom[n];
        fingerprints_atom[n] = 0.0;
      }
    } 
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputeFingerprints::memory_usage()
{
  double bytes = (size_peratom_cols*nmax_atom) * sizeof(double);
  return bytes;
}

