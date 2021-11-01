/* ------------------------------------------------------------------------- */

#include "compute_derivatives.h"
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

ComputeDerivatives::ComputeDerivatives(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), 
  cutsq(0.0), list(NULL), eta_G2(NULL), zeta(NULL), eta_G4(NULL), lambda(NULL), alocal(NULL)

{
  if (narg < 4) error->all(FLERR,"Illegal compute derivatives command");

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
      memory->create(eta_G2,n_etaG2,"derivatives:eta_G2");
      for(int c = 0; c < n_etaG2; c++)
        eta_G2[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"etaG4") == 0) {
      g4_1 = 1;
      n_etaG4 = 0;
      while (strcmp(arg[iarg+n_etaG4+1],"etaG2") && strcmp(arg[iarg+n_etaG4+1],"zeta") && strcmp(arg[iarg+n_etaG4+1],"lambda")!= 0 && strcmp(arg[iarg+n_etaG4+1],"end")!= 0)
        n_etaG4++;
      memory->create(eta_G4,n_etaG4,"derivatives:eta_G4");
      for(int c = 0; c < n_etaG4; c++)
        eta_G4[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"zeta") == 0) {
      g4_2 = 1;
      n_zeta = 0;
      while (strcmp(arg[iarg+n_zeta+1],"etaG2") && strcmp(arg[iarg+n_zeta+1],"etaG4") && strcmp(arg[iarg+n_zeta+1],"lambda")!= 0 && strcmp(arg[iarg+n_zeta+1],"end")!= 0)
        n_zeta++;
      memory->create(zeta,n_zeta,"derivatives:zeta");
      for(int c = 0; c < n_zeta; c++)
        zeta[c] = atof(arg[iarg+c+1]);
    } 
    else if (strcmp(arg[iarg],"lambda") == 0) {
      g4_3 = 1;
      n_lambda = 0;
      while (strcmp(arg[iarg+n_lambda+1],"zeta") && strcmp(arg[iarg+n_lambda+1],"etaG4") && strcmp(arg[iarg+n_lambda+1],"etaG2")!= 0 && strcmp(arg[iarg+n_lambda+1],"end")!= 0)
        n_lambda++;
      memory->create(lambda,n_lambda,"derivatives:lambda");
      for(int c = 0; c < n_lambda; c++)
        lambda[c] = atof(arg[iarg+c+1]);
    }
  }

  if (g4_1 && g4_2 && g4_3 == 1)  g4_flag = 1;

  int ntypes = atom->ntypes;
  int ntypes_combinations = ntypes*(ntypes+1)/2;
  int n_derivatives = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  size_local_cols = 3 + n_derivatives;

  nmax_local = 0;
  local_flag = 1;
}

/* ---------------------------------------------------------------------- */

ComputeDerivatives::~ComputeDerivatives()

{

  memory->destroy(alocal);
  memory->destroy(eta_G2);
  memory->destroy(eta_G4);
  memory->destroy(zeta);
  memory->destroy(lambda);

}

/* ---------------------------------------------------------------------- */

void ComputeDerivatives::init()

{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"derivatives") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute derivatives");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

}

/* ---------------------------------------------------------------------- */

void ComputeDerivatives::init_list(int /*id*/, NeighList *ptr)

{

  list = ptr;

}

/* ---------------------------------------------------------------------- */

void ComputeDerivatives::compute_local()
{
  invoked_local = update->ntimestep;
  neighbor->build_one(list);

  const int inum = list->inum;
  const int* const numneigh = list->numneigh;
  int total_list = std::accumulate(numneigh, numneigh+inum, 0);
  size_local_rows = 3*(inum+total_list);
  reallocate(size_local_rows);
  compute_derivatives();
}



void ComputeDerivatives::compute_derivatives()
{
  const int inum = list->inum;
  const int* const numneigh = list->numneigh;

  // Get initial atoms data and neighborlists
  const int* const ilist = list->ilist;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  int const ntypes = atom->ntypes;
  double** const x = atom->x;
  const int* const mask = atom->mask;
  int * const tag = atom->tag;
  double pi = 3.14159265358979323846;
  int count = 0;

  // Initialize derivatives and derivatives_i array
  double derivatives[3][size_local_cols];
  double derivatives_i[3][size_local_cols];


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
  double function, function1, function2, dfc, dfc1, dfc2;
  double ij_factor, ik_factor, jk_factor;

  // The derivatives are calculated for each atom i in the initial data
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mask[i] & groupbit) {

      // First neighborlist for atom i
      const int* const jlist = firstneigh[i];
      jnum = numneigh[i];

      derivatives_i[0][0] = i;
      derivatives_i[1][0] = tag[i];
      derivatives_i[2][0] = type[i];
	
      derivatives_i[0][1] = i;
      derivatives_i[1][1] = tag[i];
      derivatives_i[2][1] = type[i];
	
      derivatives_i[0][2] = x[i][0];
      derivatives_i[1][2] = x[i][1];
      derivatives_i[2][2] = x[i][2];

      for(int n = 3; n < size_local_cols; n++) {
	derivatives_i[0][n] = 0.0;
	derivatives_i[1][n] = 0.0;
	derivatives_i[2][n] = 0.0;
      }


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

	// Cutoff function Fc(Rij) and dFc(Rij) calculation
	if (rsq < cutsq && rsq>1e-20) { 
	  function = 0.5*(cos(sqrt(rsq/cutsq)*pi)+1);
	  dfc = -pi*0.5*sin(pi*sqrt(rsq/cutsq))/(sqrt(cutsq));

	  for(int n = 0; n < size_local_cols; n++) {
	    derivatives[0][n] = 0.0;
	    derivatives[1][n] = 0.0;
	    derivatives[2][n] = 0.0;
	  }

	  derivatives[0][0] = i;
	  derivatives[1][0] = tag[i];
	  derivatives[2][0] = type[i];
	    
	  derivatives[0][1] = j;
	  derivatives[1][1] = tag[j];
	  derivatives[2][1] = jtype;
	    
	  derivatives[0][2] = x[j][0];
	  derivatives[1][2] = x[j][1];
	  derivatives[2][2] = x[j][2];

	  // compute G1
	  derivatives[0][jtype+2] = dfc*Rx_ij;
	  derivatives[1][jtype+2] = dfc*Ry_ij;
	  derivatives[2][jtype+2] = dfc*Rz_ij;

	  derivatives_i[0][jtype+2] += -dfc*Rx_ij;
	  derivatives_i[1][jtype+2] += -dfc*Ry_ij;
	  derivatives_i[2][jtype+2] += -dfc*Rz_ij;

	  // compute G2
	  if (g2_flag == 1) {
	    // The number of G2 derivatives depend on the number of given eta_G2 parameters
	    for (int m = 0; m < n_etaG2; m++)  {
	      derivatives[0][ntypes+m*ntypes+jtype+2] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives in the x direction
	      derivatives[1][ntypes+m*ntypes+jtype+2] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives in the y direction
	      derivatives[2][ntypes+m*ntypes+jtype+2] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives in the z direction

	      derivatives_i[0][ntypes+m*ntypes+jtype+2] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives in the x direction
	      derivatives_i[1][ntypes+m*ntypes+jtype+2] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives in the y direction
	      derivatives_i[2][ntypes+m*ntypes+jtype+2] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives in the z direction 
	    }
	  }


	  // compute G4
	  if (g4_flag == 1) {
	    for (int kk = 0; kk < jnum; kk++) {            // Loop for the second neighbor k
	      k = jlist[kk];
	      k &= NEIGHMASK;

	      // Rik (rsq1) and Rjk (rsq2) calculation. G4 derivatives are only calculated if Rik<Rc and Rjk<Rc
	      Rx_ik = x[k][0] - x[i][0];
	      Ry_ik = x[k][1] - x[i][1];
	      Rz_ik = x[k][2] - x[i][2];
	      rsq1 = Rx_ik*Rx_ik + Ry_ik*Ry_ik + Rz_ik*Rz_ik;
	      Rx_jk = x[k][0] - x[j][0];
	      Ry_jk = x[k][1] - x[j][1];
	      Rz_jk = x[k][2] - x[j][2];
	      rsq2 = Rx_jk*Rx_jk + Ry_jk*Ry_jk + Rz_jk*Rz_jk;
	      cos_theta = (rsq+rsq1-rsq2)/(2*sqrt(rsq*rsq1));               // cos(theta)
	      type_comb = position[jtype-1][type[k]-1];

	      if (cos_theta < -1)  cos_theta = -1;
	      if (cos_theta > 1)   cos_theta = 1;

	      if (rsq1<cutsq && rsq1>1e-20 && rsq2<cutsq && rsq2>1e-20) {
		function1 = 0.5*(cos(sqrt(rsq1/cutsq)*pi)+1);               // fc(Rik)
		function2 = 0.5*(cos(sqrt(rsq2/cutsq)*pi)+1);               // fc(Rjk)
		dfc2 = -pi*0.5*sin(pi*sqrt(rsq2/cutsq))/(sqrt(cutsq));      // dFc(Rjk)
		dfc1 = -pi*0.5*sin(pi*sqrt(rsq1/cutsq))/(sqrt(cutsq));      // dFc(Rik)

		// The number of G4 derivatives depend on the number of given parameters
		for (int h = 0; h < n_lambda; h++)  {
		  aux = 1+(lambda[h]*cos_theta);
		  if (aux > 0)  {
		    for (int l = 0; l < n_zeta; l++)  {
		      for (int q = 0; q < n_etaG4; q++) {
			G4 = pow(2,1-zeta[l])*pow(aux,zeta[l])*exp(-eta_G4[q]*(rsq+rsq1+rsq2))*function*function1*function2;

			// Calculation of factors necessary for the derivatives of G4 with respect to atom j
			ij_factor = (1/sqrt(rsq*rsq1)-cos_theta/rsq)*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc/(sqrt(rsq)*function);
			ik_factor = (1/sqrt(rsq*rsq1)-cos_theta/rsq1)*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc1/(sqrt(rsq1)*function1);
			jk_factor = -(1/sqrt(rsq*rsq1))*lambda[h]*zeta[l]/aux-2*eta_G4[q]+dfc2/(sqrt(rsq2)*function2);

			// G4 derivatives calculation
			if ( kk != jj ) {
			  // G4 derivatives with respect to x, y, z directions
			  derivatives[0][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += G4*(ij_factor*Rx_ij-jk_factor*Rx_jk);
			  derivatives[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += G4*(ij_factor*Ry_ij-jk_factor*Ry_jk);
			  derivatives[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += G4*(ij_factor*Rz_ij-jk_factor*Rz_jk);
			}
			if ( kk > jj ) {
			  derivatives_i[0][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += -G4*(ij_factor*Rx_ij+ik_factor*Rx_ik);
			  derivatives_i[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += -G4*(ij_factor*Ry_ij+ik_factor*Ry_ik);
			  derivatives_i[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))+3]  += -G4*(ij_factor*Rz_ij+ik_factor*Rz_ik);
			}
		      }
		    }
		  }
		}
	      }          
	    }
	  }
	  // Writing the derivatives array in the array_atom matrix
	  for(int n = 0; n < size_local_cols; n++) {
	    alocal[0+count*3][n] = derivatives[0][n];
	    alocal[1+count*3][n] = derivatives[1][n];
	    alocal[2+count*3][n] = derivatives[2][n];
	  }
	  count++;
	}
      }
      // Writing the derivatives_i array in the array_atom matrix
      for(int n = 0; n < size_local_cols; n++) {
	alocal[0+count*3][n] = derivatives_i[0][n];
	alocal[1+count*3][n] = derivatives_i[1][n];
	alocal[2+count*3][n] = derivatives_i[2][n];
      }
      count++;
    } 
  }
}

void ComputeDerivatives::reallocate(int n)
{
  nmax_local = n;
  memory->destroy(alocal);
  memory->create(alocal,nmax_local,size_local_cols,"derivatives:alocal");
  array_local = alocal;
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputeDerivatives::memory_usage()
{
  double bytes = (size_local_cols*nmax_local) * sizeof(double);
  return bytes;
}

