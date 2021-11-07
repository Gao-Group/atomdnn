
#ifdef COMPUTE_CLASS

ComputeStyle(derivatives,ComputeDerivatives)

#else

#ifndef COMPUTE_DERIVATIVES_H
#define COMPUTE_DERIVATIVES_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeDerivatives : public Compute {
 public: 
  ComputeDerivatives(class LAMMPS *, int, char **);
  ~ComputeDerivatives();
  void init();
  void init_list(int, class NeighList *);
  void compute_local();
  double memory_usage();

 private:
  double cutsq;
  double *eta_G2;
  double *zeta;
  double *eta_G4;
  int *lambda;
  int n_etaG2;
  int n_etaG4;
  int n_zeta;
  int n_lambda;
  int g2_flag;
  int g4_flag;

  class NeighList *list;
  double **alocal;
  int nmax_local;

  void compute_derivatives();
  void reallocate(int);
};

}

#endif
#endif

// compute ID group-ID derivatives Rc keyword values ... end
// Keyword options: etaG2, etaG4, zeta, lambda

