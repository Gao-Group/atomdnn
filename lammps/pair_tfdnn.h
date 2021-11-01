/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(tfdnn,PairTFDNN)

#else

#ifndef LMP_PAIR_TFDNN_H
#define LMP_PAIR_TFDNN_H


#define float_type double // or float
#define tf_float_type TF_DOUBLE // or TF_FLOAT

#include "pair.h"
#include "tensorflow/c/c_api.h"

namespace LAMMPS_NS {

class PairTFDNN : public Pair {
  public:
  PairTFDNN(class LAMMPS *);
  ~PairTFDNN();
  void read_file(char *);
  void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  static void NoOpDeallocator(void* , size_t , void*);
  void create_tensorflow_model();
  void compute_fingerprints();
  void compute_derivatives();
  int getwords(char*, char**, int);
 

private:
  char *descriptor;           // type of descriptor, e.g. ACSF
  double *eta_G2;             // eta parameters for ACSF G2 descriptor 
  double *zeta;               // zeta parameters for ACSF G4 descriptor 
  double *eta_G4;             // eta parameters for ACSF G4 descriptor 
  double *lambda;             // lambda parameters for ACSF G4 descriptor 
  int n_parameter;            // number of parameters for descriptor, = 4 for ACSF
  int n_etaG2;                // number of eta parameters for ACSF G2 descriptor
  int n_etaG4;                // number of eta parameters for ACSF G4 descriptor
  int n_zeta;                 // number of zeta parameters for ACSF G4 descriptor
  int n_lambda;               // number of lambda parameters for ACSF G4 descriptor
  int maxnum_blocks = 200;

  // double eta_G2[4] = {0.01,0.1,0.3,1};             // eta parameters for ACSF G2 descriptor 
  // double zeta[3] = {1.0,2.0,4.0};               // zeta parameters for ACSF G4 descriptor 
  // double eta_G4[1] = {0.01};             // eta parameters for ACSF G4 descriptor 
  // double lambda[2] = {1.0,-1.0};             // lambda parameters for ACSF G4 descriptor 
  // int n_parameter = 4;            // number of parameters for descriptor, = 4 for ACSF
  // int n_etaG2 = 4;                // number of eta parameters for ACSF G2 descriptor
  // int n_etaG4 = 1;                // number of eta parameters for ACSF G4 descriptor
  // int n_zeta = 3;                 // number of zeta parameters for ACSF G4 descriptor
  // int n_lambda = 2;               // number of lambda parameters for ACSF G4 descriptor
  
  int g2_flag;                // = 1, if ACSF G2 descriptor is set
  int g4_flag;                // = 1, if ACSF G4 descriptor is set

  int tf_nelement;               // number of elements defined in potential 
  char **tf_element;             // names of elements defined in potential  
  int *map;                      // mapping from atom types to elements 

  // int tf_nelement = 1;
  // char tf_element[1] = {'C'};
  // int map[2] = {1,1};

  float_type *fingerprints;          // fingerprints pointer, 1D storage for 2D array
  int fp_nrows;              // rows of fingerprints array, = number of local atoms described by PairTFDNN

  int n_fpt; // number of fingerprints
  int n_der; // number of fingerprints derivatives
  int num_der_pairs; // total number of derivative pairs, including derivatives to neighbors and to selfs 
  
  int* atom_elements;
  
  float_type *dgdr;    // fignerprints derivatives, 1D storage for 2D array
  float_type *neighbor_atom_coord;
  int *center_atom_id, *neighbor_atom_id;



 
  // float *data; // define vector for fingerprints

  // class NeighList *list;

protected:
  double cut_global;
  int tf_input_number;
  int tf_output_number;
  int me;
  
  char *tf_model_dir;
  char **tf_input_tensor,**tf_output_tensor,**tf_output_tag,**tf_input_tag;
  TF_Graph *Graph;
  TF_Output *Input,*Output;
  TF_Status *Status;
  TF_SessionOptions *SessionOpts;
  TF_Buffer *RunOpts;
  int ntags;
  TF_Session *Session;
  TF_Tensor **InputValues, **OutputValues;
  
  virtual void allocate();
};
  
}

#endif
#endif
  
