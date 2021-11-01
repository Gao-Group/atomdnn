/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Wei Gao (University of Texas at San Antonio)
                         Daniela Posso (University of Texas at San Antonio)
------------------------------------------------------------------------- */

#include "pair_tfdnn.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "modify.h"
#include "compute.h"

#include "tensorflow/c/c_api.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <ctype.h>


using namespace std;
using namespace LAMMPS_NS;

#define MAXLINE 1000
#define MAXARGS 1000
#define MAXWORDS 100

/*---------------------------------------------------------------------------------------------
read the information of descriptor parameters and nerual network
---------------------------------------------------------------------------------------------*/
void PairTFDNN::read_file(char *filename)
{
  char line[MAXLINE];
  char *ptr;
  char *args[MAXWORDS];
  char keyword[MAXWORDS];
  char parameter[MAXWORDS];
  int eof=0;
  int n, nwords;
  
  n_etaG2 = 0;
  n_etaG4 = 0;
  n_zeta = 0;
  n_lambda = 0;
  g2_flag = 0;
  g4_flag = 0;

  MPI_Comm_rank(world,&me);
  
  FILE *fp = NULL;
  
  tf_model_dir = new char[strlen(filename)+1];
  strcpy(tf_model_dir, filename);

  n = strlen(filename) + strlen("/parameters") + 1;
  char *para_file = new char[n];
  strcpy(para_file, filename);
  strcat(para_file,"/parameters");


  // read potential file on root proc 0
  
  if (me == 0){
    fp = force->open_potential(para_file);
    if (fp == NULL){
      char str[500];
      snprintf(str, 128,"Cannot open TensorflowDNN potential file %s", filename);
      error->one(FLERR,str);
    }
  }
  delete [] para_file;
  while (1){

    // read line in root proc 0 and then broadcast the line
    if (me == 0){
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL){  
  	eof = 1;
  	fclose(fp);
      }
      else
  	n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

  
    if (!strncmp(line,"element",7)){
      //sscanf(line,"%s %d",keyword,&tf_nelement);
      tf_nelement = getwords(line,args,MAXARGS)-1;
      tf_element = new char*[tf_nelement];
      for (int i=0;i<tf_nelement;i++){
  	n = strlen(args[i+1])+1;
  	tf_element[i] = new char[n];
  	strcpy(tf_element[i],args[i+1]);
      }
    }

    else if (!strncmp(line,"input",5)){     
      sscanf(line,"%s %d",keyword,&tf_input_number);
      tf_input_tag = new char*[tf_input_number];
      tf_input_tensor = new char* [tf_input_number];
      for (int i=0;i<tf_input_number;i++){
	
  	// read line on root proc 0 and then broadcast the line
  	if (me == 0){
  	  utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
  	  n = strlen(line) + 1;
  	}
  	MPI_Bcast(&n,1,MPI_INT,0,world);
  	MPI_Bcast(line,n,MPI_CHAR,0,world);

  	// extract tf input tensor tag and names
  	sscanf(line,"%s %*s",parameter);
  	n = strlen(parameter)+1;
  	tf_input_tag[i] = new char[n];
  	strcpy(tf_input_tag[i],parameter);
  	sscanf(line,"%*s %s",parameter);
  	n  = strlen(parameter)+1;
  	tf_input_tensor[i] = new char[n];
  	strcpy(tf_input_tensor[i],parameter);	
      }
    }

    else if (!strncmp(line,"output",6)){     
      sscanf(line,"%s %d",keyword,&tf_output_number);
      tf_output_tag = new char*[tf_output_number];
      tf_output_tensor = new char*[tf_output_number];
      for (int i=0;i<tf_output_number;i++){

  	// read line on root proc 0 and then broadcast the line
  	if (me == 0){
  	  utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
  	  n = strlen(line) + 1;
  	}
  	MPI_Bcast(&n,1,MPI_INT,0,world);
  	MPI_Bcast(line,n,MPI_CHAR,0,world);

  	// extract tf output tensor tag and names
  	sscanf(line,"%s %*s",parameter);
  	n = strlen(parameter) + 1;
  	tf_output_tag[i] = new char[n];
  	strcpy(tf_output_tag[i],parameter);
  	sscanf(line,"%*s %s",parameter);
  	n = strlen(parameter) + 1;
  	tf_output_tensor[i] = new char[n];
  	strcpy(tf_output_tensor[i],parameter);
      } 
    }

    else if (!strncmp(line,"cutoff",6))
      sscanf(line,"%s %lf",keyword,&cut_global);
    
    else if (!strncmp(line,"descriptor",10)){
      sscanf(line,"%s %s %d",keyword,parameter,&n_parameter);
      n = strlen(parameter)+1;
      descriptor = new char[n];
      strcpy(descriptor,parameter);
      if (!strncmp(descriptor,"acsf",4)){
  	for (int ip=0;ip<n_parameter;ip++){

  	  // read line on root proc 0 and then broadcast the line
  	  if (me == 0){
  	    utils::sfgets(FLERR,line,MAXLINE,fp,filename,error);
  	    n = strlen(line) + 1;
  	  }
  	  MPI_Bcast(&n,1,MPI_INT,0,world);
  	  MPI_Bcast(line,n,MPI_CHAR,0,world);

  	  // extract descriptor parameters
  	  if (!strncmp(line,"etaG2",5)){
  	    n_etaG2 = getwords(line,args,MAXARGS) -1;
  	    eta_G2 = new double[n_etaG2];
  	    for (int i=0;i<n_etaG2;i++)
  	      eta_G2[i] = strtod(args[i+1],NULL);
  	  }
  	  else if (!strncmp(line,"etaG4",5)){
  	    n_etaG4 = getwords(line,args,MAXARGS) -1;
  	    eta_G4 = new double[n_etaG4];
  	    for (int i=0;i<n_etaG4;i++)
  	      eta_G4[i] = strtod(args[i+1],NULL);
  	  }
  	  else if (!strncmp(line,"zeta",4)){
  	    n_zeta = getwords(line,args,MAXARGS) -1;
  	    zeta = new double[n_zeta];
  	    for (int i=0;i<n_zeta;i++)
  	      zeta[i] = strtod(args[i+1],NULL);
  	  }
  	  else if (!strncmp(line,"lambda",6)){
  	    n_lambda = getwords(line,args,MAXARGS) -1;
  	    lambda = new double[n_lambda];
  	    for (int i=0;i<n_lambda;i++)
  	      lambda[i] = strtod(args[i+1],NULL);
  	  }
  	}
      } 
    } // end of reading descriptor information      
  } // end of while
 
    // check reading error and set flag
  if (!strncmp(descriptor,"acsf",4)){
    if (n_etaG2 == 0)
      error->all(FLERR,"Need to set eta of G2 parameters of acsf");
    else
      g2_flag = 1;
    if (n_etaG4 == 0)
      error->all(FLERR,"Need to set eta of G4 parameters of acsf");
    else if (n_zeta == 0)
      error->all(FLERR,"Need to set zeta of G4 parameters of acsf");
    else if (n_lambda == 0)
      error->all(FLERR,"Need to set lambda of G4 parameters of acsf");
    else
      g4_flag = 1;
  }    
}

/* ---------------------------------------------------------------------- */

PairTFDNN::PairTFDNN(LAMMPS *lmp) : Pair(lmp), fingerprints(NULL), dgdr(NULL), center_atom_id(NULL), neighbor_atom_coord(NULL), atom_elements(NULL),neighbor_atom_id(NULL)
{
  single_enable = 0;
  restartinfo = 0;
  // one_coeff = 1;
  no_virial_fdotr_compute = 1;
  manybody_flag = 1;
  num_der_pairs = 0; // number of derivative pairs
  fp_nrows = 0; // number of rows of fingerprints array  
}

/* ---------------------------------------------------------------------- */

PairTFDNN ::~PairTFDNN()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }

  delete [] tf_model_dir;

  if (tf_element)
    for (int i=0;i<tf_nelement;i++) delete [] tf_element[i];
  delete [] tf_element;

  delete [] descriptor;
  delete [] eta_G2;
  delete [] eta_G4;
  delete [] zeta;
  delete [] lambda;


  if (tf_input_tensor)
    for (int i=0;i<tf_input_number;i++) delete [] tf_input_tensor[i];
  delete [] tf_input_tensor;

  if (tf_input_tag)
    for (int i=0;i<tf_input_number;i++) delete [] tf_input_tag[i];
  delete [] tf_input_tag;

  if (tf_output_tag)
    for (int i=0;i<tf_output_number;i++) delete [] tf_output_tag[i];
  delete [] tf_output_tag;

  if (tf_output_tensor)
    for (int i=0;i<tf_output_number;i++) delete [] tf_output_tensor[i];
  delete [] tf_output_tensor;

  
  
  memory->sfree(atom_elements);
  memory->sfree(fingerprints);

  memory->sfree(dgdr);
  memory->sfree(center_atom_id);
  memory->sfree(neighbor_atom_id);
  memory->sfree(neighbor_atom_coord);
  
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);
}

/* ---------------------------------------------------------------------- */

void PairTFDNN::coeff(int narg, char **arg)
{
  int n = atom->ntypes;
  
  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
  
  read_file(arg[2]);

  int element_count = 1; // count the unique atom type
  int element_check = 0; // check if the element is set correctly 
  map[0] = -1; // first element of map is not used
  for (int i=0;i<n;i++){
    for (int j=0;j<tf_nelement;j++){
      if (strcasecmp(arg[3+i],tf_element[j]) == 0){
	map[i+1] = j+1;
	element_check=1;
      }
    }
    if (element_check==1)
      element_check=0;
    else
      error->all(FLERR,"Incorrect args for pair coefficients");
    if (i>1 & map[i]!=map[i+1])
      element_count++;
  }

  if (element_count!=tf_nelement)
    error->all(FLERR,"Incorrect args for pair coefficients");
  

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}
/* ---------------------------------------------------------------------- */
void PairTFDNN::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag,n+1,n+1,"pair:setflag");
  cutsq = memory->create(cutsq,n+1,n+1,"pair:cutsq");

  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  
  map = new int[n+1];
  /*
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cutghost,n+1,n+1,"pair:cutghost"); 
  */
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTFDNN::settings(int narg, char ** /* arg */)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTFDNN::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style tfdnn requires atom IDs");
  
  if (force->newton_pair != 1)
    error->all(FLERR,"Pair style tfdnn requires newton pair on");

  // Initialise neighbor list, including neighbors of ghosts
  int irequest_full = neighbor->request(this);
  neighbor->requests[irequest_full]->id = 1;
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  // neighbor->requests[irequest_full]->ghost = 1;
  neighbor->requests[irequest_full]->occasional = 1;
  
  // create tensorflow model
  create_tensorflow_model();
  
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTFDNN::init_one(int /*i*/, int /*j*/)
{
  return cut_global;
}


/*-----------------------------------------------------------------------*/

void PairTFDNN::NoOpDeallocator(void* data, size_t a, void* b) {}


void PairTFDNN::create_tensorflow_model()
{
  Graph = TF_NewGraph();
  Status = TF_NewStatus();
  SessionOpts = TF_NewSessionOptions();
  RunOpts = NULL;
  ntags = 1;
  const char *tf_model_tags = "serve";

  Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, tf_model_dir, &tf_model_tags, ntags, Graph, NULL, Status);

  if(TF_GetCode(Status) != TF_OK)
    error->all(FLERR,"Failed Session creation.");

  // get the tensorflow model input and output tensors
  char *tensor_name, *tag;
  int tag_int;
  
  Input = (TF_Output*)malloc(sizeof(TF_Output) * tf_input_number);

  //Input[0] = {TF_GraphOperationByName(Graph, "serving_default_atom_type"), 0};
  //Input[1] = {TF_GraphOperationByName(Graph, "serving_default_fingerprints"),0};
 
  for (int i=0;i<tf_input_number;i++){
    tensor_name = strtok(tf_input_tensor[i], ":");
    tag = strtok(NULL, ":");
    tag_int = atoi(tag);
    Input[i] = {TF_GraphOperationByName(Graph, tensor_name), tag_int};
    if(Input[i].oper == NULL)
      error->all(FLERR,"Failed load tensorflow inputs by names.");      
  }

  Output = (TF_Output*)malloc(sizeof(TF_Output) * tf_output_number);
  //Output[0] = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"),0};
  
  for (int i=0;i<tf_output_number;i++){
    tensor_name = strtok(tf_output_tensor[i], ":");
    tag = strtok(NULL, ":");
    tag_int = atoi(tag);
    Output[i] = {TF_GraphOperationByName(Graph, tensor_name), tag_int};
    if(Output[i].oper == NULL)
      error->all(FLERR,"Failed load tensorflow outputs by names.");      
  }

  // allocate input and output tensors for prediction
  InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*tf_input_number);
  OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*tf_output_number);

}


/*-----------------------------------------------------------------------*/

void PairTFDNN::compute_fingerprints()
{
  const int inum = list->inum;
  const int* const ilist = list->ilist;
  const int* const numneigh = list->numneigh;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  int const ntypes = atom->ntypes;
  double** const x = atom->x;
  const int* const mask = atom->mask;
  double pi = 3.14159265358979323846;
  double cutoffsq = cut_global*cut_global;
  int ntypes_combinations = ntypes*(ntypes+1)/2;
  n_fpt = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  
  // Initialize fingerprnts vector per atom
  double fingerprints_atom[n_fpt];
  for (int i=0;i<n_fpt;i++)
    fingerprints_atom[i] = 0;

  // allocate fingerprints array, size of the array may increase at different timestep
  if (inum > fp_nrows) {
    memory->sfree(fingerprints);
    memory->sfree(atom_elements);
    fp_nrows = inum;
    fingerprints = (float_type *) memory->smalloc(fp_nrows*n_fpt*sizeof(float_type),"PairTFDNN:fingerprints");
    atom_elements = (int *) memory->smalloc(fp_nrows * sizeof(int),"PairTFDNN:atom_elements");
  }

  //
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
    if (mask[i]) {
      
      atom_elements[i] = map[type[i]];
      
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
        if (rsq < cutoffsq && rsq>1e-20) {    
          function = 0.5*(cos(sqrt(rsq/cutoffsq)*pi)+1);

          // G1 fingerprints calculation: sum Fc(Rij)
	  fingerprints_atom[jtype-1] += function;
	  
          if (g2_flag == 1) {
            // The number of G2 fingerprints depend on the number of given eta_G2 parameters
            for (int m = 0; m < n_etaG2; m++)  {
	      fingerprints_atom[ntypes+m*ntypes+jtype-1] += exp(-eta_G2[m]*rsq)*function;     // G2 fingerprints calculation
            }
          }


	  // G4 calculation
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

              if (rsq1 < cutoffsq && rsq1>1e-20 && rsq2 < cutoffsq && rsq2>1e-20) {
                function1 = 0.5*(cos(sqrt(rsq1/cutoffsq)*pi)+1);        // fc(Rik)
                function2 = 0.5*(cos(sqrt(rsq2/cutoffsq)*pi)+1);        // fc(Rjk)

                // The number of G4 fingerprints depend on the number of given parameters
                for (int h = 0; h < n_lambda; h++)  {
                  aux = 1+(lambda[h]*cos_theta);
		  if (aux > 0){
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
      }
      // Writing The fingerprnts_atom vector in the fingerprints matrix
      for(int n = 0; n < n_fpt; n++) {
	fingerprints[i*n_fpt+n]=fingerprints_atom[n];
        fingerprints_atom[n] = 0.0;
      }
    } 
  }
  
}

//-------------------------------------------------------------
void PairTFDNN::compute_derivatives()
{
  const int inum = list->inum;  // # of I atoms neighbors are stored for, which is usually equals to atom->local, but When using pair_style hybrid,
                                // neighbor lists can be for subsets of all the atoms owned by a proc.
  
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

  double cutoffsq = cut_global*cut_global;

  int ntypes_combinations = ntypes*(ntypes+1)/2;

  n_der = n_etaG2*ntypes*g2_flag + n_lambda*n_zeta*n_etaG4*ntypes_combinations*g4_flag + ntypes;
  
  int total_neigh_pair = accumulate(numneigh, numneigh+inum, 0); // total number of neighbhor pairs 
  int count = 0; //count the neighbor pair
 
  // allocate fingerprints derivatives related arrays, reallocate when num_der_pairs increases
  if (inum + total_neigh_pair > num_der_pairs) {
    memory->sfree(dgdr);
    memory->sfree(center_atom_id);
    memory->sfree(neighbor_atom_coord);
    memory->sfree(neighbor_atom_id);
    
    num_der_pairs = inum + total_neigh_pair;  // total number of derivative pairs, including derivatives to neighbors and to selfs 

    dgdr = (float_type *)memory->smalloc(3 * num_der_pairs * n_der * sizeof(float_type),"PairTFDNN:dgdr");
    center_atom_id = (int *)memory->smalloc(num_der_pairs * sizeof(int),"PairTFDNN:center_atom_id");
    neighbor_atom_coord = (float_type *)memory->smalloc(num_der_pairs * 3 * sizeof(float_type),"PairTFDNN:neighbor_atom_coord");
    neighbor_atom_id = (int *)memory->smalloc(num_der_pairs * sizeof(int),"PairTFDNN:neighbor_atom_id");
  }
  
  
  // define derivatives (w.r.t other atoms) and derivatives_i (w.r.t itself) array
  double derivatives[3][n_der];
  double derivatives_i[3][n_der];


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
    if (mask[i]) {

      // First neighborlist for atom i
      const int* const jlist = firstneigh[i];
      jnum = numneigh[i];

      for(int n = 0; n < n_der; n++) {
	derivatives_i[0][n] = 0.0;
	derivatives_i[1][n] = 0.0;
	derivatives_i[2][n] = 0.0;
      }

      // Loop for the first neighbor j
      for (int jj = 0; jj < jnum; jj++) {            
	j = jlist[jj];
	j &= NEIGHMASK;

	// Element type of atom j. Rij calculation.
	Rx_ij = x[j][0] - x[i][0];
	Ry_ij = x[j][1] - x[i][1];
	Rz_ij = x[j][2] - x[i][2];
	rsq = Rx_ij*Rx_ij + Ry_ij*Ry_ij + Rz_ij*Rz_ij;
	jtype = type[j];

	// Cutoff function Fc(Rij) and dFc(Rij) calculation
	if (rsq < cutoffsq && rsq>1e-20) { 
	  function = 0.5*(cos(sqrt(rsq/cutoffsq)*pi)+1);
	  dfc = -pi*0.5*sin(pi*sqrt(rsq/cutoffsq))/(sqrt(cutoffsq));

	  for(int n = 1; n < n_der; n++) {
              derivatives[0][n] = 0.0;
              derivatives[1][n] = 0.0;
              derivatives[2][n] = 0.0;
            }
	  
	  derivatives[0][jtype-1] = dfc*Rx_ij;
	  derivatives[1][jtype-1] = dfc*Ry_ij;
	  derivatives[2][jtype-1] = dfc*Rz_ij;
	  
	  derivatives_i[0][jtype-1] += -dfc*Rx_ij;
	  derivatives_i[1][jtype-1] += -dfc*Ry_ij;
	  derivatives_i[2][jtype-1] += -dfc*Rz_ij;
	  
	  if (g2_flag == 1) {
	    // The number of G2 derivatives depend on the number of given eta_G2 parameters
	    for (int m = 0; m < n_etaG2; m++)  {
	      derivatives[0][ntypes+m*ntypes+jtype-1] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives in the x direction
	      derivatives[1][ntypes+m*ntypes+jtype-1] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives in the y direction
	      derivatives[2][ntypes+m*ntypes+jtype-1] = exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives in the z direction

	      derivatives_i[0][ntypes+m*ntypes+jtype-1] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rx_ij; // G2 derivatives (w.r.t itself) in the x direction
	      derivatives_i[1][ntypes+m*ntypes+jtype-1] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Ry_ij; // G2 derivatives (w.r.t itself) in the y direction
	      derivatives_i[2][ntypes+m*ntypes+jtype-1] += -exp(-eta_G2[m]*rsq)*(dfc/sqrt(rsq)-2*eta_G2[m]*function)*Rz_ij; // G2 derivatives (w.r.t itself) in the z direction 
	    }
	  }

	  /* ------------------------------------------------------------------------------------------------------------------------------- */
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

	      if (rsq1 < cutoffsq && rsq1>1e-20 && rsq2 < cutoffsq && rsq2>1e-20) {
		function1 = 0.5*(cos(sqrt(rsq1/cutoffsq)*pi)+1);               // fc(Rik)
		function2 = 0.5*(cos(sqrt(rsq2/cutoffsq)*pi)+1);               // fc(Rjk)
		dfc2 = -pi*0.5*sin(pi*sqrt(rsq2/cutoffsq))/(sqrt(cutoffsq));      // dFc(Rjk)
		dfc1 = -pi*0.5*sin(pi*sqrt(rsq1/cutoffsq))/(sqrt(cutoffsq));      // dFc(Rik)
		
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
			  derivatives[0][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += G4*(ij_factor*Rx_ij-jk_factor*Rx_jk);
			  derivatives[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += G4*(ij_factor*Ry_ij-jk_factor*Ry_jk);
			  derivatives[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += G4*(ij_factor*Rz_ij-jk_factor*Rz_jk);
			}
			if ( kk > jj ) {
			  derivatives_i[0][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += -G4*(ij_factor*Rx_ij+ik_factor*Rx_ik);
			  derivatives_i[1][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += -G4*(ij_factor*Ry_ij+ik_factor*Ry_ik);
			  derivatives_i[2][ntypes+n_etaG2*ntypes+h+n_lambda*(l+n_zeta*(q+n_etaG4*type_comb))]  += -G4*(ij_factor*Rz_ij+ik_factor*Rz_ik);
			}
		      }
		    }
		  }
		}
	      }          
	    } 
	  } // end loop of k

	  // Writing the derivatives array, for derivative of atom i w.r.t. atom j
	  for (int m = 0; m < 3; m++){
	    neighbor_atom_coord[count*3+m] = x[j][m];
	    for(int n = 0; n < n_der; n++)
	      dgdr[(n_der*count+n)*3+m] = derivatives[m][n];
	  }	  
	  center_atom_id[count] = i;
	  neighbor_atom_id[count] = j;
	  count++;	  
	} 
      } // Loop of j

      // Writing the derivatives_i array in the array_atom matrix, for derivative of atom i w.r.t. atom i itself
      for (int m = 0; m < 3; m++){	
	neighbor_atom_coord[count*3+m] = x[i][m];
	for(int n = 0; n < n_der; n++)
	  dgdr[(n_der*count+n)*3+m] = derivatives_i[m][n];
      }      
      center_atom_id[count] = i;
      neighbor_atom_id[count] = i;	  
      count++;
    } 
  }
}


/* ---------------------------------------------------------------------- */

void PairTFDNN::compute(int eflag, int vflag)
{
  MPI_Comm_rank(world,&me);

  ev_init(eflag,vflag);
  
  compute_fingerprints();
  compute_derivatives();
  
  // atom element type input dimensions
  int ndims_elements = 2;
  int64_t dims_elements[2] = {1,fp_nrows};
  int ndata_elements = sizeof(int)*dims_elements[1];

  // fingerprints input dimensions
  int ndims_fp = 3;
  int64_t dims_fp[3] = {1, fp_nrows, n_fpt};
  int ndata_fp = sizeof(float_type)*dims_fp[1]*dims_fp[2]; 

  // center_atom_id input dimensions
  int ndims_center_id = 2;
  int64_t dims_center_id[2] = {1, num_der_pairs};
  int ndata_center_id = sizeof(int)*dims_center_id[1];

  // neighbor_atom_id input dimensions
  int ndims_neighbor_id = 2;
  int64_t dims_neighbor_id[2] = {1, num_der_pairs};
  int ndata_neighbor_id = sizeof(int)*dims_neighbor_id[1];

  // dGdr input dimensions
  int ndims_dgdr = 4;
  int64_t dims_dgdr[4] = {1,num_der_pairs,n_der,3};
  int ndata_dgdr = sizeof(float_type)*dims_dgdr[1]*dims_dgdr[2]*dims_dgdr[3];

  // neighbor_atom_coord input dimensions
  int ndims_neighbor_coord = 4;
  int64_t dims_neighbor_coord[4] = {1,num_der_pairs,3,1};
  int ndata_neighbor_coord = sizeof(float_type)*dims_neighbor_coord[1]*dims_neighbor_coord[2];

  
  for (int i=0;i<tf_input_number;i++){
    if (!strncmp(tf_input_tag[i],"fingerprints",12))
      InputValues[i] = TF_NewTensor(tf_float_type, dims_fp, ndims_fp, fingerprints, ndata_fp, &NoOpDeallocator, 0);
    else if (!strncmp(tf_input_tag[i],"atom_type",9))
      InputValues[i] = TF_NewTensor(TF_INT32, dims_elements, ndims_elements, atom_elements, ndata_elements, &NoOpDeallocator, 0);
    else if (!strncmp(tf_input_tag[i],"center_atom_id",14))
      InputValues[i] = TF_NewTensor(TF_INT32, dims_center_id, ndims_center_id, center_atom_id, ndata_center_id, &NoOpDeallocator, 0);
    else if (!strncmp(tf_input_tag[i],"neighbor_atom_id",16))
       InputValues[i] = TF_NewTensor(TF_INT32, dims_neighbor_id, ndims_neighbor_id, neighbor_atom_id, ndata_neighbor_id, &NoOpDeallocator, 0);
    else if (!strncmp(tf_input_tag[i],"dgdr",4))
      InputValues[i] = TF_NewTensor(tf_float_type, dims_dgdr, ndims_dgdr, dgdr, ndata_dgdr, &NoOpDeallocator, 0);
    else if (!strncmp(tf_input_tag[i],"neighbor_atom_coord",19))
      InputValues[i] = TF_NewTensor(tf_float_type, dims_neighbor_coord, ndims_neighbor_coord, neighbor_atom_coord, ndata_neighbor_coord, &NoOpDeallocator, 0);
  }
    
  // Run the tensorflow Session for prediction
  TF_SessionRun(Session, NULL, Input, InputValues, tf_input_number, Output, OutputValues, tf_output_number, NULL, 0,NULL , Status);

  if(TF_GetCode(Status) != TF_OK)
    error->all(FLERR,"Failed TF_SessionRun.");
    

  void* buff;
  float_type* model_output;
  double **f = atom->f; 
  const int* const ilist = list->ilist;

  for (int i=0;i<tf_output_number;i++){
    if (!strncmp(tf_output_tag[i],"atom_pe",2)){
      buff = TF_TensorData(OutputValues[i]);
      model_output = (float_type*)buff;

      if (eflag_either){
	if (eflag_atom)
	  for (int j=0;j<fp_nrows;j++)
	    eatom[ilist[j]] += model_output[j];
      
	if (eflag_global){
	  for (int j=0;j<fp_nrows;j++)
	    eng_vdwl += model_output[j];
	}
      }
    }
    
    if (!strncmp(tf_output_tag[i],"force",5)){
      buff = TF_TensorData(OutputValues[i]);
      model_output = (float_type*)buff;
      int neighid;
      for (int j=0;j<num_der_pairs;j++){
	neighid = neighbor_atom_id[j];
	for (int k=0;k<3;k++)
	  f[neighid][k] += model_output[j*3+k];
      }
    }

    if (!strncmp(tf_output_tag[i],"stress",6)){
      buff = TF_TensorData(OutputValues[i]);
      model_output = (float_type*)buff;
      
      if (vflag_either){
	if (vflag_atom){
	  int centerid;
	  for (int j=0;j<num_der_pairs;j++){
	    centerid = center_atom_id[j];
	    vatom[centerid][0] += model_output[j*9];
	    vatom[centerid][1] += model_output[j*9+4];
	    vatom[centerid][2] += model_output[j*9+8];
	    vatom[centerid][3] += model_output[j*9+1];
	    vatom[centerid][4] += model_output[j*9+2];
	    vatom[centerid][5] += model_output[j*9+5];
	  }
	}
	
	if (vflag_global){
	  for (int j=0;j<num_der_pairs;j++){
	    virial[0] += model_output[j*9];
	    virial[1] += model_output[j*9+4];
	    virial[2] += model_output[j*9+8];
	    virial[3] += model_output[j*9+1];
	    virial[4] += model_output[j*9+2];
	    virial[5] += model_output[j*9+5];
	  }
	}
      } 
    }  
  } 
}
/*---------------------------------------------------------------*/
int PairTFDNN::getwords(char *line, char *words[], int maxwords)
{
  char *p = line;
  int nwords = 0; 
  while(1){
    while(isspace(*p))
      p++;
    if(*p == '\0')
      return nwords;
    words[nwords++] = p;
    while(!isspace(*p) && *p != '\0')
      p++;
    if(*p == '\0')
      return nwords;
    *p++ = '\0';
    if(nwords >= maxwords)
      return nwords;
  }
}

/*--------------------------------------------------------------*/
int strcasecmp(const char *s1, const char *s2) {
    const unsigned char *us1 = (const u_char *)s1,
                        *us2 = (const u_char *)s2;

    while (tolower(*us1) == tolower(*us2++))
        if (*us1++ == '\0')
            return (0);
    return (tolower(*us1) - tolower(*--us2));
}
