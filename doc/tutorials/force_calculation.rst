.. _force-reference-label:

============================
Force and stress calculation
============================


Theory
------

Consider a material containing :math:`N` atoms has a total potential energy
:math:`\cal E`. The atomistic environment of each atom can be described by :math:`M` fingerprints :math:`G_{im}` (:math:`i=1,2,\dots,N` and :math:`m=1,2,\dots,M`), which are used to determine the potential energy :math:`E_{i}` of :math:`i`-th atom. The atomic forces acting on each atom can be calculated as:


.. math::
   f_{j\alpha}=-\frac{\partial {\cal E}}{\partial r_{j\alpha}}= -
   \sum_{i=1}^{N} \frac{\partial E_i (G_{i1},
   G_{i2},\dots,G_{iM})}{\partial r_{j\alpha}} = -  \sum_{i=1}^{N}
   \sum_{m=1}^{M}	{\frac{\partial E_i}{\partial G_{im}}}
   {\frac{\partial G_{im}}{\partial r_{j\alpha}}}
   :label: force

where :math:`r_{j\alpha}` (:math:`\alpha=1,2,3`) are the Cartesian coordinates of :math:`j`-th atom (:math:`j=1,2,\dots,N`).



The first Piola-Kirchhoff (PK) stress tensor can be calculated as the work conjugate of deformation gradient tensor

.. math::
   \begin{equation}
	P_{\alpha\beta} = \frac{1}{V_0} \frac{\partial {\cal E}}{\partial F_{\alpha\beta}},
   \end{equation}
   :label: pk stress def

where :math:`V_0` is the volume of the reference configuration and :math:`F_{\alpha\beta}` is the deformation gradient tensor. Similar to the atomic force calculation, the potential energy can be written as the sum of atomic potential energies, so the stress can be written as

.. math::
   \begin{equation}
   P_{\alpha\beta} = \frac{1}{V_0} \sum_{i=1}^{N} \sum_{m=1}^{M} \frac{\partial E_i}{\partial G_{im}} \frac{\partial G_{im} } {\partial F_{\alpha \beta}}.
   \end{equation}
   :label: pk stress 1

The fingerprint :math:`G_{im}` is determined by the coordinates of the atoms inside the cutoff of atom :math:`i`. Therefore,

.. math::
   \begin{equation}
   \frac{\partial G_{im} } {\partial F_{\alpha \beta}} = \sum_{j \in \text{NB}_{i}} \sum_{\gamma=1}^{3} \frac{\partial G_{im}}{\partial r_{j\gamma}} \frac{\partial r_{j\gamma}}{\partial F_{\alpha \beta}},
   \end{equation}
   :label: dGdF

where atom :math:`j` is inside the neighbor list of atom :math:`i`. By definition, the deformation gradient maps the atom positionfrom the reference configuration to the current configuration.

.. math::
   \begin{equation}
	r_{j\gamma} = \sum_{\beta=1}^{3}F_{\gamma\beta}R_{j\beta} ,
   \end{equation}
   :label: coordinate mapping

where :math:`R_{j\beta}` is the coordinates of atom :math:`j` in the
reference configuration. Then, the stress in Eq. :eq:`pk stress 1` can be
written as

.. math::
   \begin{equation}
	P_{\alpha\beta} = \frac{1}{V_0} \sum_{i=1}^{N} \sum_{m=1}^{M} \sum_{j \in \text{NB}_i} \frac{\partial E_i}{\partial G_{im}} \frac{\partial G_{im} } {\partial r_{j \alpha}} R_{j\beta}.
   \end{equation}
   :label: pk stress 2

Cauchy stress can be further calculated by

.. math::
   \begin{equation}
	\sigma_{\alpha\gamma} = \det ({\bf F})^{-1} \sum_{\beta=1}^{3}P_{\alpha\beta} F_{\gamma \beta}.
   \end{equation}
   :label: cauchy stress from PK

Substitute Eq. :eq:`pk stress 2` into Eq. :eq:`cauchy stress from PK`, and
then apply :math:`V = V_0 \det ({\bf F})`, which is the volume in
current configuration, as well as :math:`r_{j\gamma} = \sum_{\beta
=1}^{3}F_{\gamma\beta}R_{j\beta}`, we get

.. math::
   \begin{equation}
	\sigma_{\alpha\beta} = \frac{1}{V}  \sum_{i=1}^{N} \sum_{m=1}^{M} \sum_{j \in \text{NB}_i} \frac{\partial E_i}{\partial G_{im}} \frac{\partial G_{im} } {\partial r_{j \alpha}} r_{j\beta}
   \end{equation}
   :label: cauchy stress

after replacing :math:`\gamma` with :math:`\beta`.





Implementation
--------------

In Tensorflow, it is more convenient and efficient to implement the above summation
in terms of matrix multiplication. Therefore, we define matrix

.. math::
   [dEdG]_i = 	\begin{pmatrix} \dfrac{\partial E_i}{\partial G_{i1}}
   & \dfrac{\partial E_i}{\partial G_{i2}}	& \dots &
   \dfrac{\partial E_i}{\partial G_{iM}} \end{pmatrix}


and

.. math::
    [dGdr]_{ij} = \begin{bmatrix}
		\dfrac{\partial G_{i1}}{\partial r_{j1}} & \dfrac{\partial G_{i1}}{\partial r_{j2}} & \dfrac{\partial G_{i1}}{\partial r_{j3}}  \\ \rule{0pt}{22pt}
		\dfrac{\partial G_{i2}}{\partial r_{j1}} & \dfrac{\partial G_{i2}}{\partial r_{j2}} & \dfrac{\partial G_{i2}}{\partial r_{j3}}  \\ \rule{0pt}{15pt}
		\vdots & \vdots & \vdots \\\rule{0pt}{15pt}
		\dfrac{\partial G_{iM}}{\partial r_{j1}} & \dfrac{\partial G_{iM}}{\partial r_{j2}} & \dfrac{\partial G_{iM}}{\partial r_{j3}}  \\
	\end{bmatrix}


Next, we use an example to demonstrate the matrix multiplicaiton
process. Assume there are 8 pairs of derivative (:math:`dGdr`) data in
one structure that contains 6 atoms. The center atoms IDs for thes 8 pairs are

.. math::
    \begin{bmatrix}
        1&2&2&4&5&0&1&2
    \end{bmatrix}

The neighbor atoms IDs are

.. math::
   \begin{equation}
   \begin{bmatrix}
   2&3&4&2&1&4&0&4
   \end{bmatrix}
   \end{equation}

The atomic forces can be computed as:

.. math::
   \begin{equation}
   \begin{bmatrix}
		\left[dEdG\right]_1 \\ \rule{0pt}{10pt}
		\left[dEdG\right]_2  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_2  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_4  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_5  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_0  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_1  \\ \rule{0pt}{10pt}
		\left[dEdG\right]_2  \\
	\end{bmatrix}
 	\begin{bmatrix}
 		\left[dGdr\right]_{12} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{23} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{24} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{42} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{51} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{04} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{10} \\ \rule{0pt}{10pt}
 		\left[dGdr\right]_{24} \\
 	\end{bmatrix}
 	= 	\begin{bmatrix}
 			(f_2^x)_1 & (f_2^y)_1 & (f_2^z)_1 \\ \rule{0pt}{10pt}
 			(f_3^x)_2 & (f_3^y)_2 & (f_3^z)_2 \\ \rule{0pt}{10pt}
 			(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2 \\ \rule{0pt}{10pt}
 			(f_2^x)_4 & (f_2^y)_4 & (f_2^z)_4 \\ \rule{0pt}{10pt}
 			(f_1^x)_5 & (f_1^y)_5 & (f_1^z)_5 \\ \rule{0pt}{10pt}
 			(f_4^x)_0 & (f_4^y)_0 & (f_4^z)_0 \\ \rule{0pt}{10pt}
 			(f_0^x)_1 & (f_0^y)_1 & (f_0^z)_1 \\ \rule{0pt}{10pt}
 			(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2
 	\end{bmatrix}
	\end{equation}

where :math:`(f_j^x)_i` is the force on :math:`j`-th atom contributed
by atom :math:`i`. Then we need to compute the total force acting on
atom :math:`j`, :math:`F_j`, by summing up all the contributions for
other atoms. This is done by using one-hot matrix generated by the
neighbor atom IDs:

.. math::
   \begin{equation}
	-\begin{bmatrix}
		0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ \rule{0pt}{10pt}
		0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ \rule{0pt}{10pt}
		1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ \rule{0pt}{10pt}
		0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ \rule{0pt}{10pt}
		0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 \\ \rule{0pt}{10pt}
		0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
	\end{bmatrix}
    \begin{bmatrix}
 			(f_2^x)_1 & (f_2^y)_1 & (f_2^z)_1 \\ \rule{0pt}{10pt}
 			(f_3^x)_2 & (f_3^y)_2 & (f_3^z)_2 \\ \rule{0pt}{10pt}
 			(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2 \\ \rule{0pt}{10pt}
 			(f_2^x)_4 & (f_2^y)_4 & (f_2^z)_4 \\ \rule{0pt}{10pt}
 			(f_1^x)_5 & (f_1^y)_5 & (f_1^z)_5 \\ \rule{0pt}{10pt}
 			(f_4^x)_0 & (f_4^y)_0 & (f_4^z)_0 \\ \rule{0pt}{10pt}
 			(f_0^x)_1 & (f_0^y)_1 & (f_0^z)_1 \\ \rule{0pt}{10pt}
 			(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2
 	\end{bmatrix}
	=	\begin{bmatrix}
		F_0^x & F_0^y & F_0^z \\ \rule{0pt}{10pt}
		F_1^x & F_1^y & F_1^z \\ \rule{0pt}{10pt}
		F_2^x & F_2^y & F_2^z \\ \rule{0pt}{10pt}
		F_3^x & F_3^y & F_3^z \\ \rule{0pt}{10pt}
		F_4^x & F_4^y & F_4^z \\ \rule{0pt}{10pt}
		F_5^x & F_5^y & F_5^z
	\end{bmatrix}
	\end{equation}


Define the coordinates of neighboring atom

.. math::
  \begin{equation}
      [r]_j = \begin{bmatrix}
      r_{jx}\\
      r_{jy}\\
      r_{jz}
      \end{bmatrix}
  \end{equation}

The stress can be calculated as

.. math::
  \begin{equation}
  \begin{bmatrix}
    [r]_2 \\ [r]_3 \\ [r]_4 \\ [r]_2 \\ [r]_1 \\ [r]_4 \\ [r]_0 \\ [r]_4
  \end{bmatrix}
  \begin{bmatrix}
    [(f_2^x)_1 & (f_2^y)_1 & (f_2^z)_1] \\ \rule{0pt}{10pt}
    [(f_3^x)_2 & (f_3^y)_2 & (f_3^z)_2] \\ \rule{0pt}{10pt}
    [(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2]\\ \rule{0pt}{10pt}
    [(f_2^x)_4 & (f_2^y)_4 & (f_2^z)_4] \\ \rule{0pt}{10pt}
    [(f_1^x)_5 & (f_1^y)_5 & (f_1^z)_5] \\ \rule{0pt}{10pt}
    [(f_4^x)_0 & (f_4^y)_0 & (f_4^z)_0] \\ \rule{0pt}{10pt}
    [(f_0^x)_1 & (f_0^y)_1 & (f_0^z)_1] \\ \rule{0pt}{10pt}
    [(f_4^x)_2 & (f_4^y)_2 & (f_4^z)_2]
   \end{bmatrix}
    = \begin{bmatrix}
         [\sigma_{\alpha\beta}]_1 \\
         [\sigma_{\alpha\beta}]_2 \\
         [\sigma_{\alpha\beta}]_3 \\
         [\sigma_{\alpha\beta}]_4 \\
         [\sigma_{\alpha\beta}]_5 \\
         [\sigma_{\alpha\beta}]_6 \\
         [\sigma_{\alpha\beta}]_7 \\
         [\sigma_{\alpha\beta}]_8 \\
      \end{bmatrix}
  \end{equation}


The final stress is

 .. math::
   \begin{equation}
       \sigma_{\alpha\beta} = \frac{1}{V} \sum_{k=1}^{8} [\sigma_{\alpha\beta}]_k
   \end{equation}
