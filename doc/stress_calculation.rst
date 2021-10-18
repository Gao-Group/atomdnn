==================
Stress calculation
==================
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

