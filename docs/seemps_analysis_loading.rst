.. _analysis_loading:

****************
Function Loading
****************

The SeeMPS library provides several methods to load univariate and multivariate functions in MPS and MPO structures. In the following, the most important are listed.

Tensorized operations
---------------------
These methods are useful to construct MPS corresponding to domain discretizations, and compose them using tensor products and sums to construct multivariate domains.

.. autosummary::

    ~seemps.analysis.mesh.RegularInterval
    ~seemps.analysis.mesh.ChebyshevInterval
    ~seemps.analysis.factories.mps_interval
    ~seemps.state.mps_tensor_product
    ~seemps.state.mps_tensor_sum

Tensor cross-interpolation (TCI)
--------------------------------
These methods are useful to compose MPS or MPO representations of black-box functions using tensor cross-interpolation (TCI). See :doc:`algorithms/tci`

<<<<<<< HEAD
.. autosummary::
    :toctree: generated/
    ~seemps.analysis.cross.cross_interpolation
    ~seemps.analysis.cross.cross_maxvol
    ~seemps.analysis.cross.cross_dmrg
    ~seemps.analysis.cross.cross_greedy
    ~seemps.analysis.cross.black_box.BlackBoxLoadMPS
    ~seemps.analysis.cross.black_box.BlackBoxLoadMPO
    ~seemps.analysis.cross.black_box.BlackBoxComposeMPS

    
Orthogonal polynomial expansions
--------------------------------
These methods are useful to compose univariate function on generic initial MPS or MPO and compute MPS approximations of functions.
See :doc:`algorithms/expansion`.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.expansion.mps_polynomial_expansion
    ~seemps.analysis.expansion.mpo_polynomial_expansion
    ~seemps.analysis.expansion.PowerExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion
    ~seemps.analysis.expansion.LegendreExpansion
=======

Polynomial expansions
---------------------
These methods are useful to compose univariate function on generic initial MPS or MPO and compute MPS approximations of functions. See :doc:`algorithms/polynomials`
>>>>>>> 3a5a0ff7a0800aee8e1cf7537c1aefa73b5e1ad4


Multiscale interpolative constructions
--------------------------------------
These methods are useful to construct polynomial interpolants of low-dimensional functions in MPS using the Chebyshev-Lagrange interpolation framework.
See :doc:`algorithms/lagrange`.
<<<<<<< HEAD

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.lagrange.mps_lagrange_chebyshev_basic
    ~seemps.analysis.lagrange.mps_lagrange_chebyshev_rr
    ~seemps.analysis.lagrange.mps_lagrange_chebyshev_lrr

Generic polynomial constructions
--------------------------------
These methods are useful to construct generic polynomials in the monomial basis from a collection of coefficients.

.. autosummary::
    :toctree: generated/
    
    ~seemps.analysis.polynomials.mps_from_polynomial
=======
>>>>>>> 3a5a0ff7a0800aee8e1cf7537c1aefa73b5e1ad4
