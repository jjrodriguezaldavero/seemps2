.. _analysis_expansion:

********************************
Orthogonal polynomial expansions
********************************

Matrix-product states (MPS) and operators (MPO) can be expanded in bases of orthogonal polynomials, enabling the approximation, evaluation, and composition of functions and operators.

In principle, such expansions can be extended to arbitrary multivariate functions. However, the associated coefficient tensor scales exponentially with the number of variables, rendering such approximations impractical. Instead, the SeeMPS library adopts a different strategy: multivariate approximations of functions with compositional algebraic structure are constructed by composing univariate orthogonal polynomial expansions with multivariate inputs. This approach avoids the exponential growth of coefficients while still enabling flexible and accurate representations of multivariate functions.

Computation of the expansion coefficients
=========================================
An expansion of a univariate function in an orthogonal polynomial basis reads

.. math::
    p_d(x) = \sum_{k=0}^d c_k P_k(x),

where :math:`\{P_k\}_k` denotes a family of orthogonal polynomials defined with respect to a given weight function on a prescribed interval. Some examples include Chebyshev, Lagrange and Hermite polynomials. These polynomials satisfy a three-term recurrence relation of the form

.. math::
    P_{k+1}(x) = (a_k x + b_k) P_k(x) - c_k P_{k-1}(x), \quad k \ge 1,

with coefficients determined by the chosen polynomial family.

The expansion coefficients :math:`c_k` encode the information of the target function and can be computed either by projection onto the polynomial basis or by interpolation (colocation) on a suitable set of quadrature nodes associated with the orthogonal family. The library currently provides routines to compute projection coefficients for Chebyshev and Legendre families through the classes :class:`ChebyshevExpansion` and :class:`LegendreExpansion`. Other families can be readily implemented by subclassing the abstract class :class:`OrthogonalExpansion`. These subclasses must implement the method :func:`OrthogonalExpansion.project()` to compute projection coefficients. Optionally, colocation routines may be implemented---e.g., for Chebyshev expansions :func:`ChebyshevExpansion.interpolate()` is available. 

These coefficients depend on the univariate input function, its domain of definition, and (optionally) the chosen truncation order. This order can be estimated automatically to near machine precision using an adaptive estimation routine :func:`OrthogonalExpansion.estimate_order()` to the given tolerance. Moreover, polynomial basis often provide additional functionality. For example, spectral derivatives and primitives are readily available in the Chebyshev basis through the :func:`ChebyshevExpansion.deriv()` and :func:`ChebyshevExpansion.integ()` methods, built from recurrence relations and enabling fast spectral derivation of functions.

Expansion in an orthogonal polynomial basis
===========================================
Once the expansion coefficients have been computed, the polynomial series can be applied to a given initial MPS or MPO, respectively using the functions :func:`mps_polynomial_expansion()` and :func:`mpo_polynomial_expansion()`. This enables function evaluation or functional composition directly within the MPS/MPO formalism.

This initial state must be supported within the canonical interval associated with the chosen polynomial family---e.g., within :math:`[-1, 1]` for Chebyshev expansions. For MPS, this corresponds to the range of values represented by the state (i.e. its extremal values) while for MPO it corresponds to the spectral range (i.e. its extremal eigenvalues). If the support lies outside of the canonical interval, it must be rescaled using an affine transformation. By default, this rescaling is performed automatically, assuming the initial condition is defined on the same domain used to compute the expansion coefficients. To disable this automatic rescaling, one may disable the ``rescale`` flag within the expansion functions.

A direct evaluation of the truncated series can be performed directly, by explicitly constructing the polynomial terms using their recurrence relation and forming the weighted sum (``clenshaw`` flag set to ``False``); or, alternatively, using Clenshaw-type recurrence schemes (``clenshaw`` flag set to ``True``). This alternative approach avoids explicit construction of intermediate polynomial states and typically offers improved numerical stability and performance. However, they can be sensitive to overestimation of the truncation order, degrading performance.

Constructing the initial condition
==================================
Polynomial expansions require an initial MPS or MPO to which the function is applied, passed to the ``initial`` argument. This is an user-defined arbitrary object defined within the canonical interval of the polynomial family. For function-loading applications where the object represents a discretized domain, they can be constructed automatically by passing an :class:`~seemps.analysis.mesh.Interval` object to the ``initial`` argument, which is automatically converted to MPS form. Current implementations accept equispaced discretizations through :class:`~seemps.analysis.mesh.RegularInterval` objects, or discretizations on Chebyshev nodes through :class:`~seemps.analysis.mesh.ChebyshevInterval` objects. 

Multivariate functions
======================
These approaches enable the construction of multivariate functions and operators through univariate function composition. The initial states may be multivariate through outer products or sums on univariate states, using the functions :func:`~seemps.analysis.factories.mps_tensor_product` and :func:`~seemps.analysis.factories.mps_tensor_sum` (as long as the output states are rescaled appropriately to the canonical domain).

An example on how to use these functions is shown in `Chebyshev.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Chebyshev.ipynb>`.

.. autosummary::
    :toctree: generated/

    ~seemps.analysis.expansion.mps_polynomial_expansion
    ~seemps.analysis.expansion.mpo_polynomial_expansion
    ~seemps.analysis.expansion.PowerExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion
    ~seemps.analysis.expansion.LegendreExpansion