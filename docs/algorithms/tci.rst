.. _alg_ttcross:

********************************
Tensor cross-interpolation (TCI)
********************************

Tensor cross-interpolation (TCI) is an efficient method for approximating the MPS/TT representation of a black-box function by sampling its elements along structured patterns known as crosses. It does not act on the explicit discretization-tensor representation, providing an exponential time and memory advantage over standard Schmidt decompositions and evading the *curse of dimensionality*.

There are several known algorithmic variants for TCI. Each exhibits complementary computational tradeoffs. The SeeMPS library implements three TCI variants:

1. :func:`~seemps.analysis.cross.cross_dmrg`: Based on two-site optimizations combining the singular-value (SVD) and skeleton decompositions. It is typically the fastest variant in practice, as it can increase the bond dimension aggressively between sweeps while still achieving good convergence. However, the rapid rank growth may also cause the algorithm to become trapped in suboptimal configurations or fail to converge in difficult problems. This makes it particularly efficient for encoding smooth, well-behaved functions over more complex scenarios. Moreover, this approach may become inefficient for structures of very large bond dimension due to its computational complexity. Has an associated prameter dataclass given by :class:`~seemps.analysis.cross.CrossStrategyDMRG`.

2. :func:`~seemps.analysis.cross.cross_greedy`: Based on two-site optimizations performing greedy searches for maximum-volume pivots. Due to its greedy pivot selection, it can require significantly less time per sweep than the other approaches, particularly for structures of large physical dimension. However, it increases the bond dimension by only one unit per sweep and does not examine all elements in the sampled fibers, becoming less reliable than other variantsm and highly unstable when the selected pivots are ill-conditioned. Has an associated parameter dataclass given by :class:`~seemps.analysis.cross.CrossStrategyGreedy`.

3. :func:`~seemps.analysis.cross.cross_maxvol`: Based on rank-adaptive one-site optimizations using the rectangular skeleton decomposition. This method can be seen as a compromise between cross_dmrg and cross_greedy. By allowing a fixed, user-controlled increase of the bond dimension at each sweep, it provides more robust and predictable convergence behavior than cross_dmrg, at the expense of increased computational cost. As a result, it is generally slower than cross_dmrg but significantly more stable in challenging cases. Has an associated parameter dataclass given by :class:`~seemps.analysis.cross.CrossStrategyMaxvol`.

By default, we recommend using :func:`seemps.analysis.cross.cross_dmrg` due to its superior efficiency. If convergence issues arise, :func:`seemps.analysis.cross.cross_maxvol` should be preferred as a more robust alternative.

Moreover, this method performs the decomposition of a given input black-box. This black-box can take several different forms and serve for different application domains. This library implements the class :class:`~seemps.analysis.cross.black_box.BlackBox` and the following subclasses:

1. :class:`~seemps.analysis.cross.black_box.BlackBoxLoadMPS`: Required to encode functions in MPS with arbitrary quantization and structure. When present, the quantization structure is specified through a linear transformation defined by an input matrix, together with an optional permutation vector to represent interleaved core orderings. If no quantization is provided, a default TT structure is assumed, in which each function dimension is assigned an unique MPS core. This encompasses arbitrary core layours, including *serial* and *interleaved* arrangements. 

2. :class:`~seemps.analysis.cross.black_box.BlackBoxComposeMPS`: Required to compose scalar functions acting on collections of MPS.

3. :class:`~seemps.analysis.cross.black_box.BlackBoxLoadMPO`: Required to load bivariate functions represented as MPOs, by internally constructing the equivalent MPS representation. This intermediate MPS has square physical dimension (e.g., 4 for an MPO of dimension 2) and can be unfolded at the end to recover the desired MPO structure.

An example on how to use TCI for all these scenarios is shown in `TCI.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/TCI.ipynb>`_.