def translate_indices(self, k: int) -> tuple[np.ndarray, np.ndarray]:
    I_small = self.I_l[k + 1]
    J_small = self.I_g[k]
    I_large = self.combine_indices(self.I_l[k], self.I_s[k])
    J_large = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])

    find_indices = lambda
    def find_indices(small, large):
        # This will store the indices of each row of `small` in `large`
        indices = np.array(
            [
                np.nonzero((large == single_row).all(axis=1))[0][0]
                for single_row in small
            ]
        )
        return indices

    # Find indices of I_small in I_large
    I = find_indices(I_small, I_large)
    J = find_indices(J_small, J_large)
    return I, J

# Translate the binary indices to integer indices
# Maybe I can progressively keep track of it in cross
I, J = cross.translate_indices(k)
C = B[:, J]
R = B[I, :]

# Update the tensors
if forward:
    Q, T = np.linalg.qr(C)
    P = Q[I]
    P_inv = np.linalg.inv(P)
    cross.fibers[k] = Q.reshape(r_l, s1, -1)
    cross.pivots[k] = P_inv
    cross.fibers[k + 1] = R.reshape(-1, s2, r_g)
    if k == cross.sites - 2:
        cross.fibers[k + 1] = _contract_last_and_first(
            Q[I] @ T, cross.fibers[k + 1]
        )
else:
    Q, T = np.linalg.qr(R.T)
    P = Q[J].T
    P_inv = np.linalg.inv(P)
    cross.fibers[k] = C.reshape(r_l, s1, -1)
    cross.pivots[k] = P_inv
    cross.fibers[k + 1] = Q.reshape(-1, s2, r_g)
    if k == 0:
        cross.pivots[k] = (Q[I] @ T).T @ cross.pivots[k]

i_l = cross.combine_indices(cross.I_l[k], cross.I_s[k])[i]
i_g = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[j]

# Update the indices and the tensors
# TODO: The updates are not stable. Maybe I have to use the QR-trick

    cross.I_g[k] = np.vstack((cross.I_g[k], i_g))
    cross.I_l[k + 1] = np.vstack((cross.I_l[k + 1], i_l))
    cross.update_tensors(k, i_l, i_g)

# cross.update_fiber(k, i_g=i_g)
# cross.update_fiber(k + 1, i_l=i_l)
# cross.update_pivot(k, i_l, i_g)

def update_pivot(
    self, k: int, i_l: Optional[np.ndarray] = None, i_g: Optional[np.ndarray] = None
):
    self.pivots[k] = self.sample_pivot(k)

def update_fiber(
    self, k: int, i_l: Optional[np.ndarray] = None, i_g: Optional[np.ndarray] = None
) -> None:
    self.fibers[k] = self.sample_fiber(k)

def update_tensors(self, k: int, i_l: np.ndarray, i_g: np.ndarray) -> None:
    C = _contract_last_and_first(self.fibers[k], self.pivots[k])
    Q, T = np.linalg.qr(C)
    P = np.linalg.inv(Q[self.I_l[k + 1]])


### Partial search code

if cross_strategy.greedy_method == "full_search":
    update_method = _update_full_search
elif cross_strategy.greedy_method == "partial_search":
    update_method = _update_partial_search

def sample_submatrix(
    self,
    k: int,
    row_idx: Optional[Union[int, np.ndarray]] = None,
    col_idx: Optional[Union[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    TODO: This implementation evaluates the whole superblock, reshapes it to matrix form,
    and slices it using row_idx and col_idx. It would be more efficient to just sample the
    required elements instead of the whole superblock.
    """
    i_l, i_g = self.I_l[k], self.I_g[k + 1]
    i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
    r_l, s1, s2, r_g = len(i_l), len(i_s1), len(i_s2), len(i_g)
    row_idx = np.arange(r_l * s1) if row_idx is None else np.asarray(row_idx)
    col_idx = np.arange(s2 * r_g) if col_idx is None else np.asarray(col_idx)
    mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
    return self.black_box[mps_indices].reshape(r_l * s1, s2 * r_g)[row_idx, col_idx]


def _update_partial_search(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    # Compute the skeleton decomposition at site k
    skeleton = cross.skeleton(k)
    r_l, s1, s2, r_g = skeleton.shape
    A = skeleton.reshape(r_l * s1, s2 * r_g)

    # Choose an initial point that has maximum error from a random set
    rng = cross_strategy.rng
    I_random = rng.integers(low=0, high=r_l * s1, size=cross_strategy.partial_points)
    J_random = rng.integers(low=0, high=s2 * r_g, size=cross_strategy.partial_points)
    random_set = cross.sample_submatrix(k, I_random, J_random)
    idx = np.argmax(np.abs(A[I_random, J_random] - random_set))
    i, j = I_random[idx], J_random[idx]

    # Find the pivots that have a maximum error doing an alternate row-column search
    # Note: the residuals `col` and `row` may be used as `u` and `v` to update the pivot matrix
    cost_function = lambda A, B: np.abs(A - B)
    for _ in range(cross_strategy.partial_maxiter):
        col = cross.sample_submatrix(k, col_idx=j)  # type: ignore
        i_k = np.argmax(cost_function(A[:, j], col))
        row = cross.sample_submatrix(k, row_idx=i_k)  # type: ignore
        diff = cost_function(A[i_k, :], row)
        j_k = np.argmax(diff)
        if (i_k, j_k) == (i, j):
            break
        (i, j) = (i_k, j_k)
    i_l = cross.combine_indices(cross.I_l[k], cross.I_s[k])[i]
    i_g = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[j]

    # Update the pivots and the cross-interpolation
    if (
        i_l.tolist() not in cross.I_l[k + 1].tolist()
        and i_g.tolist() not in cross.I_g[k].tolist()
    ):
        cross.I_g[k] = np.vstack((cross.I_g[k], i_g))
        cross.I_l[k + 1] = np.vstack((cross.I_l[k + 1], i_l))
        cross.update_fiber(k, i_g=i_g)
        cross.update_fiber(k + 1, i_l=i_l)
        cross.update_pivot(k, i_l, i_g)


# if k == 0:
#     cross.fibers[k] = _contract_last_and_first(
#         cross.fibers[k], (Q[J_cols] @ T).T
#     )
#         if k == cross.sites - 2:
#     cross.fibers[k + 1] = _contract_last_and_first(
#         Q[J_rows] @ T, cross.fibers[k + 1]
#     )