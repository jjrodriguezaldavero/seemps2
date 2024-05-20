    def update_tensors(
        self,
        k: int,
        forward: bool,
        r: np.ndarray,
        c: np.ndarray,
    ) -> None:
        i_l, i_s1, chi = self.mps[k].shape
        G = self.mps[k].reshape(i_l * i_s1, chi)
        chi, i_s2, i_g = self.mps[k + 1].shape
        G_plus_1 = self.mps[k + 1].reshape(chi, i_s2 * i_g)

        if forward:
            j_l = self.J_l[k + 1][:-1]
            new_j_l = self.J_l[k + 1][-1]

            S = c[new_j_l] - np.dot(G[new_j_l], c[j_l])
            G1 = (np.outer((G @ c[j_l]), G[new_j_l]) - np.outer(c, G[new_j_l])) / S
            G2 = ((c - G @ c[j_l]) / S).reshape(-1, 1)
            new_G = np.hstack((G + G1, G2))

            R = self.fibers[k + 1].reshape(chi, i_s2 * i_g)
            new_R = np.vstack((R, r))
            new_G_plus_1 = np.vstack((G_plus_1, r))

            self.mps[k] = new_G.reshape(i_l, i_s1, chi + 1)
            self.fibers[k + 1] = new_R.reshape(chi + 1, i_s2, i_g)
            self.mps[k + 1] = new_G_plus_1.reshape(chi + 1, i_s2, i_g)
        else:
            j_g = self.J_g[k][:-1]
            new_j_g = self.J_g[k][-1]

    # def update_tensors(
    #     self,
    #     k: int,
    #     forward: bool,
    #     r: np.ndarray,
    #     c: np.ndarray,
    #     A: np.ndarray,
    # ) -> None:
    #     i_l, i_s1, chi = self.mps[k].shape
    #     G = self.mps[k].reshape(i_l * i_s1, chi)
    #     chi, i_s2, i_g = self.mps[k + 1].shape
    #     R_fiber = self.fibers[k + 1].reshape(chi, i_s2 * i_g)
    #     R_mps = self.mps[k + 1].reshape(chi, i_s2 * i_g)
    #     I = self.J_l[k + 1][:-1]
    #     i = self.J_l[k + 1][-1]

    #     # S = c[i] - _contract_last_and_first(G[i], c[I])  # Schur complement
    #     # G1 = (np.outer(_contract_last_and_first(G, c[I]), G[i]) - np.outer(c, G[i])) / S
    #     # G2 = (c - _contract_last_and_first(G, c[I])) / S

    #     S = c[i] - np.dot(G[i], c[I])
    #     G1 = (np.outer((G @ c[I]), G[i]) - np.outer(c, G[i])) / S
    #     G2 = (c - G @ c[I]) / S

    #     new_G = np.hstack((G + G1, G2.reshape(-1, 1)))
    #     new_R_fiber = np.vstack((R_fiber, r))
    #     new_R_mps = np.vstack((R_mps, r))

    #     # DEBUG: Test if the decomposition is accurate
    #     diff = A - new_G @ new_R_fiber

    #     self.mps[k] = new_G.reshape(i_l, i_s1, chi + 1)
    #     self.mps[k + 1] = new_R_mps.reshape(chi + 1, i_s2, i_g)
    #     self.fibers[k + 1] = new_R_fiber.reshape(chi + 1, i_s2, i_g)
