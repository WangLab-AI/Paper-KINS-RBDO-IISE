# -*- coding: utf-8 -*-

import numpy as np

def G2(X):

    Nd = 21
    N = X.shape[0]
    A = X[:, 0:Nd] * 1e-4  # (N, 21)
    E = X[:, Nd:2*Nd] * 1e9  # (N, 21)
    F = X[:, 2*Nd:] * 1e3  # (N, 3)


    ele_nod = np.array([
        [1, 2], [1, 3], [2, 3], [2, 4], [1, 4],
        [3, 4], [3, 6], [4, 5], [4, 6], [3, 5],
        [5, 6], [4, 7], [4, 8], [6, 7], [6, 8],
        [7, 8], [7, 9], [7,10], [8, 9], [8,10], [9,10]
    ]) - 1
    num_ele = ele_nod.shape[0]
    num_nod = 10
    nod_coor = np.array([
        [0, 0], [0, 3], [3, 0], [3, 3], [6, 0],
        [6, 3], [3, 6], [6, 6], [3, 9], [6, 9]
    ])
	
    ele_dof = np.zeros((num_ele, 4), dtype=int)
	
    for i in range(num_ele):
        ele_dof[i, :] = [
            ele_nod[i, 0]*2, ele_nod[i, 0]*2+1,
            ele_nod[i, 1]*2, ele_nod[i, 1]*2+1
        ]

    Nbc = np.array([0, 1, 4, 5, 8, 9])
    Nall = np.setdiff1d(np.arange(2*num_nod), Nbc)

    #y_all = np.zeros(N)
    stress_all = np.zeros(N)

    for idx in range(N):

        a = A[idx]
        e = E[idx]
        f = F[idx]

        displacement = np.zeros(2*num_nod)
        force = np.zeros(2*num_nod)
        stiffness = np.zeros((2*num_nod, 2*num_nod))

        # 载荷施加
        force[2] = f[0]
        #force[3] = 0.0
        #force[6:16] = 0.0
        force[16] = f[1]
        #force[17] = 0.0
        force[18] = f[2]

        displacement[Nbc] = 0.0


        for e_id in range(num_ele):
            ni, nj = ele_nod[e_id]
            xi, yi = nod_coor[ni]
            xj, yj = nod_coor[nj]
            L = np.hypot(xj - xi, yj - yi)
            C = (xj - xi) / L
            S = (yj - yi) / L
			
            AE_L = a[e_id] * e[e_id] / L
            k_local = AE_L * np.array([
                [ C*C,  C*S, -C*C, -C*S],
                [ C*S,  S*S, -C*S, -S*S],
                [-C*C, -C*S,  C*C,  C*S],
                [-C*S, -S*S,  C*S,  S*S]
            ])
			
            dof = ele_dof[e_id]
            for i in range(4):
                for j in range(4):
                    stiffness[dof[i], dof[j]] += k_local[i, j]			

        dis_new = np.linalg.solve(stiffness[np.ix_(Nall, Nall)], force[Nall])
        displacement[Nall] = dis_new			


        max_stress = 0.0
        for e_id in range(num_ele):
            ni, nj = ele_nod[e_id]
            xi, yi = nod_coor[ni]
            xj, yj = nod_coor[nj]
            L = np.hypot(xj - xi, yj - yi)
            C = (xj - xi) / L
            S = (yj - yi) / L

            dof = ele_dof[e_id]
            u = displacement[dof]
            du = (u[2] - u[0]) * C + (u[3] - u[1]) * S
            stress = e[e_id] * du / L  # σ = E * ΔL / L

            if abs(stress) > max_stress:
                max_stress = abs(stress)


        stress_all[idx] = 60e6 - max_stress #200e6 - 

    return stress_all/1e7
