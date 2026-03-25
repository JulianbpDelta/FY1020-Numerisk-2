import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


k = 100        # intern fjærstivhet
K = 10000      # clamps
B = 1.0        # tykkelse


def make_simple_mesh(a0):
    b0 = np.sqrt(3) * a0
    xy = np.array([ 
        [0, 0],
        [b0, 0],
        [b0/2, a0/2],
        [0, a0],
        [b0, a0]
    ])

    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4]
    ])

    return xy, edges


def spring_energy(xy, k, edges, ell0_):
    energy = 0.0
    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        rij = xy[j] - xy[i]
        ell = np.linalg.norm(rij)
        energy += 0.5 * k * (ell - ell0)**2
    return energy


def spring_forces(xy, edges, k, ell0_):
    forces = np.zeros_like(xy)

    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        rij = xy[j] - xy[i]
        ell = np.linalg.norm(rij)

        if ell > 0:
            f_mag = k * (ell - ell0)
            f_vec = f_mag * (rij / ell)

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces


def total_energy(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy = xy_flat.reshape((-1, 2))

    energy = spring_energy(xy, k, edges, ell0_)

    # clamps x-retning
    energy += 0.5 * K * (xy[ids_left, 0]**2).sum()
    energy += 0.5 * K * ((xy[ids_right, 0] - Lx_plate)**2).sum()

    # stabilisering i y-retning
    energy += 0.5 * K * ((xy[ids_left, 1] - xy0[ids_left, 1]).mean()**2)
    energy += 0.5 * K * ((xy[ids_right, 1] - xy0[ids_right, 1]).mean()**2)

    return energy



def total_energy_jacobian(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy = xy_flat.reshape((-1, 2))

    grad = -spring_forces(xy, edges, k, ell0_)

    grad[ids_left, 0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - Lx_plate)

    grad[ids_left, 1] += K * (xy[ids_left, 1] - xy0[ids_left, 1]).mean()
    grad[ids_right, 1] += K * (xy[ids_right, 1] - xy0[ids_right, 1]).mean()

    return grad.flatten()



a0 = 1.0
xy, edges = make_simple_mesh(a0)


ell0_ = np.linalg.norm(xy[edges[:,0]] - xy[edges[:,1]], axis=1)


xy0 = xy.copy()


tol = 1e-12
Lx0 = np.max(xy[:,0])
Ly0 = np.max(xy[:,1])

ids_left = np.where(xy[:,0] < tol)[0]
ids_right = np.where(xy[:,0] > Lx0 - tol)[0]
ids_bottom = np.where(xy[:,1] < tol)[0]
ids_top = np.where(xy[:,1] > Ly0 - tol)[0]

eps_list = []
E_list = []
nu_list = []

# deformasjoner
f_values = np.linspace(1e-3, 0.5, 10)

for f in f_values:
    Lx_plate = Lx0 * (1 + f)

    res = minimize(
        total_energy,
        xy.flatten(),
        args=(edges, k, K, ell0_, Lx_plate),
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-12,
        options={'maxiter': 1000}
    )

    xy_eq = res.x.reshape((-1,2))

    Lx = xy_eq[ids_right, 0].mean() - xy_eq[ids_left, 0].mean()
    Ly = xy_eq[ids_top, 1].mean() - xy_eq[ids_bottom, 1].mean()

    # kraft
    Fn = -K * (xy_eq[ids_right, 0] - Lx_plate).sum()

 
    eps_x = (Lx - Lx0) / Lx0
    eps_y = (Ly - Ly0) / Ly0

    sigma = Fn / (B * Ly)


    E = sigma / eps_x if eps_x != 0 else 0

    nu = -eps_y / eps_x if eps_x != 0 else 0
    eps_list.append(eps_x)
    E_list.append(E)
    nu_list.append(nu)

    print(f"strain={eps_x:.4f}, E={E:.2f}, nu={nu:.3f}")



plt.figure()
plt.plot(eps_list, E_list, 'o-')
plt.xlabel("strain epsilon")
plt.ylabel("Young's modulus E")
plt.title("E vs strain")
plt.grid()
plt.show()

plt.figure()
plt.plot(eps_list, nu_list, 'o-')
plt.xlabel("strain epsilon")
plt.ylabel("Poisson ratio nu")
plt.title("poisson ratio vs strain")
plt.grid()
plt.show()
