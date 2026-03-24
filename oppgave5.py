import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


k = 20      # fjærstivhet (interne fjærer)
K = 1000    # sterke clamps


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


def plot_mesh(xy, edges):
    fig, ax = plt.subplots()

    for edge in edges:
        ax.plot(xy[edge, 0], xy[edge, 1], 'k')

    ell0_ = np.linalg.norm(xy[edges[:,0]] - xy[edges[:,1]], axis=1)

    ax.scatter(xy[:,0], xy[:,1], color='red')
    ax.set_aspect('equal')
    ax.set_title("Initial mesh")
    plt.show()

    return ell0_


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


    energy += 0.5 * K * (xy[ids_left, 0]**2).sum()
    energy += 0.5 * K * ((xy[ids_right, 0] - Lx_plate)**2).sum()

    # stabilisering i y-retning
    energy += 0.5 * K * ((xy[ids_left, 1] - xy0[ids_left, 1]).mean()**2)
    energy += 0.5 * K * ((xy[ids_right, 1] - xy0[ids_right, 1]).mean()**2)

    return energy


#jacobi
def total_energy_jacobian(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy = xy_flat.reshape((-1, 2))

    grad = -spring_forces(xy, edges, k, ell0_)

    grad[ids_left, 0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - Lx_plate)

    grad[ids_left, 1] += K * (xy[ids_left, 1] - xy0[ids_left, 1]).mean()
    grad[ids_right, 1] += K * (xy[ids_right, 1] - xy0[ids_right, 1]).mean()

    return grad.flatten()



def plot_deformed_mesh(xy, edges, ell0_):
    fig, ax = plt.subplots()

    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        rij = xy[j] - xy[i]
        ell = np.linalg.norm(rij)

        strain = (ell - ell0) / ell0

        if strain > 0:
            color = (1, 0, 0)   
        elif strain < 0:
            color = (0, 0, 1)   
        else:
            color = (0, 0, 0)

        ax.plot(xy[edge, 0], xy[edge, 1], color=color, linewidth=2)

    ax.scatter(xy[:,0], xy[:,1], color='black')
    ax.set_aspect('equal')
    ax.set_title("Red = tension, Blue = compression")
    plt.show()


a0 = 1.0
xy, edges = make_simple_mesh(a0)

# plott start og få hvilelengder
ell0_ = plot_mesh(xy, edges)

xy0 = xy.copy()


tol = 1e-12
Lx0 = np.max(xy[:,0])

ids_left = np.where(xy[:,0] < tol)[0]
ids_right = np.where(xy[:,0] > Lx0 - tol)[0]

# test ulike strekk
stretch_factors = [1.0, 1.1, 1.2]

for factor in stretch_factors:
    Lx_plate = factor * Lx0

    res = minimize(
        total_energy,
        xy.flatten(),
        args=(edges, k, K, ell0_, Lx_plate),
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-12,
        options={'maxiter': 1000}
    )

    xy_eq = res.x.reshape((-1, 2))

    print(f"Lplate = {Lx_plate:.3f}, success = {res.success}")

plot_deformed_mesh(xy_eq, edges, ell0_)
