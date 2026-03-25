import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import Delaunay


k = 100
K = 10000
B = 1.0

L0 = 0.2
H0 = 0.1
N = 100


def make_random_mesh(N, L0, H0):
    # tilfeldige punkter
    xy = np.zeros((N,2))
    xy[:,0] = np.random.rand(N) * L0
    xy[:,1] = np.random.rand(N) * H0

    # Delaunay triangulering
    tri = Delaunay(xy)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1,3):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)

    edges = np.array(list(edges))

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
    xy = xy_flat.reshape((-1,2))

    energy = spring_energy(xy, k, edges, ell0_)

    # clamps x
    energy += 0.5 * K * (xy[ids_left,0]**2).sum()
    energy += 0.5 * K * ((xy[ids_right,0] - Lx_plate)**2).sum()

    # stabilisering i y
    energy += 0.5 * K * ((xy[ids_left,1] - xy0[ids_left,1]).mean()**2)
    energy += 0.5 * K * ((xy[ids_right,1] - xy0[ids_right,1]).mean()**2)

    return energy


def total_energy_jacobian(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy = xy_flat.reshape((-1,2))

    grad = -spring_forces(xy, edges, k, ell0_)

    grad[ids_left,0] += K * xy[ids_left,0]
    grad[ids_right,0] += K * (xy[ids_right,0] - Lx_plate)

    grad[ids_left,1] += K * (xy[ids_left,1] - xy0[ids_left,1]).mean()
    grad[ids_right,1] += K * (xy[ids_right,1] - xy0[ids_right,1]).mean()

    return grad.flatten()



def plot_mesh_colormap(xy, edges, ell0_):
    fig, ax = plt.subplots()

    strains = []
    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        ell = np.linalg.norm(xy[j] - xy[i])
        strains.append((ell - ell0) / ell0)

    strains = np.array(strains)
    max_s = np.max(np.abs(strains)) + 1e-12

    for (edge, strain) in zip(edges, strains):
        i, j = edge
        s = strain / max_s
        color = plt.cm.seismic((s+1)/2)

        ax.plot(xy[edge,0], xy[edge,1], color=color)

    ax.set_aspect('equal')
    plt.title("Red = tension, Blue = compression")
    plt.show()



np.random.seed(0)

xy, edges = make_random_mesh(N, L0, H0)

ell0_ = np.linalg.norm(xy[edges[:,0]] - xy[edges[:,1]], axis=1)
xy0 = xy.copy()

tol = 0.01

ids_left = np.where(xy[:,0] < tol)[0]
ids_right = np.where(xy[:,0] > L0 - tol)[0]
ids_bottom = np.where(xy[:,1] < tol)[0]
ids_top = np.where(xy[:,1] > H0 - tol)[0]
if len(ids_left) == 0 or len(ids_right) == 0:
    raise ValueError("No boundary nodes found!")
# lagring
eps_list = []
E_list = []
nu_list = []

# små deformasjoner (for E og ν)
f_values = np.linspace(0.0001, 0.5, 8)

for i, f in enumerate(f_values):
    Lx_plate = L0 * (1 + f)

    res = minimize(
        total_energy,
        xy.flatten(),
        args=(edges, k, K, ell0_, Lx_plate),
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-10
    )

    xy_eq = res.x.reshape((-1,2))

    # mål
    Lx = xy_eq[ids_right,0].mean() - xy_eq[ids_left,0].mean()
    Ly = xy_eq[ids_top,1].mean() - xy_eq[ids_bottom,1].mean()

    Fn = -K * (xy_eq[ids_right,0] - Lx_plate).sum()

    eps_x = (Lx - L0) / L0
    eps_y = (Ly - H0) / H0

    sigma = Fn / (B * Ly)

    E = sigma / eps_x
    nu = -eps_y / eps_x

    eps_list.append(eps_x)
    E_list.append(E)
    nu_list.append(nu)

    print(f"eps={eps_x:.4f}, E={E:.2f}, nu={nu:.3f}")
    
   # Bare plott noen få tilfeller



N_values = [20, 40, 60, 100, 150, 200]

E_vals = []
nu_vals = []

for N in N_values:

    xy, edges = make_random_mesh(N, L0, H0)

    ell0_ = np.linalg.norm(xy[edges[:,0]] - xy[edges[:,1]], axis=1)
    xy0 = xy.copy()

    # finn randnoder (bruk robust metode!)
    ids_left = np.argsort(xy[:,0])[:10]
    ids_right = np.argsort(xy[:,0])[-10:]
    ids_bottom = np.argsort(xy[:,1])[:10]
    ids_top = np.argsort(xy[:,1])[-10:]

    # liten deformasjon
    f = 1e-3
    Lx_plate = L0 * (1 + f)

    res = minimize(
        total_energy,
        xy.flatten(),
        args=(edges, k, K, ell0_, Lx_plate),
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-10
    )

    xy_eq = res.x.reshape((-1,2))

    # mål
    Lx = xy_eq[ids_right,0].mean() - xy_eq[ids_left,0].mean()
    Ly = xy_eq[ids_top,1].mean() - xy_eq[ids_bottom,1].mean()

    Fn = -K * (xy_eq[ids_right,0] - Lx_plate).sum()

    eps_x = (Lx - L0)/L0
    eps_y = (Ly - H0)/H0

    sigma = Fn / (B * Ly)

    E = sigma / eps_x
    nu = -eps_y / eps_x

    E_vals.append(E)
    nu_vals.append(nu)

plt.figure()
plt.plot(N_values, E_vals, 'o-')
plt.xlabel("N")
plt.ylabel("E")
plt.title("E vs N")
plt.grid()
plt.show()

plt.figure()
plt.plot(N_values, nu_vals, 'o-')
plt.xlabel("N")
plt.ylabel("nu")
plt.title("Poisson ratio vs N")
plt.grid()
plt.show()
