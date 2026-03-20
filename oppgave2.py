import numpy as np
import matplotlib.pyplot as plt



def make_simple_mesh (a0):
    b0 = np.sqrt(3) * a0
    xy = np.array([[0, 0],
                    [b0, 0],
                    [b0/2, a0/2],
                    [0, a0],
                    [b0, a0]])
    edges = np.array ([[0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 2],
                        [1, 4],
                        [2, 3],
                        [2, 4],
                        [3, 4]])
    return xy, edges


def plot_mesh(xy, edges):
    fig, ax = plt.subplots(1,1)
    for edge in edges:
        ax.plot(xy[edge, 0], xy[edge, 1], 'k')
        ell0_ = np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]], axis=1)
    ax.scatter(xy[:,0], xy[:,1], s=20, color='red')
    ax.set_aspect('equal')
    plt.show()

plot_mesh(make_simple_mesh(0.01)[0], make_simple_mesh(0.01)[1])
