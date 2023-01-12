import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash

def plotOnSimplex(traj, fname):
    """
    Input:
    ------------------
    Matrix of size  trajectory_count, 3, step_count
    
    """
    traj = np.transpose(traj, (1, 2, 0))
    print(traj.shape)
    f, ax = plt.subplots(1, 1)

    proj = np.array(
        [[-1 * np.cos(30 / 360 * 2 * np.pi), np.cos(30 / 360 * 2 * np.pi), 0],
         [-1 * np.sin(30 / 360 * 2 * np.pi), -1 * np.sin(30 / 360 * 2 * np.pi), 1]
         ])

    ts = np.linspace(0, 1, 10000)

    e1 = proj @ np.array([ts, 1 - ts, 0 * ts])
    e2 = proj @ np.array([0 * ts, ts, 1 - ts])
    e3 = proj @ np.array([ts, 0 * ts, 1 - ts])

    ax.plot(e1[0], e1[1], 'k-', alpha = 0.3)
    ax.plot(e2[0], e2[1], 'k-', alpha = 0.3)
    ax.plot(e3[0], e3[1], 'k-', alpha = 0.3)

    for i in range(traj.shape[2]):
        d = proj @ traj[:, :, i]
        ax.plot(d[0], d[1], '--', alpha=0.6)
        ax.scatter(d[0, -1], d[1, -1], marker='+')
    plt.savefig(fname)
    plt.axis('off')
    plt.show()
