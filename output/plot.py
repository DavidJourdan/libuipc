import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys
from pathlib import Path


def load_csv(filepath: str) -> dict:
    data = np.genfromtxt(filepath, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def plot_fem_energy(filepath: str, save=False):
    d = load_csv(filepath)

    frames    = d["frame"]
    lambdas = d["lambda"]
    energy = d["energy"]
    force = d["force"]
    position = d["position"]
    hessian = d["hessian"]

    plt.plot(lambdas, energy / 4.33)
    plt.plot(lambdas, force / 4.33)

    deriv = (force[1:] - force[:-1]) / (lambdas[1:] - lambdas[:-1])

    print(deriv[1:49])
    print(deriv[50:][::-1])

    # plt.plot(lambdas[1:], deriv)
    
    
    plt.plot(lambdas, hessian / 4.33)

    # ax = plt.gca()
    # ax.set_xlim([0, 2])
    # ax.set_ylim([0, 2])

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os 
    
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    plot_fem_energy(os.path.join(folder, "fem_energy.csv"))