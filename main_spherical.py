"""
LS-ASM:

This is the main executive script used for the diffraction field calculation using LS-ASM.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Wei:23,
title       = {Modeling Off-Axis Diffraction with the Least-Sampling Angular Spectrum Method},
author      = {Haoyu Wei and Xin Liu and Xiang Hao and Edmund Y. Lam and Yifan Peng},
journal     = {Optica},
volume      = {10}, number = {7}, pages = {959--962},
publisher   = {Optica Publishing Group},
year        = {2023}, 
month       = {Jul}, 
doi         = {10.1364/OPTICA.490223}
}

-----

$ python main.py
"""

import numpy as np
import time
import glob
from input_field import InputField


from LSASM import LeastSamplingASM
import numpy as np
import time
import glob
from utils import remove_linear_phase, snr
from input_field import InputField

device = "cuda:0"

from matplotlib import pyplot as plt


def save_image(filename, image_array, cmap="gray"):
    """
    Save an image using matplotlib.
    """
    plt.imsave(filename, image_array, cmap=cmap)


def run_diffraction(
    thetaX,
    thetaY,
    wvls=500e-9,
    f=35e-3,
    z0=1.7,
    s_LSASM=1.5,
    s_RS=4,
    Mx=512,
    My=512,
    result_folder="results",
    RS_folder="RS",
    calculate_SNR=False,
    device="cuda:0",
):

    # Derived parameters
    k = 2 * np.pi / wvls
    zf = 1 / (1 / f - 1 / z0)
    z = zf
    r = f / 32  # same as f/16/2

    # Define observation window
    l = r * 0.25
    thetaX_rad = thetaX / 180 * np.pi
    thetaY_rad = thetaY / 180 * np.pi
    sqrt_term = np.sqrt(1 - np.sin(thetaX_rad) ** 2 - np.sin(thetaY_rad) ** 2)
    xc = -z * np.sin(thetaX_rad) / sqrt_term
    yc = -z * np.sin(thetaY_rad) / sqrt_term

    x = np.linspace(-l / 2 + xc, l / 2 + xc, Mx, endpoint=True)
    y = np.linspace(-l / 2 + yc, l / 2 + yc, My, endpoint=True)
    print(f"Observation window diameter = {l}.")

    print("----------------- Propagating with LSASM -----------------")
    Uin = InputField("2", wvls, (thetaX, thetaY), r, z0, f, zf, s_LSASM)
    prop2 = LeastSamplingASM(Uin, x, y, z, device)
    path = (
        f"{result_folder}/LSASM({len(Uin.xi)},{len(prop2.fx)})-{thetaX}-{s_LSASM:.2f}"
    )

    start = time.time()
    U2 = prop2(Uin.E0)
    end = time.time()
    runtime = end - start
    print(f"Time elapsed for LSASM: {runtime:.2f}")

    save_image(f"{path}.png", abs(U2), cmap="gray")
    phase = remove_linear_phase(np.angle(U2), thetaX, thetaY, x, y, k)
    save_image(f"{path}-Phi.png", phase, cmap="twilight")

    if calculate_SNR:
        files = glob.glob(f"{RS_folder}/RS*-{thetaX}-{s_RS:.1f}.npy")
        if files:
            u_GT = np.load(files[0])
            print(f"SNR is {snr(U2, u_GT):.2f}")

    return U2, x, y, runtime


if __name__ == "__main__":
    U2_LSASM, x, y, runtime_LSASM = run_diffraction(0, 0)
    U3_LSASM, x, y, runtime_LSASM = run_diffraction(0, 0.02)

    difference = U3_LSASM * U3_LSASM.conj() - U2_LSASM * U2_LSASM.conj()
    log_diff = (np.log(U3_LSASM * U3_LSASM.conj() + 1e-9) - np.log(
        U2_LSASM * U2_LSASM.conj() + 1e-9
    )).real

    print(difference.max())
    print(difference.min())
    print(difference)
    save_image(
        "log_diff.png",
        log_diff,
        cmap="twilight",
    )
    save_image("difference.png", difference.real, cmap="twilight")
