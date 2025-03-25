import numpy as np
import matplotlib.pyplot as plt

def plt_gradients(x, y, cost_fn, grad_fn):
    w_vals = [100, 200, 300]
    b = 100
    for w in w_vals:
        dj_dw, _ = grad_fn(x, y, w, b)
        slope = dj_dw
        y1 = cost_fn(x, y, w, b)
        plt.plot(w, y1, 'ro')
        plt.arrow(w, y1, -slope * 0.1, 0, head_width=10, color='blue')
    plt.title("Gradient direction w.r.t. w at b = 100")
    plt.xlabel("w")
    plt.ylabel("Cost")
    plt.grid(True)

def plt_contour_wgrad(x, y, history, ax=None, w_range=[0, 400, 5], b_range=[0, 400, 5], contours=None, resolution=5):
    w_vals = np.arange(*w_range)
    b_vals = np.arange(*b_range)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            b = B[i, j]
            Z[i, j] = 0.5 * np.mean((w * x + b - y) ** 2)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    CS = ax.contour(W, B, Z, levels=contours or 20)
    ax.clabel(CS, inline=1, fontsize=10)
    hist = np.array(history)
    ax.plot(hist[:, 0], hist[:, 1], 'ro-', markersize=3)
    ax.set_title("Gradient Descent Path")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.grid(True)

def plt_divergence(p_hist, J_hist, x_train, y_train):
    w_vals = [p[0] for p in p_hist]
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    ax1.plot(w_vals, label="w values")
    ax1.set_title("Weight oscillation (divergence)")
    ax1.set_ylabel("w value")
    ax1.set_xlabel("Iteration")
    ax1.legend()
    ax1.grid(True)
