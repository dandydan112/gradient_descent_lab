import math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_contour_wgrad, plt_divergence, plt_gradients


# Data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    return cost / (2 * m)

# Gradient function
def compute_gradient(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw += (f_wb - y[i]) * x[i] 
        dj_db += (f_wb - y[i]) 
    return dj_dw / m, dj_db / m

# Gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    J_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]:.2e}, dj_dw {dj_dw:.3e}, dj_db {dj_db:.3e}, w {w:.3e}, b {b:.3e}")
    
    return w, b, J_history, p_history

# Run training
w_init = 0
b_init = 0
iterations = 10000
alpha = 0.01

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

print(f"\nFinal parameters: w = {w_final:.4f}, b = {b_final:.4f}")
print(f"Prediction for 1000 sqft: {w_final*1 + b_final:.1f}k")
print(f"Prediction for 1200 sqft: {w_final*1.2 + b_final:.1f}k")
print(f"Prediction for 2000 sqft: {w_final*2 + b_final:.1f}k")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel("Cost")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost")
ax2.set_xlabel("Iteration")
plt.show()

# Visualization tools
plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt_contour_wgrad(x_train, y_train, p_hist)
plt_contour_wgrad(x_train, y_train, p_hist, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5], contours=[1, 5, 10, 20], resolution=0.5)

# Try divergence
w_init = 0
b_init = 0
alpha = 0.8
iterations = 10

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

plt_divergence(p_hist, J_hist, x_train, y_train)
