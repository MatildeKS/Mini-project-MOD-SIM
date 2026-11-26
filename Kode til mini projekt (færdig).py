# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:17:31 2026

@author: matil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Known parameters
# --------------------------------------------------

dt = 0.00667
g = 9.82
l = 0.282
m = 0.250

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    "16112026_MATTEK_3.txt",
    header=None,
    names=["current", "theta"]
)

current = df["current"].to_numpy()
theta = df["theta"].to_numpy()

# Shift angle by 2*pi
theta = theta + 2*np.pi

# Keep samples where there are two values
current = current[874:14395]
theta = theta[874:14395]

# --------------------------------------------------
# Derivatives
# --------------------------------------------------

theta_dot = np.gradient(theta, dt)
theta_ddot = np.gradient(theta_dot, dt)

# --------------------------------------------------
# Least-squares matrix
# --------------------------------------------------

A = np.column_stack([
    current,
    -m * g * l * np.sin(theta),
    -theta_dot,
    -np.sign(theta_dot)
])

y = theta_ddot


# --------------------------------------------------
# Parameter estimation
# --------------------------------------------------

x = np.linalg.pinv(A) @ y

KT_over_J = x[0]
invJ = x[1]
alpha_over_J = x[2]
Ff_over_J = x[3]

J = 1 / invJ
KT = KT_over_J * J
alpha = alpha_over_J * J
Ff = Ff_over_J * J

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Estimated parameters")
print("--------------------")
print(f"K_T   = {KT:.6e}  [N·m/A]")
print(f"J     = {J:.6e}  [kg·m²]")
print(f"alpha = {alpha:.6e}  [N·m·s/rad]")
print(f"F_f   = {Ff:.6e}  [N·m]")

# --------------------------------------------------
# Simulate nonlinear model
# --------------------------------------------------

theta_hat = np.zeros_like(theta)
theta_dot_hat = np.zeros_like(theta)

theta_hat[0] = theta[0]
theta_dot_hat[0] = theta_dot[0]

for k in range(len(theta)-1):

    theta_ddot_hat = (
        KT * current[k]
        - m*g*l*np.sin(theta_hat[k])
        - alpha*theta_dot_hat[k]
        - Ff*np.sign(theta_dot_hat[k])
    ) / J

    theta_dot_hat[k+1] = (
        theta_dot_hat[k]
        + dt * theta_ddot_hat
    )

    theta_hat[k+1] = (
        theta_hat[k]
        + dt * theta_dot_hat[k]
    )

# --------------------------------------------------
# Moving average data
# --------------------------------------------------

window = 11

theta_ma = (
    pd.Series(theta)
    .rolling(window=window, center=True)
    .mean()
    .bfill()
    .ffill()
    .to_numpy()
)

theta_dot_ma = np.gradient(theta_ma, dt)

# --------------------------------------------------
# RMSE angle (root mean square error)
# --------------------------------------------------

rmse_theta = np.sqrt(
    np.mean((theta - theta_hat)**2)
)

rmse_theta_ma = np.sqrt(
    np.mean((theta_ma - theta_hat)**2)
)

rmse_theta_dot_ma = np.sqrt(
    np.mean((theta_dot_ma - theta_dot_hat)**2)
)

rmse_theta_dot = np.sqrt(
    np.mean((theta_dot - theta_dot_hat)**2)
)

print()
print(f"Theta RMSE = {rmse_theta:.6e}")

print(f"Theta_ma RMSE = {rmse_theta_ma:.6e}")

print(f"Theta_dot_ma RMSE = {rmse_theta_dot_ma:.6e}")

print(f"Theta_dot RMSE = {rmse_theta_dot:.6e}")

# --------------------------------------------------
# Plot angle
# --------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    theta_hat,
    label="estimated angle",
    linewidth=2
)

plt.plot(
    theta_ma,
    label="ma data",
    linewidth=2
)

plt.plot(
    theta,
    label="raw data",
    alpha=0.6
)

plt.xlabel("Sample")
plt.ylabel(r"$\theta$ [rad]")
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------
# Plot angular velocity
# --------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    theta_dot_hat,
    label="estimated angular velocity",
    linewidth=2
)

plt.plot(
    theta_dot_ma,
    label="ma data",
    linewidth=2
)

plt.plot(
    theta_dot,
    label="raw data",
    alpha=0.4
)


plt.xlabel("Sample")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")
plt.legend()
plt.grid()
plt.show()
