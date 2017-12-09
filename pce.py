#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from orthopoly import OrthoPoly

from scipy.special import erf
from math import sqrt

def pdf(z, coeffs):
    mu, sigma, a, b = coeffs
    return 0.5/(b-a) * ( erf((z-a-mu)/sigma/sqrt(2)) - erf((z-b-mu)/sigma/sqrt(2)) )

def pdf_germ(z, alpha):
    return 0.25/alpha * ( erf((z+alpha)/sqrt(2)) - erf((z-alpha)/sqrt(2)) )

def generate_samples(N, mean, cov, bias):
    return mean * (1 + cov*np.random.randn(N) + bias*np.random.rand(N))

def integrate(y1, y2, w):
    return np.sum(y1*y2*w)

def generate_germ(N, alpha):
    return np.random.randn(N) + alpha*np.random.uniform(low=-1., high=1., size=N)

def get_pce_coeffs(y, h1, h2, w1, w2):
    H = np.kron(h1, h2)
    W = np.kron(w1, w2)

    D = np.diag(np.dot(H.T, np.dot(np.diag(W), H)))

    y_i = np.dot(H.T, np.dot(np.diag(W), y))
    y_i = np.dot(np.diag(1./D), y_i)

    return y_i

def get_pce_samples(yi, h1, h2):
    n1 = h1.shape[1]; n2 = h2.shape[1]
    k = 0
    y = 0
    for i in range(n1):
        for j in range(n2):
            y = y + h1[:, i]*h2[:, j]*yi[k]
            k = k + 1

    return y

def tensor_quad_rule(q1, q2, w1, w2):
    qq1, qq2 = np.meshgrid(q1, q2)
    ww1, ww2 = np.meshgrid(w1, w2)
    return (qq1.flatten(), qq2.flatten(),
            ww1.flatten(), ww2.flatten())

if __name__ == '__main__':

    # MC samples
    N = 5000

    # PCE order
    order = 2

    # Measurement uncertainty
    ## Pressure
    p_mean = 5e6
    p_cov = 0.005
    p_bias = 0.01
    ## Temperature
    #T_mean = 273.15 + 50
    #T_cov = 0.0075/2.
    #T_bias = 0.0075

    # Generate MC samples
    g = generate_germ(N, p_bias/p_cov)

    # Generate PCE polynomials
    poly = OrthoPoly(pdf_germ, margs=p_bias/p_cov)
    poly.gen_poly(order)

    # Quadrature points and weights
    q0, w0 = poly.get_quad_rule()
    q1, q2, w1, w2 = tensor_quad_rule(q0, q0, w0, w0)

    # Generate samples at quadrature points
    p_q = p_mean*(1 + p_cov*q1)

    # Evaluate polynomials at quadrature points
    hp_q = poly.eval(q0)
    hT_q = poly.eval(q0)

    # Get PCE coeffs
    p_i = get_pce_coeffs(p_q, hp_q, hT_q, w0, w0)

    # Reconstruct PCE samples
    hp = poly.eval(g)
    hT = poly.eval(g)
    p_pce = get_pce_samples(p_i, hp, hT)

    # Compute PDF
    xx = np.linspace(np.min(p_pce), np.max(p_pce), 500)
    ff = pdf(xx, [p_mean, p_mean*p_cov, -p_mean*p_bias, p_mean*p_bias])

    # Plot
    fig, ax = plt.subplots()
    ax.hist(p_pce, 50, normed=True)
    ax.plot(xx, ff, 'r-')
    plt.show()

