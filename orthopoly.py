from __future__ import print_function, division

import scipy.integrate as integrate
import numpy as np
import numpy.polynomial.polynomial as nppoly

from scipy.special import erf
from math import sqrt

class OrthoPoly(object):

    def __init__(self, measure, **kwargs):
        self.measure = measure
        self.measure_args = kwargs.get('margs', None)

        self.poly = []
        self.jacobi = None
        self.order = None

        self.epsrel = kwargs.get('epsrel', 1e-6)
        self.intlims = kwargs.get('intlims', [-np.inf, np.inf])

    def gen_poly(self, n):
        self.order = n

        # zeroth polynomial
        self.poly = [nppoly.polyone]
        alpha = [self.get_alpha(self.poly[0])]
        beta = [1.]

        # first polynomial
        self.poly.append(nppoly.polymulx(self.poly[0]))
        self.poly[1] = nppoly.polyadd(self.poly[1], -alpha[0]*self.poly[0])
        alpha.append(self.get_alpha(self.poly[1]))
        beta.append(self.get_beta(self.poly[1], self.poly[0]))

        # reccurence relation for other polynomials
        for i in range(2, n+1):
            p_i = nppoly.polymulx(self.poly[i-1])
            p_i = nppoly.polyadd(p_i, -alpha[i-1] * self.poly[i-1])
            p_i = nppoly.polyadd(p_i, -beta[i-1] * self.poly[i-2])

            self.poly.append(p_i)

            alpha.append(self.get_alpha(self.poly[i]))
            beta.append(self.get_beta(self.poly[i], self.poly[i-1]))

        # normalise polynomials
        for i in range(len(self.poly)):
            self.poly[i] = self.poly[i] / np.prod(beta[:i])

        # create Jacobi matrix
        self.jacobi = (np.diag(np.sqrt(beta[1:]), -1)
                + np.diag(alpha, 0)
                + np.diag(np.sqrt(beta[1:]), 1))

    def eval(self, x, **kwargs):
        n = kwargs.get('i', None)

        if n == None:
            y = np.zeros((len(x), self.order))
            for i in range(self.order):
                y[:, i] = nppoly.polyval(x, self.poly[i])
            return y
        else:
            return nppoly.polyval(x, self.poly[n])

    def integrate(self, p, lim1, lim2):
        if self.measure_args:
            return integrate.quad(lambda x: nppoly.polyval(x, p) *
                    self.measure(x, self.measure_args), lim1, lim2)[0]
        else:
            return integrate.quad(lambda x: nppoly.polyval(x, p) *
                    self.measure(x), lim1, lim2)[0]

    def get_alpha(self, p):
        p2 = nppoly.polypow(p, 2)
        xp2 = nppoly.polymulx(p2)

        return (self.integrate(xp2, self.intlims[0], self.intlims[1]) /
                self.integrate(p2, self.intlims[0], self.intlims[1]))

    def get_beta(self, p, p0):
        p2 = nppoly.polypow(p, 2)
        p02 = nppoly.polypow(p0, 2)

        return (self.integrate(p2, self.intlims[0], self.intlims[1]) /
                self.integrate(p02, self.intlims[0], self.intlims[1]))

    def get_quad_rule(self):
        if self.jacobi.any():
            S, U = np.linalg.eigh(self.jacobi)
            locs = S
            weights = U[0, :]**2
            return locs, weights
        else:
            print('Run gen_poly() first!')
            return None

    def quadrature(self, func, **kwargs):
        args = kwargs.get('args', None)
        order = kwargs.get('order', self.order)

        if not len(self.poly):
            self.gen_poly(order)

        x, w = self.get_quad_rule()

        I = 0
        if args:
            for i in range(order+1):
                I = I + func(x[i], args)*w[i]
        else:
            for i in range(order+1):
                I = I + func(x[i])*w[i]
        return I

def pdf(z, coeffs):
    mu, sigma, a, b = coeffs
    return 0.5/(b-a) * ( erf((z-a-mu)/sigma/sqrt(2)) - erf((z-b-mu)/sigma/sqrt(2)) )


if __name__ == '__main__':

    coeffs = [0, 1, -1, 1]
    pp = OrthoPoly(pdf, margs=coeffs)
    pp.gen_poly(20)

    import matplotlib.pyplot as plt
    xx = np.linspace(-1, 1, 50)
    for p_i in pp.poly:
        plt.plot(xx, nppoly.polyval(xx, p_i))
    plt.show()

    mean = [pp.quadrature(lambda x: x),
            integrate.quad(lambda x, cc: x*pdf(x, cc), -np.inf, np.inf, args=coeffs)[0]]
    print('Mean:')
    print(mean)

    var = [pp.quadrature(lambda x: (x-mean[0])**2),
           integrate.quad(lambda x, cc: (x-mean[1])**2 * pdf(x, cc), -np.inf, np.inf, args=coeffs)[0]]
    std = np.sqrt(var)
    print('Std. dev.:')
    print(std)

    skew = [pp.quadrature(lambda x: ((x-mean[0])/var[0])**3),
            integrate.quad(lambda x, cc: ((x-mean[1])/var[1])**3 * pdf(x, cc), -np.inf, np.inf, args=coeffs)[0]]
    print('Skew:')
    print(skew)
