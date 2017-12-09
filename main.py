#!/usr/bin/env python
from __future__ import print_function, division

import sys
import csv

from orthopoly import OrthoPoly

from scipy.special import erf
from math import sqrt

def pdf(z, coeffs):
    mu, sigma, a, b = coeffs
    return 0.5/(b-a) * ( erf((z-a-mu)/sigma/sqrt(2)) - erf((z-b-mu)/sigma/sqrt(2)) )

def write_to_csv(coeffs, locs, weights):

    order = len(coeffs)

    # save to csv file
    with open('poly.csv', 'w') as f:
        writer = csv.writer(f)

        # header
        header = []
        for i in range(order):
            header.append('c{}'.format(i))
        header += ['q', 'w']

        writer.writerow(header)

        for i in range(order):
            writer.writerow(coeffs[i].tolist() + [0]*(order-i-1)+ [locs[i], weights[i]])

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Usage')
        print('=====')
        print('main.py mu sigma a b order')
        print()
        print('Arguments')
        print('=========')
        print('mu   : mean of Gaussian pdf')
        print('sigma: std of Gaussian pdf')
        print('a    : lower bound of Uniform pdf')
        print('b    : upper bound of Uniform pdf')
        print('degree: max. degree of polynomials')
        print()
        print('Note')
        print('====')
        print('Output will also be written to the file "poly.csv"')
    else:
        mu = float(sys.argv[1])
        sigma= float(sys.argv[2])
        a = float(sys.argv[3])
        b = float(sys.argv[4])
        order = int(sys.argv[5])

        #
        pp = OrthoPoly(pdf, margs=[mu, sigma, a, b])
        pp.gen_poly(order)

        qloc, ww = pp.get_quad_rule()

        # print output
        print('Coefficients')
        print('============')
        for cc in pp.poly:
            print(cc)
        print()
        print('Quad. points')
        print('============')
        print(qloc)
        print()
        print('Weights')
        print('=======')
        print(ww)

        write_to_csv(pp.poly, qloc, ww)
