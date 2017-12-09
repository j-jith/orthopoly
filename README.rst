OrthoPoly
#########

**OrthoPoly** is a Python class for generating orthogonal polynomials with
respect to arbitrary probability density functions. The primary application of
this script is in performing *Arbitrary* Polynomial Chaos Expansion (PCE) in
uncertainty quantification studies.

Installation
============

Orthopoly is written in Python 3 and requires the following packages to be
present in your system:

- `numpy <https://pypi.python.org/pypi/numpy>`_

- `scipy <https://pypi.python.org/pypi/scipy>`_

Since OrthoPoly is a simple class, no special installation steps are
required. You can simply copy ``orthopoly.py`` to your working directory,
and start using it.

Usage
=====

OrthoPoly can be imported into your script as follows.

.. code:: python

   from orthopoly import OrthoPoly

To create an instance of OrthoPoly, you need to supply a probability density
function (pdf). The code below defines the pdf of a random variable created by
adding a uniform random variable and a normal random variable. This pdf is used
to create an instance of OrthoPoly, and orthogonal polynomials are generated
with respect to the supplied pdf.

.. code:: python

    def pdf(z, coeffs):
        mu, sigma, a, b = coeffs
        return 0.5/(b-a) * ( erf((z-a-mu)/sigma/sqrt(2)) - erf((z-b-mu)/sigma/sqrt(2)) )

    pp = OrthoPoly(pdf, margs=[0, 1, -1, 1])
    pp.gen_poly(5)

The function ``gen_poly()`` takes as an argument the largest order of the
polynomial to be generated, and populates the variable ``pp.poly`` with the
appropriate polynomials (which are ``numpy.polynomial.polynomial``).

For numerical integration with respect to these polynomials, the quadrature
points and weights can be obtained as follows.

.. code:: python

   points, weights = pp.get_quad_rule()

Numerical integration can also be performed directly using the ``quadrature()``
function. For instance, to compute the mean of the supplied pdf, one can do the
following.

.. code:: python

   mean = pp.quadrature(lambda x: x)


For more examples, please look at the ``__main__`` section of ``orthopoly.py``,
as well as the ``main,py`` script. ``main.py`` is a script which generates
orthogonal polynomials for the pdf of a random variable which is a sum of a
uniform and a normal random variable. It generates the polynomials, calculates
the quadrature points and weights, and writes them to an output file.

References
==========

To understand the mathematics behind OrthoPoly, please take a look at
`notes.pdf <notes/notes.pdf>`_. For a more in-depth reading, consult
[gautschi1982]_ and [golub1969]_.

.. [gautschi1982] Gautschi W. On Generating Orthogonal Polynomials. SIAM J Sci
   and Stat Comput 1982;3:289–317. doi:10.1137/0903018.

.. [golub1969] Golub GH, Welsch JH. Calculation of Gauss Quadrature Rules.
   Mathematics of Computation 1969;23:221. doi:10.2307/2004418.


