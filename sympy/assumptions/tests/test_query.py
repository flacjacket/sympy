from sympy.abc import t, w, x, y, z
from sympy.assumptions import (ask, AssumptionsContext, global_assumptions, Q,
                               register_handler, remove_handler)
from sympy.assumptions.ask import (compute_known_facts, known_facts_cnf,
                                   known_facts_dict)
from sympy.assumptions.handlers import AskHandler
from sympy.core import I, Integer, oo, pi, Rational, S, symbols
from sympy.functions import Abs, cos, exp, im, log, re, sign, sin, sqrt
from sympy.logic import Equivalent, Implies, Xor
from sympy.utilities.pytest import raises, XFAIL, slow

def test_int_1():
    z = 1
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(True)
    assert ask(Q.rational(z))         == S(True)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(True)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(True)

def test_float_1():
    z = 1.0
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(True)
    assert ask(Q.rational(z))         == S(True)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(True)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(True)

    z = 7.2123
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(True)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_zero_0():
    z = Integer(0)
    assert ask(Q.nonzero(z))          == S(False)
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(True)
    assert ask(Q.rational(z))         == S(True)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(False)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(True)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(True)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_negativeone():
    z = Integer(-1)
    assert ask(Q.nonzero(z))          == S(True)
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(True)
    assert ask(Q.rational(z))         == S(True)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(False)
    assert ask(Q.negative(z))         == S(True)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(True)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_infinity():
    assert ask(Q.commutative(oo))     == S(True)
    assert ask(Q.integer(oo))         == S(False)
    assert ask(Q.rational(oo))        == S(False)
    assert ask(Q.real(oo))            == S(False)
    assert ask(Q.extended_real(oo))   == S(True)
    assert ask(Q.complex(oo))         == S(False)
    assert ask(Q.irrational(oo))      == S(False)
    assert ask(Q.imaginary(oo))       == S(False)
    assert ask(Q.positive(oo))        == S(True)
    assert ask(Q.negative(oo))        == S(False)
    assert ask(Q.even(oo))            == S(False)
    assert ask(Q.odd(oo))             == S(False)
    assert ask(Q.bounded(oo))         == S(False)
    assert ask(Q.infinitesimal(oo))   == S(False)
    assert ask(Q.prime(oo))           == S(False)
    assert ask(Q.composite(oo))       == S(False)

def test_neg_infinity():
    mm = S.NegativeInfinity
    assert ask(Q.commutative(mm))    == S(True)
    assert ask(Q.integer(mm))        == S(False)
    assert ask(Q.rational(mm))       == S(False)
    assert ask(Q.real(mm))           == S(False)
    assert ask(Q.extended_real(mm))  == S(True)
    assert ask(Q.complex(mm))        == S(False)
    assert ask(Q.irrational(mm))     == S(False)
    assert ask(Q.imaginary(mm))      == S(False)
    assert ask(Q.positive(mm))       == S(False)
    assert ask(Q.negative(mm))       == S(True)
    assert ask(Q.even(mm))           == S(False)
    assert ask(Q.odd(mm))            == S(False)
    assert ask(Q.bounded(mm))        == S(False)
    assert ask(Q.infinitesimal(mm))  == S(False)
    assert ask(Q.prime(mm))          == S(False)
    assert ask(Q.composite(mm))      == S(False)

def test_nan():
    nan = S.NaN
    assert ask(Q.commutative(nan))   == S(True)
    assert ask(Q.integer(nan))       == S(False)
    assert ask(Q.rational(nan))      == S(False)
    assert ask(Q.real(nan))          == S(False)
    assert ask(Q.extended_real(nan)) == S(False)
    assert ask(Q.complex(nan))       == S(False)
    assert ask(Q.irrational(nan))    == S(False)
    assert ask(Q.imaginary(nan))     == S(False)
    assert ask(Q.positive(nan))      == S(False)
    assert ask(Q.nonzero(nan))       == S(True)
    assert ask(Q.even(nan))          == S(False)
    assert ask(Q.odd(nan))           == S(False)
    assert ask(Q.bounded(nan))       == S(False)
    assert ask(Q.infinitesimal(nan)) == S(False)
    assert ask(Q.prime(nan))         == S(False)
    assert ask(Q.composite(nan))     == S(False)

def test_Rational_number():
    r = Rational(3,4)
    assert ask(Q.commutative(r))      == S(True)
    assert ask(Q.integer(r))          == S(False)
    assert ask(Q.rational(r))         == S(True)
    assert ask(Q.real(r))             == S(True)
    assert ask(Q.complex(r))          == S(True)
    assert ask(Q.irrational(r))       == S(False)
    assert ask(Q.imaginary(r))        == S(False)
    assert ask(Q.positive(r))         == S(True)
    assert ask(Q.negative(r))         == S(False)
    assert ask(Q.even(r))             == S(False)
    assert ask(Q.odd(r))              == S(False)
    assert ask(Q.bounded(r))          == S(True)
    assert ask(Q.infinitesimal(r))    == S(False)
    assert ask(Q.prime(r))            == S(False)
    assert ask(Q.composite(r))        == S(False)

    r = Rational(1,4)
    assert ask(Q.positive(r))         == S(True)
    assert ask(Q.negative(r))         == S(False)

    r = Rational(5,4)
    assert ask(Q.negative(r))         == S(False)
    assert ask(Q.positive(r))         == S(True)

    r = Rational(5,3)
    assert ask(Q.positive(r))         == S(True)
    assert ask(Q.negative(r))         == S(False)

    r = Rational(-3,4)
    assert ask(Q.positive(r))         == S(False)
    assert ask(Q.negative(r))         == S(True)

    r = Rational(-1,4)
    assert ask(Q.positive(r))         == S(False)
    assert ask(Q.negative(r))         == S(True)

    r = Rational(-5,4)
    assert ask(Q.negative(r))         == S(True)
    assert ask(Q.positive(r))         == S(False)

    r = Rational(-5,3)
    assert ask(Q.positive(r))         == S(False)
    assert ask(Q.negative(r))         == S(True)

def test_sqrt_2():
    z = sqrt(2)
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_pi():
    z = S.Pi
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = S.Pi + 1
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = 2*S.Pi
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = S.Pi ** 2
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = (1+S.Pi) ** 2
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_E():
    z = S.Exp1
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(True)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(True)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(True)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

def test_I():
    z = I
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(False)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(True)
    assert ask(Q.positive(z))         == S(False)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = 1 + I
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(False)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(False)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

    z = I*(1+I)
    assert ask(Q.commutative(z))      == S(True)
    assert ask(Q.integer(z))          == S(False)
    assert ask(Q.rational(z))         == S(False)
    assert ask(Q.real(z))             == S(False)
    assert ask(Q.complex(z))          == S(True)
    assert ask(Q.irrational(z))       == S(False)
    assert ask(Q.imaginary(z))        == S(False)
    assert ask(Q.positive(z))         == S(False)
    assert ask(Q.negative(z))         == S(False)
    assert ask(Q.even(z))             == S(False)
    assert ask(Q.odd(z))              == S(False)
    assert ask(Q.bounded(z))          == S(True)
    assert ask(Q.infinitesimal(z))    == S(False)
    assert ask(Q.prime(z))            == S(False)
    assert ask(Q.composite(z))        == S(False)

@slow
def test_bounded():
    x, y, z = symbols('x,y,z')
    assert ask(Q.bounded(x)) == S(None)
    assert ask(Q.bounded(x), Q.bounded(x)) == S(True)
    assert ask(Q.bounded(x), Q.bounded(y)) == S(None)
    assert ask(Q.bounded(x), Q.complex(x)) == S(None)

    assert ask(Q.bounded(x+1)) == S(None)
    assert ask(Q.bounded(x+1), Q.bounded(x)) == S(True)
    a = x + y
    x, y = a.args
    # B + B
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(x)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(x) & Q.positive(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(x) & ~Q.positive(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & ~Q.positive(x) & Q.positive(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & ~Q.positive(x) & ~Q.positive(y)) == S(True)
    # B + U
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(x)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(x) & Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(x) & ~Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & ~Q.positive(x) & Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & ~Q.positive(x) & ~Q.positive(y)) == S(False)
    # B + ?
    assert ask(Q.bounded(a), Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(x)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(x) & ~Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.positive(x) & ~Q.positive(y)) == S(None)
    # U + U
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(x)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(x) & Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(x) & ~Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & ~Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & ~Q.positive(x) & ~Q.positive(y)) == S(False)
    # U + ?
    assert ask(Q.bounded(a), ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & Q.positive(x)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & Q.positive(x) & Q.positive(y)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(y) & Q.positive(x) & ~Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & ~Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & ~Q.positive(x) & ~Q.positive(y)) == S(False)
    # ? + ?
    assert ask(Q.bounded(a),) == S(None)
    assert ask(Q.bounded(a),Q.positive(x)) == S(None)
    assert ask(Q.bounded(a),Q.positive(y)) == S(None)
    assert ask(Q.bounded(a),Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a),Q.positive(x) & ~Q.positive(y)) == S(None)
    assert ask(Q.bounded(a),~Q.positive(x) & Q.positive(y)) == S(None)
    assert ask(Q.bounded(a),~Q.positive(x) & ~Q.positive(y)) == S(None)
    a = x + y + z
    x, y, z = a.args
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.negative(z) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.positive(z) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.positive(z) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & ~Q.bounded(z))== S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.negative(z))== S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.negative(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)& ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)& Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)& Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)& Q.positive(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(z) & Q.bounded(z))== S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.negative(z) & ~Q.bounded(z))== S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z)& Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z)& ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.positive(z)& ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.negative(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& Q.positive(z) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.bounded(y)& Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.negative(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.negative(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.negative(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & ~Q.bounded(y)& Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & ~Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(y) & ~Q.bounded(y) & Q.positive(z)) == S(False)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & ~Q.bounded(x) & Q.positive(y) & Q.positive(z)) == S(False)
    assert ask(Q.bounded(a), Q.negative(x) & Q.negative(y) & Q.negative(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.negative(y)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.negative(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.negative(x) & Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a)) == S(None)
    assert ask(Q.bounded(a), Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(y) & Q.positive(z)) == S(None)
    assert ask(Q.bounded(a), Q.positive(x) & Q.positive(y) & Q.positive(z)) == S(None)

    x, y, z = symbols('x,y,z')
    assert ask(Q.bounded(2*x)) == S(None)
    assert ask(Q.bounded(2*x), Q.bounded(x)) == S(True)
    a = x*y
    x, y = a.args
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.bounded(y)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a)) == S(None)
    a = x*y*z
    x, y, z = a.args
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & Q.bounded(z)) == S(True)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.bounded(y) & Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y) & ~Q.bounded(z)) == S(False)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(x)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(y) & Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y)) == S(None)
    assert ask(Q.bounded(a), Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(z) & Q.nonzero(x) & Q.nonzero(y) & Q.nonzero(z)) == S(None)
    assert ask(Q.bounded(a), ~Q.bounded(y) & ~Q.bounded(z) & Q.nonzero(x) & Q.nonzero(y) & Q.nonzero(z)) == S(False)

    x, y, z = symbols('x,y,z')
    assert ask(Q.bounded(x**2)) == S(None)
    assert ask(Q.bounded(2**x)) == S(None)
    assert ask(Q.bounded(2**x), Q.bounded(x)) == S(True)
    assert ask(Q.bounded(x**x)) == S(None)
    assert ask(Q.bounded(Rational(1,2) ** x)) == S(None)
    assert ask(Q.bounded(Rational(1,2) ** x), Q.positive(x)) == S(True)
    assert ask(Q.bounded(Rational(1,2) ** x), Q.negative(x)) == S(None)
    assert ask(Q.bounded(S(2) ** x), Q.negative(x)) == S(True)
    assert ask(Q.bounded(sqrt(x))) == S(None)
    assert ask(Q.bounded(2**x), ~Q.bounded(x))==False
    assert ask(Q.bounded(x**2), ~Q.bounded(x))==False

    # sign function
    assert ask(Q.bounded(sign(x))) == S(True)
    assert ask(Q.bounded(sign(x)), ~Q.bounded(x)) == S(True)

    # exponential functions
    assert ask(Q.bounded(log(x))) == S(None)
    assert ask(Q.bounded(log(x)), Q.bounded(x)) == S(True)
    assert ask(Q.bounded(exp(x))) == S(None)
    assert ask(Q.bounded(exp(x)), Q.bounded(x)) == S(True)
    assert ask(Q.bounded(exp(2))) == S(True)

    # trigonometric functions
    assert ask(Q.bounded(sin(x))) == S(True)
    assert ask(Q.bounded(sin(x)), ~Q.bounded(x)) == S(True)
    assert ask(Q.bounded(cos(x))) == S(True)
    assert ask(Q.bounded(cos(x)), ~Q.bounded(x)) == S(True)
    assert ask(Q.bounded(2*sin(x))) == S(True)
    assert ask(Q.bounded(sin(x)**2)) == S(True)
    assert ask(Q.bounded(cos(x)**2)) == S(True)
    assert ask(Q.bounded(cos(x) + sin(x))) == S(True)

@XFAIL
def test_bounded_xfail():
    """We need to support relations in ask for this to work"""
    assert ask(Q.bounded(sin(x)**x)) == S(True)
    assert ask(Q.bounded(cos(x)**x)) == S(True)
    assert ask(Q.bounded(sin(x) ** x)) == S(True)

def test_commutative():
    """By default objects are Q.commutative that is why it returns True
    for both key=True and key=False"""
    assert ask(Q.commutative(x)) == S(True)
    assert ask(Q.commutative(x), ~Q.commutative(x)) == S(False)
    assert ask(Q.commutative(x), Q.complex(x)) == S(True)
    assert ask(Q.commutative(x), Q.imaginary(x)) == S(True)
    assert ask(Q.commutative(x), Q.real(x)) == S(True)
    assert ask(Q.commutative(x), Q.positive(x)) == S(True)
    assert ask(Q.commutative(x), ~Q.commutative(y))  == S(True)

    assert ask(Q.commutative(2*x)) == S(True)
    assert ask(Q.commutative(2*x), ~Q.commutative(x)) == S(False)

    assert ask(Q.commutative(x + 1)) == S(True)
    assert ask(Q.commutative(x + 1), ~Q.commutative(x)) == S(False)

    assert ask(Q.commutative(x**2)) == S(True)
    assert ask(Q.commutative(x**2), ~Q.commutative(x)) == S(False)

    assert ask(Q.commutative(log(x))) == S(True)

def test_complex():
    assert ask(Q.complex(x)) == S(None)
    assert ask(Q.complex(x), Q.complex(x)) == S(True)
    assert ask(Q.complex(x), Q.complex(y)) == S(None)
    assert ask(Q.complex(x), ~Q.complex(x)) == S(False)
    assert ask(Q.complex(x), Q.real(x)) == S(True)
    assert ask(Q.complex(x), ~Q.real(x)) == S(None)
    assert ask(Q.complex(x), Q.rational(x)) == S(True)
    assert ask(Q.complex(x), Q.irrational(x)) == S(True)
    assert ask(Q.complex(x), Q.positive(x)) == S(True)
    assert ask(Q.complex(x), Q.imaginary(x)) == S(True)

    # a+b
    assert ask(Q.complex(x+1), Q.complex(x)) == S(True)
    assert ask(Q.complex(x+1), Q.real(x)) == S(True)
    assert ask(Q.complex(x+1), Q.rational(x)) == S(True)
    assert ask(Q.complex(x+1), Q.irrational(x)) == S(True)
    assert ask(Q.complex(x+1), Q.imaginary(x)) == S(True)
    assert ask(Q.complex(x+1), Q.integer(x))  == S(True)
    assert ask(Q.complex(x+1), Q.even(x))  == S(True)
    assert ask(Q.complex(x+1), Q.odd(x))  == S(True)
    assert ask(Q.complex(x+y), Q.complex(x) & Q.complex(y)) == S(True)
    assert ask(Q.complex(x+y), Q.real(x) & Q.imaginary(y)) == S(True)

    # a*x +b
    assert ask(Q.complex(2*x+1), Q.complex(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.real(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.positive(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.rational(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.irrational(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.imaginary(x)) == S(True)
    assert ask(Q.complex(2*x+1), Q.integer(x))  == S(True)
    assert ask(Q.complex(2*x+1), Q.even(x))  == S(True)
    assert ask(Q.complex(2*x+1), Q.odd(x))  == S(True)

    # x**2
    assert ask(Q.complex(x**2), Q.complex(x)) == S(True)
    assert ask(Q.complex(x**2), Q.real(x)) == S(True)
    assert ask(Q.complex(x**2), Q.positive(x)) == S(True)
    assert ask(Q.complex(x**2), Q.rational(x)) == S(True)
    assert ask(Q.complex(x**2), Q.irrational(x)) == S(True)
    assert ask(Q.complex(x**2), Q.imaginary(x)) == S(True)
    assert ask(Q.complex(x**2), Q.integer(x))  == S(True)
    assert ask(Q.complex(x**2), Q.even(x))  == S(True)
    assert ask(Q.complex(x**2), Q.odd(x))  == S(True)

    # 2**x
    assert ask(Q.complex(2**x), Q.complex(x)) == S(True)
    assert ask(Q.complex(2**x), Q.real(x)) == S(True)
    assert ask(Q.complex(2**x), Q.positive(x)) == S(True)
    assert ask(Q.complex(2**x), Q.rational(x)) == S(True)
    assert ask(Q.complex(2**x), Q.irrational(x)) == S(True)
    assert ask(Q.complex(2**x), Q.imaginary(x)) == S(True)
    assert ask(Q.complex(2**x), Q.integer(x))  == S(True)
    assert ask(Q.complex(2**x), Q.even(x))  == S(True)
    assert ask(Q.complex(2**x), Q.odd(x))  == S(True)
    assert ask(Q.complex(x**y), Q.complex(x) & Q.complex(y)) == S(True)

    # trigonometric expressions
    assert ask(Q.complex(sin(x))) == S(True)
    assert ask(Q.complex(sin(2*x + 1))) == S(True)
    assert ask(Q.complex(cos(x))) == S(True)
    assert ask(Q.complex(cos(2*x+1))) == S(True)

    # exponential
    assert ask(Q.complex(exp(x))) == S(True)
    assert ask(Q.complex(exp(x))) == S(True)

    # Q.complexes
    assert ask(Q.complex(Abs(x))) == S(True)
    assert ask(Q.complex(re(x))) == S(True)
    assert ask(Q.complex(im(x))) == S(True)

def test_even():
    assert ask(Q.even(x)) == S(None)
    assert ask(Q.even(x), Q.integer(x)) == S(None)
    assert ask(Q.even(x), ~Q.integer(x)) == S(False)
    assert ask(Q.even(x), Q.rational(x)) == S(None)
    assert ask(Q.even(x), Q.positive(x)) == S(None)

    assert ask(Q.even(2*x)) == S(None)
    assert ask(Q.even(2*x), Q.integer(x)) == S(True)
    assert ask(Q.even(2*x), Q.even(x)) == S(True)
    assert ask(Q.even(2*x), Q.irrational(x)) == S(False)
    assert ask(Q.even(2*x), Q.odd(x)) == S(True)
    assert ask(Q.even(2*x), ~Q.integer(x)) == S(None)
    assert ask(Q.even(3*x), Q.integer(x)) == S(None)
    assert ask(Q.even(3*x), Q.even(x)) == S(True)
    assert ask(Q.even(3*x), Q.odd(x)) == S(False)

    assert ask(Q.even(x+1), Q.odd(x)) == S(True)
    assert ask(Q.even(x+1), Q.even(x)) == S(False)
    assert ask(Q.even(x+2), Q.odd(x)) == S(False)
    assert ask(Q.even(x+2), Q.even(x)) == S(True)
    assert ask(Q.even(7-x), Q.odd(x)) == S(True)
    assert ask(Q.even(7+x), Q.odd(x)) == S(True)
    assert ask(Q.even(x+y), Q.odd(x) & Q.odd(y)) == S(True)
    assert ask(Q.even(x+y), Q.odd(x) & Q.even(y)) == S(False)
    assert ask(Q.even(x+y), Q.even(x) & Q.even(y)) == S(True)

    assert ask(Q.even(2*x + 1), Q.integer(x)) == S(False)
    assert ask(Q.even(2*x*y), Q.rational(x) & Q.rational(x)) == S(None)
    assert ask(Q.even(2*x*y), Q.irrational(x) & Q.irrational(x)) == S(None)

    assert ask(Q.even(x+y+z), Q.odd(x) & Q.odd(y) & Q.even(z)) == S(True)
    assert ask(Q.even(x+y+z+t), Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) == S(None)

    assert ask(Q.even(Abs(x)), Q.even(x)) == S(True)
    assert ask(Q.even(Abs(x)), ~Q.even(x)) == S(None)
    assert ask(Q.even(re(x)), Q.even(x)) == S(True)
    assert ask(Q.even(re(x)), ~Q.even(x)) == S(None)
    assert ask(Q.even(im(x)), Q.even(x)) == S(True)
    assert ask(Q.even(im(x)), Q.real(x)) == S(True)

def test_extended_real():
    assert ask(Q.extended_real(x), Q.positive(x)) == S(True)
    assert ask(Q.extended_real(-x), Q.positive(x)) == S(True)
    assert ask(Q.extended_real(-x), Q.negative(x)) == S(True)

    assert ask(Q.extended_real(x+S.Infinity), Q.real(x)) == S(True)

def test_rational():
    assert ask(Q.rational(x), Q.integer(x)) == S(True)
    assert ask(Q.rational(x), Q.irrational(x)) == S(False)
    assert ask(Q.rational(x), Q.real(x)) == S(None)
    assert ask(Q.rational(x), Q.positive(x)) == S(None)
    assert ask(Q.rational(x), Q.negative(x)) == S(None)
    assert ask(Q.rational(x), Q.nonzero(x)) == S(None)

    assert ask(Q.rational(2*x), Q.rational(x)) == S(True)
    assert ask(Q.rational(2*x), Q.integer(x)) == S(True)
    assert ask(Q.rational(2*x), Q.even(x)) == S(True)
    assert ask(Q.rational(2*x), Q.odd(x)) == S(True)
    assert ask(Q.rational(2*x), Q.irrational(x)) == S(False)

    assert ask(Q.rational(x/2), Q.rational(x)) == S(True)
    assert ask(Q.rational(x/2), Q.integer(x)) == S(True)
    assert ask(Q.rational(x/2), Q.even(x)) == S(True)
    assert ask(Q.rational(x/2), Q.odd(x)) == S(True)
    assert ask(Q.rational(x/2), Q.irrational(x)) == S(False)

    assert ask(Q.rational(1/x), Q.rational(x)) == S(True)
    assert ask(Q.rational(1/x), Q.integer(x)) == S(True)
    assert ask(Q.rational(1/x), Q.even(x)) == S(True)
    assert ask(Q.rational(1/x), Q.odd(x)) == S(True)
    assert ask(Q.rational(1/x), Q.irrational(x)) == S(False)

    assert ask(Q.rational(2/x), Q.rational(x)) == S(True)
    assert ask(Q.rational(2/x), Q.integer(x)) == S(True)
    assert ask(Q.rational(2/x), Q.even(x)) == S(True)
    assert ask(Q.rational(2/x), Q.odd(x)) == S(True)
    assert ask(Q.rational(2/x), Q.irrational(x)) == S(False)

    # with multiple symbols
    assert ask(Q.rational(x*y), Q.irrational(x) & Q.irrational(y)) == S(None)
    assert ask(Q.rational(y/x), Q.rational(x) & Q.rational(y)) == S(True)
    assert ask(Q.rational(y/x), Q.integer(x) & Q.rational(y)) == S(True)
    assert ask(Q.rational(y/x), Q.even(x) & Q.rational(y)) == S(True)
    assert ask(Q.rational(y/x), Q.odd(x) & Q.rational(y)) == S(True)
    assert ask(Q.rational(y/x), Q.irrational(x) & Q.rational(y)) == S(False)

def test_imaginary():
    assert ask(Q.imaginary(x)) == S(None)
    assert ask(Q.imaginary(x), Q.real(x)) == S(False)
    assert ask(Q.imaginary(x), Q.prime(x)) == S(False)

    assert ask(Q.imaginary(x+1), Q.real(x)) == S(False)
    assert ask(Q.imaginary(x+1), Q.imaginary(x)) == S(False)
    assert ask(Q.imaginary(x+I), Q.real(x)) == S(False)
    assert ask(Q.imaginary(x+I), Q.imaginary(x)) == S(True)
    assert ask(Q.imaginary(x+y), Q.imaginary(x) & Q.imaginary(y)) == S(True)
    assert ask(Q.imaginary(x+y), Q.real(x) & Q.real(y)) == S(False)
    assert ask(Q.imaginary(x+y), Q.imaginary(x) & Q.real(y)) == S(False)
    assert ask(Q.imaginary(x+y), Q.complex(x) & Q.real(y)) == S(None)

    assert ask(Q.imaginary(I*x), Q.real(x)) == S(True)
    assert ask(Q.imaginary(I*x), Q.imaginary(x)) == S(False)
    assert ask(Q.imaginary(I*x), Q.complex(x)) == S(None)
    assert ask(Q.imaginary(x*y), Q.imaginary(x) & Q.real(y)) == S(True)

    assert ask(Q.imaginary(x+y+z), Q.real(x) & Q.real(y) & Q.real(z)) == S(False)
    assert ask(Q.imaginary(x+y+z), Q.real(x) & Q.real(y) & Q.imaginary(z)) == S(None)
    assert ask(Q.imaginary(x+y+z), Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) == S(False)

def test_infinitesimal():
    assert ask(Q.infinitesimal(x)) == S(None)
    assert ask(Q.infinitesimal(x), Q.infinitesimal(x)) == S(True)

    assert ask(Q.infinitesimal(2*x), Q.infinitesimal(x)) == S(True)
    assert ask(Q.infinitesimal(x*y), Q.infinitesimal(x)) == S(None)
    assert ask(Q.infinitesimal(x*y), Q.infinitesimal(x) & Q.infinitesimal(y)) == S(True)
    assert ask(Q.infinitesimal(x*y), Q.infinitesimal(x) & Q.bounded(y)) == S(True)

    assert ask(Q.infinitesimal(x**2), Q.infinitesimal(x)) == S(True)

def test_integer():
    assert ask(Q.integer(x)) == S(None)
    assert ask(Q.integer(x), Q.integer(x)) == S(True)
    assert ask(Q.integer(x), ~Q.integer(x)) == S(False)
    assert ask(Q.integer(x), ~Q.real(x)) == S(False)
    assert ask(Q.integer(x), ~Q.positive(x)) == S(None)
    assert ask(Q.integer(x), Q.even(x) | Q.odd(x)) == S(True)

    assert ask(Q.integer(2*x), Q.integer(x)) == S(True)
    assert ask(Q.integer(2*x), Q.even(x)) == S(True)
    assert ask(Q.integer(2*x), Q.prime(x)) == S(True)
    assert ask(Q.integer(2*x), Q.rational(x)) == S(None)
    assert ask(Q.integer(2*x), Q.real(x)) == S(None)
    assert ask(Q.integer(sqrt(2)*x), Q.integer(x)) == S(False)

    assert ask(Q.integer(x/2), Q.odd(x)) == S(False)
    assert ask(Q.integer(x/2), Q.even(x)) == S(True)
    assert ask(Q.integer(x/3), Q.odd(x)) == S(None)
    assert ask(Q.integer(x/3), Q.even(x)) == S(None)

def test_negative():
    assert ask(Q.negative(x), Q.negative(x)) == S(True)
    assert ask(Q.negative(x), Q.positive(x)) == S(False)
    assert ask(Q.negative(x), ~Q.real(x)) == S(False)
    assert ask(Q.negative(x), Q.prime(x)) == S(False)
    assert ask(Q.negative(x), ~Q.prime(x)) == S(None)

    assert ask(Q.negative(-x), Q.positive(x)) == S(True)
    assert ask(Q.negative(-x), ~Q.positive(x)) == S(None)
    assert ask(Q.negative(-x), Q.negative(x)) == S(False)
    assert ask(Q.negative(-x), Q.positive(x)) == S(True)

    assert ask(Q.negative(x-1), Q.negative(x)) == S(True)
    assert ask(Q.negative(x+y)) == S(None)
    assert ask(Q.negative(x+y), Q.negative(x)) == S(None)
    assert ask(Q.negative(x+y), Q.negative(x) & Q.negative(y)) == S(True)

    assert ask(Q.negative(x**2)) == S(None)
    assert ask(Q.negative(x**2), Q.real(x)) == S(False)
    assert ask(Q.negative(x**1.4), Q.real(x)) == S(None)

    assert ask(Q.negative(x*y)) == S(None)
    assert ask(Q.negative(x*y), Q.positive(x) & Q.positive(y)) == S(False)
    assert ask(Q.negative(x*y), Q.positive(x) & Q.negative(y)) == S(True)
    assert ask(Q.negative(x*y), Q.complex(x) & Q.complex(y)) == S(None)

    assert ask(Q.negative(x**y)) == S(None)
    assert ask(Q.negative(x**y), Q.negative(x) & Q.even(y)) == S(False)
    assert ask(Q.negative(x**y), Q.negative(x) & Q.odd(y)) == S(True)
    assert ask(Q.negative(x**y), Q.positive(x) & Q.integer(y)) == S(False)

    assert ask(Q.negative(Abs(x))) == S(False)

def test_nonzero():
    assert ask(Q.nonzero(x)) == S(None)
    assert ask(Q.nonzero(x), Q.real(x)) == S(None)
    assert ask(Q.nonzero(x), Q.positive(x)) == S(True)
    assert ask(Q.nonzero(x), Q.negative(x)) == S(True)
    assert ask(Q.nonzero(x), Q.negative(x) | Q.positive(x)) == S(True)

    assert ask(Q.nonzero(x+y)) == S(None)
    assert ask(Q.nonzero(x+y), Q.positive(x) & Q.positive(y)) == S(True)
    assert ask(Q.nonzero(x+y), Q.positive(x) & Q.negative(y)) == S(None)
    assert ask(Q.nonzero(x+y), Q.negative(x) & Q.negative(y)) == S(True)

    assert ask(Q.nonzero(2*x)) == S(None)
    assert ask(Q.nonzero(2*x), Q.positive(x)) == S(True)
    assert ask(Q.nonzero(2*x), Q.negative(x)) == S(True)
    assert ask(Q.nonzero(x*y), Q.nonzero(x)) == S(None)
    assert ask(Q.nonzero(x*y), Q.nonzero(x) & Q.nonzero(y)) == S(True)

    assert ask(Q.nonzero(Abs(x))) == S(None)
    assert ask(Q.nonzero(Abs(x)), Q.nonzero(x)) == S(True)

def test_odd():
    assert ask(Q.odd(x)) == S(None)
    assert ask(Q.odd(x), Q.odd(x)) == S(True)
    assert ask(Q.odd(x), Q.integer(x)) == S(None)
    assert ask(Q.odd(x), ~Q.integer(x)) == S(False)
    assert ask(Q.odd(x), Q.rational(x)) == S(None)
    assert ask(Q.odd(x), Q.positive(x)) == S(None)

    assert ask(Q.odd(-x), Q.odd(x)) == S(True)

    assert ask(Q.odd(2*x)) == S(None)
    assert ask(Q.odd(2*x), Q.integer(x)) == S(False)
    assert ask(Q.odd(2*x), Q.odd(x)) == S(False)
    assert ask(Q.odd(2*x), Q.irrational(x)) == S(False)
    assert ask(Q.odd(2*x), ~Q.integer(x)) == S(None)
    assert ask(Q.odd(3*x), Q.integer(x)) == S(None)

    assert ask(Q.odd(x/3), Q.odd(x)) == S(None)
    assert ask(Q.odd(x/3), Q.even(x)) == S(None)

    assert ask(Q.odd(x+1), Q.even(x)) == S(True)
    assert ask(Q.odd(x+2), Q.even(x)) == S(False)
    assert ask(Q.odd(x+2), Q.odd(x))  == S(True)
    assert ask(Q.odd(3-x), Q.odd(x))  == S(False)
    assert ask(Q.odd(3-x), Q.even(x))  == S(True)
    assert ask(Q.odd(3+x), Q.odd(x))  == S(False)
    assert ask(Q.odd(3+x), Q.even(x))  == S(True)
    assert ask(Q.odd(x+y), Q.odd(x) & Q.odd(y)) == S(False)
    assert ask(Q.odd(x+y), Q.odd(x) & Q.even(y)) == S(True)
    assert ask(Q.odd(x-y), Q.even(x) & Q.odd(y)) == S(True)
    assert ask(Q.odd(x-y), Q.odd(x) & Q.odd(y)) == S(False)

    assert ask(Q.odd(x+y+z), Q.odd(x) & Q.odd(y) & Q.even(z)) == S(False)
    assert ask(Q.odd(x+y+z+t), Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) == S(None)

    assert ask(Q.odd(2*x + 1), Q.integer(x)) == S(True)
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.odd(y)) == S(True)
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.even(y)) == S(False)
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.integer(y)) == S(None)
    assert ask(Q.odd(x*y), Q.odd(x) & Q.even(y)) == S(False)
    assert ask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) == S(True)
    assert ask(Q.odd(2*x*y), Q.rational(x) & Q.rational(x)) == S(None)
    assert ask(Q.odd(2*x*y), Q.irrational(x) & Q.irrational(x)) == S(None)

    assert ask(Q.odd(Abs(x)), Q.odd(x)) == S(True)

def test_prime():
    assert ask(Q.prime(x), Q.prime(x)) == S(True)
    assert ask(Q.prime(x), ~Q.prime(x)) == S(False)
    assert ask(Q.prime(x), Q.integer(x)) == S(None)
    assert ask(Q.prime(x), ~Q.integer(x)) == S(False)

    assert ask(Q.prime(2*x), Q.integer(x)) == S(False)
    assert ask(Q.prime(x*y)) == S(None)
    assert ask(Q.prime(x*y), Q.prime(x)) == S(None)
    assert ask(Q.prime(x*y), Q.integer(x) & Q.integer(y)) == S(False)

    assert ask(Q.prime(x**2), Q.integer(x)) == S(False)
    assert ask(Q.prime(x**2), Q.prime(x)) == S(False)
    assert ask(Q.prime(x**y), Q.integer(x) & Q.integer(y)) == S(False)

def test_positive():
    assert ask(Q.positive(x), Q.positive(x)) == S(True)
    assert ask(Q.positive(x), Q.negative(x)) == S(False)
    assert ask(Q.positive(x), Q.nonzero(x)) == S(None)

    assert ask(Q.positive(-x), Q.positive(x)) == S(False)
    assert ask(Q.positive(-x), Q.negative(x)) == S(True)

    assert ask(Q.positive(x+y), Q.positive(x) & Q.positive(y)) == S(True)
    assert ask(Q.positive(x+y), Q.positive(x) & Q.negative(y)) == S(None)

    assert ask(Q.positive(2*x), Q.positive(x)) == S(True)
    assumptions =  Q.positive(x) & Q.negative(y) & Q.negative(z) & Q.positive(w)
    assert ask(Q.positive(x*y*z))  == S(None)
    assert ask(Q.positive(x*y*z), assumptions) == S(True)
    assert ask(Q.positive(-x*y*z), assumptions) == S(False)

    assert ask(Q.positive(x**2), Q.positive(x)) == S(True)
    assert ask(Q.positive(x**2), Q.negative(x)) == S(True)

    #exponential
    assert ask(Q.positive(exp(x)), Q.real(x)) == S(True)
    assert ask(Q.positive(x + exp(x)), Q.real(x)) == S(None)

    #absolute value
    assert ask(Q.positive(Abs(x))) == S(None) # Abs(0) = 0
    assert ask(Q.positive(Abs(x)), Q.positive(x)) == S(True)

@XFAIL
def test_positive_xfail():
    assert ask(Q.positive(1/(1 + x**2)), Q.real(x)) == S(True)

def test_real():
    assert ask(Q.real(x)) == S(None)
    assert ask(Q.real(x), Q.real(x)) == S(True)
    assert ask(Q.real(x), Q.nonzero(x)) == S(True)
    assert ask(Q.real(x), Q.positive(x)) == S(True)
    assert ask(Q.real(x), Q.negative(x)) == S(True)
    assert ask(Q.real(x), Q.integer(x)) == S(True)
    assert ask(Q.real(x), Q.even(x)) == S(True)
    assert ask(Q.real(x), Q.prime(x)) == S(True)

    assert ask(Q.real(x/sqrt(2)), Q.real(x)) == S(True)
    assert ask(Q.real(x/sqrt(-2)), Q.real(x)) == S(False)

    assert ask(Q.real(x+1), Q.real(x)) == S(True)
    assert ask(Q.real(x+I), Q.real(x)) == S(False)
    assert ask(Q.real(x+I), Q.complex(x)) == S(None)

    assert ask(Q.real(2*x), Q.real(x)) == S(True)
    assert ask(Q.real(I*x), Q.real(x)) == S(False)
    assert ask(Q.real(I*x), Q.imaginary(x)) == S(True)
    assert ask(Q.real(I*x), Q.complex(x)) == S(None)

    assert ask(Q.real(x**2), Q.real(x)) == S(True)
    assert ask(Q.real(sqrt(x)), Q.negative(x)) == S(False)
    assert ask(Q.real(x**y), Q.real(x) & Q.integer(y)) == S(True)
    assert ask(Q.real(x**y), Q.real(x) & Q.real(y)) == S(None)
    assert ask(Q.real(x**y), Q.positive(x) & Q.real(y)) == S(True)

    # trigonometric functions
    assert ask(Q.real(sin(x))) == S(None)
    assert ask(Q.real(cos(x))) == S(None)
    assert ask(Q.real(sin(x)), Q.real(x)) == S(True)
    assert ask(Q.real(cos(x)), Q.real(x)) == S(True)

    # exponential function
    assert ask(Q.real(exp(x))) == S(None)
    assert ask(Q.real(exp(x)), Q.real(x)) == S(True)
    assert ask(Q.real(x + exp(x)), Q.real(x)) == S(True)

    # Q.complexes
    assert ask(Q.real(re(x))) == S(True)
    assert ask(Q.real(im(x))) == S(True)

def test_algebraic():
    assert ask(Q.algebraic(x)) == S(None)

    assert ask(Q.algebraic(I)) == S(True)
    assert ask(Q.algebraic(2*I)) == S(True)
    assert ask(Q.algebraic(I/3)) == S(True)

    assert ask(Q.algebraic(sqrt(7))) == S(True)
    assert ask(Q.algebraic(2*sqrt(7))) == S(True)
    assert ask(Q.algebraic(sqrt(7)/3)) == S(True)

    assert ask(Q.algebraic(I*sqrt(3))) == S(True)
    assert ask(Q.algebraic(sqrt(1+I*sqrt(3)))) == S(True)

    assert ask(Q.algebraic((1+I*sqrt(3)**(S(17)/31)))) == S(True)
    assert ask(Q.algebraic((1+I*sqrt(3)**(S(17)/pi)))) == S(False)

    assert ask(Q.algebraic(sin(7))) == S(None)
    assert ask(Q.algebraic(sqrt(sin(7)))) == S(None)
    assert ask(Q.algebraic(sqrt(y+I*sqrt(7)))) == S(None)

    assert ask(Q.algebraic(oo)) == S(False)
    assert ask(Q.algebraic(-oo)) == S(False)

    assert ask(Q.algebraic(2.47)) == S(False)

def test_global():
    """Test ask with global assumptions"""
    assert ask(Q.integer(x)) == S(None)
    global_assumptions.add(Q.integer(x))
    assert ask(Q.integer(x)) == S(True)
    global_assumptions.clear()
    assert ask(Q.integer(x)) == S(None)

def test_custom_context():
    """Test ask with custom assumptions context"""
    assert ask(Q.integer(x)) == S(None)
    local_context = AssumptionsContext()
    local_context.add(Q.integer(x))
    assert ask(Q.integer(x), context = local_context) == S(True)
    assert ask(Q.integer(x)) == S(None)

def test_functions_in_assumptions():
    assert ask(Q.negative(x), Q.real(x) >> Q.positive(x)) is S(False)
    assert ask(Q.negative(x), Equivalent(Q.real(x), Q.positive(x))) is S(False)
    assert ask(Q.negative(x), Xor(Q.real(x), Q.negative(x))) is S(False)

def test_composite_ask():
    assert ask(Q.negative(x) & Q.integer(x),
           assumptions=Q.real(x) >> Q.positive(x)) is S(False)

def test_composite_proposition():
    assert ask(True) is S(True)
    assert ask(~Q.negative(x), Q.positive(x)) is S(True)
    assert ask(~Q.real(x), Q.commutative(x)) is S(None)
    assert ask(Q.negative(x) & Q.integer(x), Q.positive(x)) is S(False)
    assert ask(Q.negative(x) & Q.integer(x)) is S(None)
    assert ask(Q.real(x) | Q.integer(x), Q.positive(x)) is S(True)
    assert ask(Q.real(x) | Q.integer(x)) is S(None)
    assert ask(Q.real(x) >> Q.positive(x), Q.negative(x)) is S(False)
    assert ask(Implies(Q.real(x), Q.positive(x), evaluate=False), Q.negative(x)) is S(False)
    assert ask(Implies(Q.real(x), Q.positive(x), evaluate=False)) is S(None)
    assert ask(Equivalent(Q.integer(x), Q.even(x)), Q.even(x)) is S(True)
    assert ask(Equivalent(Q.integer(x), Q.even(x))) is S(None)
    assert ask(Equivalent(Q.positive(x), Q.integer(x)), Q.integer(x)) is S(None)

def test_incompatible_resolutors():
    class Prime2AskHandler(AskHandler):
        @staticmethod
        def Number(expr, assumptions):
            return S(True)
    register_handler('prime', Prime2AskHandler)
    raises(ValueError, 'ask(Q.prime(4))')
    remove_handler('prime', Prime2AskHandler)

    class InconclusiveHandler(AskHandler):
        @staticmethod
        def Number(expr, assumptions):
            return S(None)
    register_handler('prime', InconclusiveHandler)
    assert ask(Q.prime(3)) == S(True)

def test_key_extensibility():
    """test that you can add keys to the ask system at runtime"""
    # make sure the key is not defined
    raises(AttributeError, "ask(Q.my_key(x))")
    class MyAskHandler(AskHandler):
        @staticmethod
        def Symbol(expr, assumptions):
            return S(True)
    register_handler('my_key', MyAskHandler)
    assert ask(Q.my_key(x)) == S(True)
    assert ask(Q.my_key(x+1)) == S(None)
    remove_handler('my_key', MyAskHandler)
    del Q.my_key
    raises(AttributeError, "ask(Q.my_key(x))")

def test_type_extensibility():
    """test that new types can be added to the ask system at runtime
    We create a custom type MyType, and override ask Q.prime=True with handler
    MyAskHandler for this type

    TODO: test incompatible resolutors
    """
    from sympy.core import Basic

    class MyType(Basic):
        pass

    class MyAskHandler(AskHandler):
        @staticmethod
        def MyType(expr, assumptions):
            return S(True)

    a = MyType()
    register_handler(Q.prime, MyAskHandler)
    assert ask(Q.prime(a)) == S(True)

def test_compute_known_facts():
    ns = {}
    exec 'from sympy.logic.boolalg import And, Or, Not' in globals(), ns
    exec compute_known_facts() in globals(), ns
    assert ns['known_facts_cnf'] == known_facts_cnf
    assert ns['known_facts_dict'] == known_facts_dict
