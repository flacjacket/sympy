"""
Handlers for predicates related to set membership: integer, rational, etc.
"""
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import CommonHandler
from sympy.core.singleton import S

class AskIntegerHandler(CommonHandler):
    """
    Handler for Q.integer
    Test that an expression belongs to the field of integer numbers
    """

    @staticmethod
    def _number(expr, assumptions):
        # helper method
        if expr.as_real_imag()[1] == 0:
            return S(expr.evalf(1) == expr)
        return S(False)

    @staticmethod
    def Add(expr, assumptions):
        """
        Integer + Integer       -> Integer
        Integer + !Integer      -> !Integer
        !Integer + !Integer -> ?
        """
        if expr.is_number:
            return AskIntegerHandler._number(expr, assumptions)
        return test_closed_group(expr, assumptions, Q.integer)

    @staticmethod
    def Mul(expr, assumptions):
        """
        Integer*Integer      -> Integer
        Integer*Irrational   -> !Integer
        Odd/Even             -> !Integer
        Integer*Rational     -> ?
        """
        if expr.is_number:
            return AskIntegerHandler._number(expr, assumptions)
        _output = S(True)
        for arg in expr.args:
            if not ask(Q.integer(arg), assumptions):
                if arg.is_Rational:
                    if arg.q == 2:
                        return ask(Q.even(2*expr), assumptions)
                    if ~(arg.q & 1):
                        return S(None)
                elif ask(Q.irrational(arg), assumptions):
                    if _output:
                        _output = S(False)
                    else:
                        return S(None)
                else:
                    return S(None)
        else:
            return _output
        return S(None)

    Pow = Add

    @staticmethod
    def int(expr, assumptions):
        return S(True)

    @staticmethod
    def Integer(expr, assumptions):
        return S(True)

    @staticmethod
    def Rational(expr, assumptions):
        # rationals with denominator one get
        # evaluated to Integers
        return S(False)

    @staticmethod
    def Float(expr, assumptions):
        return S(int(expr) == expr)

    @staticmethod
    def Pi(expr, assumptions):
        return S(False)

    @staticmethod
    def Exp1(expr, assumptions):
        return S(False)

    @staticmethod
    def Infinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(False)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Abs(expr, assumptions):
        return ask(Q.integer(expr.args[0]), assumptions)

class AskRationalHandler(CommonHandler):
    """
    Handler for Q.rational
    Test that an expression belongs to the field of rational numbers
    """

    @staticmethod
    def Add(expr, assumptions):
        """
        Rational + Rational     -> Rational
        Rational + !Rational    -> !Rational
        !Rational + !Rational   -> ?
        """
        if expr.is_number:
            if expr.as_real_imag()[1]:
                return S(False)
        return test_closed_group(expr, assumptions, Q.rational)

    Mul = Add

    @staticmethod
    def Pow(expr, assumptions):
        """
        Rational ** Integer      -> Rational
        Irrational ** Rational   -> Irrational
        Rational ** Irrational   -> ?
        """
        if ask(Q.integer(expr.exp), assumptions):
            return ask(Q.rational(expr.base), assumptions)
        elif ask(Q.rational(expr.exp), assumptions):
            if ask(Q.prime(expr.base), assumptions):
                return S(False)
        return S(None)

    @staticmethod
    def Rational(expr, assumptions):
        return S(True)

    @staticmethod
    def Float(expr, assumptions):
        # it's finite-precission
        return S(True)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Infinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(False)

    @staticmethod
    def Pi(expr, assumptions):
        return S(False)

    @staticmethod
    def Exp1(expr, assumptions):
        return S(False)

class AskIrrationalHandler(CommonHandler):

    @staticmethod
    def Basic(expr, assumptions):
        _real = ask(Q.real(expr), assumptions)
        if _real:
            _rational = ask(Q.rational(expr), assumptions)
            if _rational is S(None): return S(None)
            return S(not _rational)
        else: return _real

class AskRealHandler(CommonHandler):
    """
    Handler for Q.real
    Test that an expression belongs to the field of real numbers
    """

    @staticmethod
    def _number(expr, assumptions):
        return S(not expr.as_real_imag()[1])

    @staticmethod
    def Add(expr, assumptions):
        """
        Real + Real              -> Real
        Real + (Complex & !Real) -> !Real
        """
        if expr.is_number:
            return AskRealHandler._number(expr, assumptions)
        return test_closed_group(expr, assumptions, Q.real)

    @staticmethod
    def Mul(expr, assumptions):
        """
        Real*Real               -> Real
        Real*Imaginary          -> !Real
        Imaginary*Imaginary     -> Real
        """
        if expr.is_number:
            return AskRealHandler._number(expr, assumptions)
        result = S(True)
        for arg in expr.args:
            if ask(Q.real(arg), assumptions):
                pass
            elif ask(Q.imaginary(arg), assumptions):
                result = result ^ True
            else:
                break
        else:
            return result
        return S(None)

    @staticmethod
    def Pow(expr, assumptions):
        """
        Real**Integer         -> Real
        Positive**Real        -> Real
        Real**(Integer/Even)  -> Real if base is nonnegative
        Real**(Integer/Odd)   -> Real
        """
        if expr.is_number:
            return AskRealHandler._number(expr, assumptions)
        if ask(Q.real(expr.base), assumptions):
            if ask(Q.integer(expr.exp), assumptions):
                return S(True)
            elif expr.exp.is_Rational:
                if (expr.exp.q % 2 == 0):
                    return S(ask(Q.real(expr.base), assumptions) and \
                       not ask(Q.negative(expr.base), assumptions))
                else: return S(True)
            elif ask(Q.real(expr.exp), assumptions):
                if ask(Q.positive(expr.base), assumptions):
                    return S(True)
        return S(None)

    @staticmethod
    def Rational(expr, assumptions):
        return S(True)

    @staticmethod
    def Float(expr, assumptions):
        return S(True)

    @staticmethod
    def Pi(expr, assumptions):
        return S(True)

    @staticmethod
    def Exp1(expr, assumptions):
        return S(True)

    @staticmethod
    def Abs(expr, assumptions):
        return S(True)

    @staticmethod
    def re(expr, assumptions):
        return S(True)

    im = re

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Infinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(False)

    @staticmethod
    def sin(expr, assumptions):
        if ask(Q.real(expr.args[0]), assumptions):
            return S(True)
        return S(None)

    cos, exp = sin, sin

class AskExtendedRealHandler(AskRealHandler):
    """
    Handler for Q.extended_real
    Test that an expression belongs to the field of extended real numbers,
    that is real numbers union {Infinity, -Infinity}
    """

    @staticmethod
    def Add(expr, assumptions):
        return test_closed_group(expr, assumptions, Q.extended_real)

    Mul, Pow = Add, Add

    @staticmethod
    def Infinity(expr, assumptions):
        return S(True)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(True)

class AskComplexHandler(CommonHandler):
    """
    Handler for Q.complex
    Test that an expression belongs to the field of complex numbers
    """

    @staticmethod
    def Add(expr, assumptions):
        return test_closed_group(expr, assumptions, Q.complex)

    Mul, Pow = Add, Add

    @staticmethod
    def Number(expr, assumptions):
        return S(True)

    @staticmethod
    def NumberSymbol(expr, assumptions):
        return S(True)

    @staticmethod
    def Abs(expr, assumptions):
        return S(True)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(True)

    @staticmethod
    def Infinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(False)

    sin, cos, exp, re, im = [Abs]*5 # they are all complex functions

class AskImaginaryHandler(CommonHandler):
    """
    Handler for Q.imaginary
    Test that an expression belongs to the field of imaginary numbers,
    that is, numbers in the form x*I, where x is real
    """

    @staticmethod
    def _number(expr, assumptions):
        # helper method
        return S(not expr.as_real_imag()[0])

    @staticmethod
    def Add(expr, assumptions):
        """
        Imaginary + Imaginary -> Imaginary
        Imaginary + Complex   -> ?
        Imaginary + Real      -> !Imaginary
        """
        if expr.is_number:
            return AskImaginaryHandler._number(expr, assumptions)
        reals = 0
        for arg in expr.args:
            if ask(Q.imaginary(arg), assumptions):
                pass
            elif ask(Q.real(arg), assumptions):
                reals += 1
            else:
                break
        else:
            if reals == 0:
                return S(True)
            if reals == 1 or (len(expr.args) == reals):
                # two reals could sum 0 thus giving an imaginary
                return S(False)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        """
        Real*Imaginary      -> Imaginary
        Imaginary*Imaginary -> Real
        """
        if expr.is_number:
            return AskImaginaryHandler._number(expr, assumptions)
        result = S(False)
        reals = 0
        for arg in expr.args:
            if ask(Q.imaginary(arg), assumptions):
                result = result ^ True
            elif not ask(Q.real(arg), assumptions):
                break
        else:
            if reals == len(expr.args):
                return S(False)
            return result
        return S(None)

    Pow = Add

    @staticmethod
    def Number(expr, assumptions):
        return S(not (expr.as_real_imag()[1] == 0))

    NumberSymbol = Number

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(True)

class AskAlgebraicHandler(CommonHandler):
    """Handler for Q.algebraic key. """

    @staticmethod
    def Add(expr, assumptions):
        return test_closed_group(expr, assumptions, Q.algebraic)

    @staticmethod
    def Mul(expr, assumptions):
        return test_closed_group(expr, assumptions, Q.algebraic)

    @staticmethod
    def Pow(expr, assumptions):
        return S(expr.exp.is_Rational and ask(Q.algebraic(expr.base), assumptions))

    @staticmethod
    def Number(expr, assumptions):
        return S(False)

    @staticmethod
    def Rational(expr, assumptions):
        return S(expr.q != 0)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(True)

    @staticmethod
    def AlgebraicNumber(expr, assumptions):
        return S(True)

#### Helper methods

def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation
    """
    result = S(True)
    for arg in expr.args:
        _out = ask(key(arg), assumptions)
        if _out is S(None): return S(None)
        elif _out is S(False):
            if result: result = S(False)
            else: break
    else:
        return result
    return S(None)

