"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import CommonHandler
from sympy.core.singleton import S
from sympy.ntheory import isprime

class AskPrimeHandler(CommonHandler):
    """
    Handler for key 'prime'
    Test that an expression represents a prime number
    """

    @staticmethod
    def _number(expr, assumptions):
        # helper method
        if (expr.as_real_imag()[1] == 0) and int(expr.evalf()) == expr:
            return S(isprime(expr.evalf(1)))
        return S(False)

    @staticmethod
    def Basic(expr, assumptions):
        # Just use int(expr) once
        # http://code.google.com/p/sympy/issues/detail?id=1462
        # is solved
        if expr.is_number:
            return AskPrimeHandler._number(expr, assumptions)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        if expr.is_number:
            return AskPrimeHandler._number(expr, assumptions)
        for arg in expr.args:
            if ask(Q.integer(arg), assumptions):
                pass
            else: break
        else:
            # a product of integers can't be a prime
            return S(False)
        return S(None)

    @staticmethod
    def Pow(expr, assumptions):
        """
        Integer**Integer     -> !Prime
        """
        if expr.is_number:
            return AskPrimeHandler._number(expr, assumptions)
        if ask(Q.integer(expr.exp), assumptions) and \
                ask(Q.integer(expr.base), assumptions):
            return S(False)
        return S(None)

    @staticmethod
    def Integer(expr, assumptions):
        return S(isprime(expr))

    @staticmethod
    def Rational(expr, assumptions):
        return S(False)

    @staticmethod
    def Float(expr, assumptions):
        return AskPrimeHandler._number(expr, assumptions)

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
    def NumberSymbol(expr, assumptions):
        return AskPrimeHandler._number(expr, assumptions)

class AskCompositeHandler(CommonHandler):

    @staticmethod
    def Basic(expr, assumptions):
        _positive = ask(Q.positive(expr), assumptions)
        if _positive:
            _integer = ask(Q.integer(expr), assumptions)
            if _integer:
                _prime = ask(Q.prime(expr), assumptions)
                if _prime is S(None): return _prime
                return S(not _prime)
            else: return _integer
        else: return _positive

class AskEvenHandler(CommonHandler):

    @staticmethod
    def _number(expr, assumptions):
        # helper method
        if (expr.as_real_imag()[1] == 0) and expr.evalf(1) == expr:
            return S(float(expr.evalf()) % 2 == 0)
        else: return S(False)

    @staticmethod
    def Basic(expr, assumptions):
        if expr.is_number:
            return AskEvenHandler._number(expr, assumptions)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        """
        Even * Integer -> Even
        Even * Odd     -> Even
        Integer * Odd  -> ?
        Odd * Odd      -> Odd
        """
        if expr.is_number:
            return AskEvenHandler._number(expr, assumptions)
        even, odd, irrational = False, 0, False
        for arg in expr.args:
            # check for all integers and at least one even
            if ask(Q.integer(arg), assumptions):
                if ask(Q.even(arg), assumptions):
                    even = True
                elif ask(Q.odd(arg), assumptions):
                    odd += 1
            elif ask(Q.irrational(arg), assumptions):
                # one irrational makes the result False
                # two makes it undefined
                if irrational:
                    break
                irrational = True
            else: break
        else:
            if irrational: return S(False)
            if even: return S(True)
            if odd == len(expr.args): return S(False)
        return S(None)

    @staticmethod
    def Add(expr, assumptions):
        """
        Even + Odd  -> Odd
        Even + Even -> Even
        Odd  + Odd  -> Even

        TODO: remove float() when issue
        http://code.google.com/p/sympy/issues/detail?id=1473
        is solved
        """
        if expr.is_number:
            return AskEvenHandler._number(expr, assumptions)
        _result = S(True)
        for arg in expr.args:
            if ask(Q.even(arg), assumptions):
                pass
            elif ask(Q.odd(arg), assumptions):
                _result = S(not _result)
            else: break
        else:
            return _result
        return S(None)

    @staticmethod
    def Integer(expr, assumptions):
        return S(not bool(expr.p & 1))

    @staticmethod
    def Rational(expr, assumptions):
        return S(False)

    @staticmethod
    def Float(expr, assumptions):
        return S(expr % 2 == 0)

    @staticmethod
    def Infinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NegativeInfinity(expr, assumptions):
        return S(False)

    @staticmethod
    def NumberSymbol(expr, assumptions):
        return AskEvenHandler._number(expr, assumptions)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Abs(expr, assumptions):
        if ask(Q.real(expr.args[0]), assumptions):
            return ask(Q.even(expr.args[0]), assumptions)
        return S(None)

    @staticmethod
    def re(expr, assumptions):
        if ask(Q.real(expr.args[0]), assumptions):
            return ask(Q.even(expr.args[0]), assumptions)
        return S(None)

    @staticmethod
    def im(expr, assumptions):
        if ask(Q.real(expr.args[0]), assumptions):
            return S(True)
        return S(None)

class AskOddHandler(CommonHandler):
    """
    Handler for key 'odd'
    Test that an expression represents an odd number
    """

    @staticmethod
    def Basic(expr, assumptions):
        _integer = ask(Q.integer(expr), assumptions)
        if _integer:
            _even = ask(Q.even(expr), assumptions)
            if _even is S(None): return S(None)
            return S(not _even)
        return _integer
