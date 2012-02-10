"""
AskHandlers related to order relations: positive, negative, etc.
"""
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import CommonHandler
from sympy.core.singleton import S


class AskNegativeHandler(CommonHandler):
    """
    This is called by ask() when key='negative'

    Test that an expression is less (strict) than zero.

    Examples:

    >>> from sympy import ask, Q, pi
    >>> ask(Q.negative(pi+1)) # this calls AskNegativeHandler.Add
    False
    >>> ask(Q.negative(pi**2)) # this calls AskNegativeHandler.Pow
    False

    """

    @staticmethod
    def _number(expr, assumptions):
        if not expr.as_real_imag()[1]:
            return S(expr.evalf() < 0)
        else: return S(False)

    @staticmethod
    def Basic(expr, assumptions):
        if expr.is_number:
            return AskNegativeHandler._number(expr, assumptions)
        return S(None)

    @staticmethod
    def Add(expr, assumptions):
        """
        Positive + Positive -> Positive,
        Negative + Negative -> Negative
        """
        if expr.is_number:
            return AskNegativeHandler._number(expr, assumptions)
        for arg in expr.args:
            if not ask(Q.negative(arg), assumptions):
                break
        else:
            # if all argument's are negative
            return S(True)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        if expr.is_number:
            return AskNegativeHandler._number(expr, assumptions)
        result = S(None)
        for arg in expr.args:
            if result is S(None): result = S(False)
            if ask(Q.negative(arg), assumptions):
                result = S(not result)
            elif ask(Q.positive(arg), assumptions):
                pass
            else: return S(None)
        return result

    @staticmethod
    def Pow(expr, assumptions):
        """
        Real ** Even -> NonNegative
        Real ** Odd  -> same_as_base
        NonNegative ** Positive -> NonNegative
        """
        if expr.is_number:
            return AskNegativeHandler._number(expr, assumptions)
        if ask(Q.real(expr.base), assumptions):
            if ask(Q.positive(expr.base), assumptions):
                return S(False)
            if ask(Q.even(expr.exp), assumptions):
                return S(False)
            if ask(Q.odd(expr.exp), assumptions):
                return ask(Q.negative(expr.base), assumptions)
        return S(None)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Abs(expr, assumptions):
        return S(False)

class AskNonZeroHandler(CommonHandler):
    """
    Handler for key 'zero'
    Test that an expression is not identically zero
    """

    @staticmethod
    def Basic(expr, assumptions):
        if expr.is_number:
            # if there are no symbols just evalf
            return S(expr.evalf() != 0)
        return S(None)

    @staticmethod
    def Add(expr, assumptions):
        if all(ask(Q.positive(x), assumptions) for x in expr.args) \
            or all(ask(Q.negative(x), assumptions) for x in expr.args):
            return S(True)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        for arg in expr.args:
            result = ask(Q.nonzero(arg), assumptions)
            if result: continue
            return result
        return S(True)

    @staticmethod
    def Pow(expr, assumptions):
        return ask(Q.nonzero(expr.base), assumptions)

    @staticmethod
    def NaN(expr, assumptions):
        return S(True)

    @staticmethod
    def Abs(expr, assumptions):
        return ask(Q.nonzero(expr.args[0]), assumptions)

class AskPositiveHandler(CommonHandler):
    """
    Handler for key 'positive'
    Test that an expression is greater (strict) than zero
    """

    @staticmethod
    def _number(expr, assumptions):
        if not expr.as_real_imag()[1]:
            return S(expr.evalf() > 0)
        else: return S(False)

    @staticmethod
    def Basic(expr, assumptions):
        if expr.is_number:
            return AskPositiveHandler._number(expr, assumptions)
        return S(None)

    @staticmethod
    def Mul(expr, assumptions):
        if expr.is_number:
            return AskPositiveHandler._number(expr, assumptions)
        result = S(True)
        for arg in expr.args:
            if ask(Q.positive(arg), assumptions): continue
            elif ask(Q.negative(arg), assumptions):
                result = S(result ^ True)
            else: return S(None)
        return result

    @staticmethod
    def Add(expr, assumptions):
        if expr.is_number:
            return AskPositiveHandler._number(expr, assumptions)
        for arg in expr.args:
            if ask(Q.positive(arg), assumptions) is not S(True):
                break
        else:
            # if all argument's are positive
            return S(True)
        return S(None)

    @staticmethod
    def Pow(expr, assumptions):
        if expr.is_number: return S(expr.evalf() > 0)
        if ask(Q.positive(expr.base), assumptions):
            return S(True)
        if ask(Q.negative(expr.base), assumptions):
            if ask(Q.even(expr.exp), assumptions):
                return S(True)
            if ask(Q.even(expr.exp), assumptions):
                return S(False)
        return S(None)

    @staticmethod
    def exp(expr, assumptions):
        if ask(Q.real(expr.args[0]), assumptions):
            return S(True)
        return S(None)

    @staticmethod
    def ImaginaryUnit(expr, assumptions):
        return S(False)

    @staticmethod
    def Abs(expr, assumptions):
        return ask(Q.nonzero(expr), assumptions)
