from sympy.core.singleton import S
from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask

class AskHandler(object):
    """Base class that all Ask Handlers must inherit"""
    pass

class CommonHandler(AskHandler):
    """Defines some useful methods common to most Handlers """

    @staticmethod
    def NaN(expr, assumptions):
        return S(False)

class AskCommutativeHandler(CommonHandler):
    """
    Handler for key 'commutative'
    """

    @staticmethod
    def Symbol(expr, assumptions):
        """Objects are expected to be commutative unless otherwise stated"""
        assumps = conjuncts(assumptions)
        if Q.commutative(expr) in assumps:
            return S(True)
        elif ~Q.commutative(expr) in assumps:
            return S(False)
        return S(True)

    @staticmethod
    def Basic(expr, assumptions):
        for arg in expr.args:
            if not ask(Q.commutative(arg), assumptions):
                return S(False)
        return S(True)

    @staticmethod
    def Number(expr, assumptions):
        return S(True)

    @staticmethod
    def NaN(expr, assumptions):
        return S(True)

class TautologicalHandler(AskHandler):
    """Wrapper allowing to query the truth value of a boolean expression."""

    @staticmethod
    def bool(expr, assumptions):
        return S(expr)

    @staticmethod
    def BooleanValue(expr, assumptions):
        return expr

    @staticmethod
    def AppliedPredicate(expr, assumptions):
        return ask(expr, assumptions)

    @staticmethod
    def Not(expr, assumptions):
        value = ask(expr.args[0], assumptions=assumptions)
        if value in (S(True), S(False)):
            return S(not value)
        else:
            return S(None)


    @staticmethod
    def Or(expr, assumptions):
        result = S(False)
        for arg in expr.args:
            p = ask(arg, assumptions=assumptions)
            if p == S(True):
                return S(True)
            if p == S(None):
                result = S(None)
        return result

    @staticmethod
    def And(expr, assumptions):
        result = S(True)
        for arg in expr.args:
            p = ask(arg, assumptions=assumptions)
            if p == S(False):
                return S(False)
            if p == S(None):
                result = S(None)
        return result

    @staticmethod
    def Implies(expr, assumptions):
        p, q = expr.args
        return ask(~p | q, assumptions=assumptions)

    @staticmethod
    def Equivalent(expr, assumptions):
        p, q = expr.args
        pt = ask(p, assumptions=assumptions)
        if pt == S(None):
            return S(None)
        qt = ask(q, assumptions=assumptions)
        if qt == S(None):
            return S(None)
        if pt == qt:
            return S(True)
        return S(False)
