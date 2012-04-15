#TODO:
# -Implement Clebsch-Gordan symmetries
# -Improve simplification method
# -Implement new simpifications
"""Clebsch-Gordon Coefficients."""

from sympy import (Add, expand, Eq, Expr, Mul, Piecewise, Pow, S, sqrt, sign,
                   Sum, symbols, sympify, Wild)
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j

__all__ = [
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    'CG',
    'cg_simp'
]

#-----------------------------------------------------------------------------
# CG Coefficients
#-----------------------------------------------------------------------------

class Wigner3j(Expr):
    """Class for the Wigner-3j symbols

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calcualte its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6,4,2,0,0,0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    def __new__(cls, j1, m1, j2, m2, j3, m3):
        j1,m1,j2,m2,j3,m3 = map(sympify, (j1,m1,j2,m2,j3,m3))
        return Expr.__new__(cls, j1, m1, j2, m2, j3, m3)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all([arg.is_number for arg in self.args])

    def _sympystr(self, printer, *args):
        return '%s(%s,%s,%s,%s,%s,%s)' % (
            self.__class__.__name__,
            printer._print(self.j1), printer._print(self.j2), printer._print(self.j3),
            printer._print(self.m1), printer._print(self.m2), printer._print(self.m3)
        )

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.m1)), \
            (printer._print(self.j2), printer._print(self.m2)), \
            (printer._print(self.j3), printer._print(self.m3)))
        hsep = 2
        vsep = 1
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([ m[j][i].width() for i in range(2) ])
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft  = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens())
        return D

    def _latex(self, printer, *args):
        return r'\left(\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right)' % \
            (printer._print(self.j1), printer._print(self.j2), printer._print(self.j3), \
            printer._print(self.m1), printer._print(self.m2), printer._print(self.m3))

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)


class CG(Wigner3j):
    """Class for Clebsch-Gordan coefficient

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_1,m_1}_{j_2,m_2,j_3,m_3} = \langle j_1,m_1;j_2,m_2 | j_3,m_3\\rangle

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return clebsch_gordan(self.j1,self.j2, self.j3, self.m1, self.m2, self.m3)

    def _sympystr(self, printer, *args):
        return 'CG(%s, %s, %s, %s, %s, %s)' % (
            printer._print(self.j1), printer._print(self.m1), printer._print(self.j2), \
            printer._print(self.m2), printer._print(self.j3), printer._print(self.m3)
        )

    def _pretty(self, printer, *args):
        bot = printer._print(self.j1)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.m1)))
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.j2)))
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.m2)))
        top = printer._print(self.j3)
        top = prettyForm(*top.right(','))
        top = prettyForm(*top.right(printer._print(self.m3)))

        pad = max(top.width(), bot.width())

        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))
        if not pad == bot.width():
            bot = prettyForm(*bot.right(' ' * (pad-bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' ' * (pad-top.width())))
        s = stringPict('C' + ' '*pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        return r'C^{%s,%s}_{%s,%s,%s,%s}' % \
            (printer._print(self.j3), printer._print(self.m3),
            printer._print(self.j1), printer._print(self.m1),
            printer._print(self.j2), printer._print(self.m2))


class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j, j23):
        j1,j2,j12,j3,j,j23 = map(sympify, (j1,j2,j12,j3,j,j23))
        return Expr.__new__(cls, j1, j2, j12, j3, j, j23)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j(self):
        return self.args[4]

    @property
    def j23(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all([arg.is_number for arg in self.args])

    # This is modified from the _print_Matrix method
    def _sympystr(self, printer, *args):
        res = [[printer._print(self.j1), printer._print(self.j2), printer._print(self.j12)], \
            [printer._print(self.j3), printer._print(self.j), printer._print(self.j23)]]
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([ len(res[i][j]) for i in range(2) ])
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = elem.rjust(maxw[j])
            res[i] = "{" + ", ".join(row) + "}"
        return '\n'.join(res)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3)), \
            (printer._print(self.j2), printer._print(self.j)), \
            (printer._print(self.j12), printer._print(self.j23)))
        hsep = 2
        vsep = 1
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([ m[j][i].width() for i in range(2) ])
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft  = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        return r'\left{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right}' % \
            (printer._print(self.j1), printer._print(self.j2), printer._print(self.j12), \
            printer._print(self.j3), printer._print(self.j), printer._print(self.j23))

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j3)


class Wigner9j(Expr):
    """Class for the Wigner-9j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j):
        j1,j2,j12,j3,j4,j34,j13,j24,j = map(sympify, (j1,j2, j12, j3, j4, j34, j13, j24, j))
        return Expr.__new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j4(self):
        return self.args[4]

    @property
    def j34(self):
        return self.args[5]

    @property
    def j13(self):
        return self.args[6]

    @property
    def j24(self):
        return self.args[7]

    @property
    def j(self):
        return self.args[8]

    @property
    def is_symbolic(self):
        return not all([arg.is_number for arg in self.args])

    # This is modified from the _print_Matrix method
    def _sympystr(self, printer, *args):
        res = [[printer._print(self.j1), printer._print(self.j2), printer._print(self.j12)], \
            [printer._print(self.j3), printer._print(self.j4), printer._print(self.j34)], \
            [printer._print(self.j13), printer._print(self.j24), printer._print(self.j)]]
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([ len(res[i][j]) for i in range(2) ])
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = elem.rjust(maxw[j])
            res[i] = "{" + ", ".join(row) + "}"
        return '\n'.join(res)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3), printer._print(self.j13)), \
            (printer._print(self.j2), printer._print(self.j4), printer._print(self.j24)), \
            (printer._print(self.j12), printer._print(self.j34), printer._print(self.j)))
        hsep = 2
        vsep = 1
        maxw = [-1] * 3
        for j in range(3):
            maxw[j] = max([ m[j][i].width() for i in range(3) ])
        D = None
        for i in range(3):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft  = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        return r'\left{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right}' % \
            (printer._print(self.j1), printer._print(self.j2), printer._print(self.j12), \
            printer._print(self.j3), printer._print(self.j4), printer._print(self.j34), \
            printer._print(self.j13), printer._print(self.j24), printer._print(self.j))

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)


def cg_simp(e):
    """Simplify and combine CG coefficients

    This function uses various symmetry and properties of sums and
    products of Clebsch-Gordan coefficients to simplify statements
    involving these terms [1]_.

    Examples
    ========

    Simplify the sum over CG(a,alpha,0,0,a,alpha) for all alpha to
    2*a+1

        >>> from sympy.physics.quantum.cg import CG, cg_simp
        >>> a = CG(1,1,0,0,1,1)
        >>> b = CG(1,0,0,0,1,0)
        >>> c = CG(1,-1,0,0,1,-1)
        >>> cg_simp(a+b+c)
        3

    See Also
    ========

    CG: Clebsh-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    for sums in e.atoms(Sum):
        new_term = _cg_simp_sum(sums)
        e = e.subs(sums, new_term)

    if isinstance(e, Add):
        return _cg_simp_add(e)
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    else:
        return e


def _cg_simp_add(e):
    #TODO: Improve simplification method
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    First, we create two lists, cg_part, which is all the terms involving CG
    coefficients, and other_part, which is all other terms. The cg_part list
    is then passed to the simplification methods, which return the new cg_part
    and any additional terms that are added to other_part
    """
    e = expand(e)
    cg_part = [arg for arg in e.args if arg.has(CG)]
    other_part = [arg for arg in e.args if not arg.has(CG)]

    # Varshalovich Eq 8.7.1 Eq 1
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    param = a, alpha, b = symbols('a alpha b', cls=Wild)
    expr = CG(a,alpha,b,0,a,alpha)
    simp = (2*a+1)*KroneckerDelta(b,0)
    const_param = a, b
    index_expr = a+alpha
    index_max = 2*a
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max)
    other_part.extend(other)

    # Varshalovich Eq 8.7.1 Eq 2
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    param = a, alpha, c = symbols('a alpha c', cls=Wild)
    expr = CG(a,alpha,a,-alpha,c,0)
    simp = sqrt(2*a+1)*KroneckerDelta(c,0)
    sign = (-1)**(a-alpha)
    const_param = a, c
    index_expr = a+alpha
    index_max = 2*a
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max, sign)
    other_part.extend(other)

    # Varshalovich Eq 8.7.2 Eq 9
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    # Case alpha = alpha', beta = beta'
    # For numerical alpha,beta
    param = a,alpha,alphap,b,beta,betap,c,gamma = symbols('a alpha alphap b beta betap c gamma', cls=Wild)
    expr = CG(a,alpha,b,beta,c,gamma)**2
    simp = S.One
    const_param = a, alpha, b, beta
    x = abs(a-b)
    y = abs(alpha+beta)
    index_expr = a + b - c
    index_max = a + b - abs(x - y)
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max)
    other_part.extend(other)
    # For symbolic alpha,beta
    x = abs(a-b)
    y = a+b
    index_expr = (c-x)*(x+c)+c+gamma
    index_max = (y+1-x)*(x+y+1) - 1
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max)
    other_part.extend(other)
    # Case alpha!=alphap or beta!=betap
    # For numerical alpha,alphap,beta,betap
    # Note: this is currently broken, as the .match function cannot handle the leading term in addition to the expression
    expr = CG(a,alpha,b,beta,c,gamma) * CG(a,alphap,b,betap,c,gamma)
    simp = KroneckerDelta(alpha,alphap) * KroneckerDelta(beta,betap)
    const_param = a, alpha, alphap, b, beta, betap
    x = abs(a-b)
    y = abs(alpha+beta)
    index_expr = a + b - c
    index_max = a + b - abs(x - y)
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max)
    other_part.extend(other)
    # For symbolic alpha,alphap,beta,betap
    x = abs(a-b)
    y = a+b
    index_expr = (y+1-x)*(x+y+1)
    index_max = (c-x)*(x+c)+c+gamma
    cg_part, other = __match_cg_add(cg_part, expr, simp, param, const_param, index_expr, index_max)
    other_part.extend(other)

    return Add(*cg_part)+Add(*other_part)

def __match_cg_add(terms, expr, simp, wilds, const, index, index_max, index_sign=1):
    """Perform Clebsh-Gordan coefficient simplification

    Checks for simplifications that can be made, returning a tuple of the
    simplified list of terms and any terms generated by simplification.

    Parameters
    ==========

    term : list
        A list of all of the terms is the sum to be simplified

    expr : expression
        The expression with Wild terms that will be matched to the terms in
        the sum

    simp : expression
        The expression with Wild terms that is substituted in place of the CG
        terms in the case of simplification

    wilds : list
        A list of all the variables that appears in expr

    const : list
        A list of variables that can be used to characterized the sum, such
        that it is unique for any distinc sum that simplifies, i.e. the free
        variables of the sum.

    index_expr : expression
        Expression with Wild terms giving the index terms have when storing
        them to cg_index

    index_max : expression
        Expression with Wild terms giving the number of elements in cg_index

    index_sign : expression
        The expression with Wild terms denoting the sign that is on expr that
        must match all terms. This is used when terms have alternating signs
        (which the pattern matching has trouble detecting).

    """
    from sympy import sign
    lt = Wild('lt')
    wilds += (lt,)
    expr *= lt
    # determine all matches
    # they are placed in a dict, indexed by the const terms
    matches = {}
    cg_terms = []
    other_terms = []
    for term in terms:
        match_sub = term.match(expr)
        # match fails, so add term to list of cg terms
        if match_sub is None:
            cg_terms.append(term)
            continue
        # match succeeds, add to matches dict
        match_const = tuple([match_sub[i] for i in const] + [(sign(lt)*index_sign).subs(match_sub)])
        match_list = matches.pop(match_const, [])
        match_list.append(match_sub)
        matches[match_const] = match_list

    for match_list in matches.itervalues():
        # the list of the indicies for each term matched
        index_list = [index.subs(match) for match in match_list]
        # the indicies that we must have
        index_range_max = index_max.subs(match_list[0])
        if not index_range_max.is_number:
            cg_terms.extend([expr.subs(term) for term in match_list])
            continue
        index_range = range(index_range_max+1)
        # if all the indicies were matched
        if all([i in index_list for i in index_range]):
            # list of all matches with correct index
            cg_matches = [match_list[index_list.index(i)] for i in index_range]
            # list of all other matches
            other_matches = [i for i in match_list if i not in cg_matches]
            # determine the new term and add to other_terms
            new_lt = sign(lt*index_sign).subs(cg_matches[0]) * min([abs(match[lt]) for match in cg_matches])
            new_term = new_lt * simp.subs(match_list[0])
            other_terms.append(new_term)
            # based on leading terms, create new list of terms that survive
            for match in cg_matches:
                match[lt] -= new_lt * sign(index_sign).subs(match)
            cg_matches = [match for match in cg_matches if not match[lt] == 0]
            # add surviving terms to cg_terms
            cg_terms.extend([expr.subs(term) for term in cg_matches])
            cg_terms.extend([expr.subs(term) for term in other_matches])
        else:
            cg_terms.extend([expr.subs(term) for term in match_list])

    return cg_terms, other_terms

def _cg_simp_sum(e):
    # Varshalovich Eq 8.7.1 Eq 1
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    a, alpha, b = symbols('a alpha b', cls=Wild)
    match = e.match(Sum(CG(a,alpha,b,0,a,alpha),(alpha,-a,a)))
    if match is not None:
        return ((2*a+1)*KroneckerDelta(b,0)).subs(match)
    # Varshalovich Eq 8.7.1 Eq 2
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    a, alpha, c = symbols('a alpha c', cls=Wild)
    match = e.match(Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a)))
    if match is not None:
        return (sqrt(2*a+1)*KroneckerDelta(c,0)).subs(match)
    # Varshalovich Eq 8.7.2 Eq 9
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    a, alpha, b, beta, c, cp, gamma, gammap = symbols('a alpha b beta c cp gamma gammap', cls=Wild)
    match = e.match(Sum(CG(a,alpha,b,beta,c,gamma)*CG(a,alpha,b,beta,cp,gammap),(alpha,-a,a),(beta,-b,b)))
    if match is not None:
        return (KroneckerDelta(c,cp)*KroneckerDelta(gamma,gammap)).subs(match)
    match = e.match(Sum(CG(a,alpha,b,beta,c,gamma)**2,(alpha,-a,a),(beta,-b,b)))
    if match is not None:
        return 1
    return e
