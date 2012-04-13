"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory of
Many-Particle Systems."
"""
from sympy import Add, Dummy, Expr, Mul, S, sqrt, sympify
from sympy.printing.pretty.stringpict import prettyForm
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum import Bra, FockSpace, Ket, Operator, State

__all__ = [
    'B',
    'Bd',
    'F',
    'Fd',
    'BBra',
    'BKet',
    'FBra',
    'FKet',
    'NO',
    'contraction',
    'wicks'
]

#-----------------------------------------------------------------------------
# Second Quant Exceptions
#-----------------------------------------------------------------------------

class SecondQuantizationError(Exception):
    pass

class ViolationOfPauliPrinciple(Exception):
    pass

#-----------------------------------------------------------------------------
# Second Quantization Operators
#-----------------------------------------------------------------------------

class SecondQuantOpBase(Operator):
    """Base class for second quantization operators."""

    op_label = 'c'
    op_decorator = None
    op_udecorator = None
    op_latexdecorator = None

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    @property
    def is_symbolic(self):
        """Returns True if the state is a symbol (as opposed to a number).

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.quantum.secondquant import F
        >>> p = Symbol('p')
        >>> F(p).is_symbolic
        True
        >>> F(1).is_symbolic
        False

        """
        return self.state.is_Integer is not True

    @property
    def state(self):
        """Returns the state index related to the operator.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.quantum.secondquant import F, Fd, B, Bd
        >>> p = Symbol('p')
        >>> F(p).state
        p
        >>> Fd(p).state
        p
        >>> B(p).state
        p
        >>> Bd(p).state
        p

        """
        return self.label[0]

    def _sympystr(self, printer, *args):
        if self.op_decorator:
            op = "%s%s" % (self.op_label, self.op_decorator)
        else:
            op = "%s" % self.op_label
        return "%s(%s)" % (op, printer._print(self.state))

    def _pretty(self, printer, *args):
        op = printer._print(self.op_label)
        if self.op_udecorator:
            d = printer._print(self.op_udecorator)
            op = self._print_superscript_pretty(op, d)
        label = printer._print(self.state)
        label = prettyForm(*label.parens())
        return prettyForm(*op.right(label))

    def _latex(self, printer, *args):
        op = printer._print(self.op_label)
        if self.op_latexdecorator:
            op = r'%s^{%s}' % (op, self.op_latexdecorator)
        label = printer._print(self.state)
        return r'%s_{%s}' % (op, label)

class BosonicOperator(object):
    """Base class for boson operators."""

    op_label = 'b'

    def _apply_operator_FockStateBosonKet(self, ket, **options):
        if not self.is_symbolic:
            return self._apply_boson(ket, **options)

class FermionicOperator(object):
    """Base class for fermion operators."""

    op_label = 'a'

    def _apply_operator_FockStateFermionKet(self, ket, **options):
        return self._apply_fermion(ket, **options)

    @property
    def is_above_fermi(self):
        """Does this operator allow values above the fermi level

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import F
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')

        >>> F(p).is_above_fermi
        True
        >>> F(h).is_above_fermi
        False
        >>> F(n).is_above_fermi #doctest: +SKIP

        """
        bf = self.state.assumptions0.get('below_fermi')
        af = self.state.assumptions0.get('above_fermi')
        if af:
            return True
        if bf:
            return False

    @property
    def is_below_fermi(self):
        """Does this operator allow values above the fermi level

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import F
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')

        >>> F(p).is_below_fermi
        False
        >>> F(h).is_below_fermi
        True
        >>> F(n).is_below_fermi #doctest: +SKIP

        """
        bf = self.state.assumptions0.get('below_fermi')
        af = self.state.assumptions0.get('above_fermi')
        if af:
            return False
        if bf:
            return True

class AnnihilatorOpBase(SecondQuantOpBase):
    """Base class for annihilation operators."""

    def _apply_boson(self, ket, **options):
        state = self.state
        if len(ket) <= state:
            return S.Zero
        a = sqrt(ket.occupation[state])
        return a * ket.down(state)

    def _apply_fermion(self, ket, **options):
        return ket.down(self.state)

class CreatorOpBase(SecondQuantOpBase):
    """Base class for creation operators."""

    op_decorator = '+'
    op_udecorator = u'\u2020'
    op_latexdecorator = r'\dag'

    def _apply_boson(self, ket, **options):
        state = self.state
        if len(ket) <= state:
            a = 1
        else:
            a = sqrt(ket.occupation[state]+1)
        return a * ket.up(state)

    def _apply_fermion(self, ket, **options):
        return ket.up(self.state)

class AnnihilateBoson(BosonicOperator, AnnihilatorOpBase):
    """Operator for annihilating boson states.

    Examples
    ========

    >>> from sympy.physics.quantum.secondquant import B
    >>> from sympy.abc import x
    >>> B(x)
    b(x)

    """

    def _eval_commutator_AnnihilateBoson(self, other):
        return S.Zero

    def _eval_commutator_CreateBoson(self, other):
        return KroneckerDelta(self.state, other.state)

    def _eval_dagger(self):
        return CreateBoson(self.state)

class CreateBoson(BosonicOperator, CreatorOpBase):
    """Operator for creating boson states."""

    def _eval_commutator_AnnihilateBoson(self, other):
        return -KroneckerDelta(self.state, other.state)

    def _eval_commutator_CreateBoson(self, other):
        return S.Zero

    def _eval_dagger(self):
        return AnnihilateBoson(self.state)

class AnnihilateFermion(FermionicOperator, AnnihilatorOpBase):
    """Operator for annihilating fermion states."""

    def _eval_commutator_FermionicOpBase(self, other):
        return wicks(self * other) - wicks(other * self)

    def _eval_dagger(self):
        return CreateFermion(self.state)

    def _apply_contraction(self, other):
        if not isinstance(other, FermionicOperator):
            raise NotImplementedError
        if isinstance(other, CreateFermion):
            if other.is_below_fermi or self.is_below_fermi:
                return S.Zero
            if other.is_above_fermi or self.is_above_fermi:
                return KroneckerDelta(self.state, other.state)
            return KroneckerDelta(self.state, other.state) * \
                    KroneckerDelta(other.state, Dummy('p', above_fermi=True))
        return S.Zero

    @property
    def q_creator(self):
        """Does the operator create a quasi-particle

        Will return -1 if the annihilator can act below the fermi level (i.e.
        creating a hole).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import F
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')
        >>> F(p).q_creator
        0
        >>> F(h).q_creator
        -1
        >>> F(n).q_creator
        -1

        """
        if self.is_below_fermi is not False:
            return -1
        return 0

    @property
    def q_annihilator(self):
        """Does the operator destroy a quasi-particle

        Will return 1 if the annihilator can act above the fermi level (i.e.
        destroying a particle).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import F
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')
        >>> F(p).q_annihilator
        1
        >>> F(h).q_annihilator
        0
        >>> F(n).q_annihilator
        1

        """
        if self.is_above_fermi is not False:
            return 1
        return 0

class CreateFermion(FermionicOperator, CreatorOpBase):
    """Operator for annihilating fermion states."""

    def _eval_commutator_FermionicOpBase(self, other):
        return wicks(self * other) - wicks(other * self)

    def _eval_dagger(self):
        return AnnihilateFermion(self.state)

    def _apply_contraction(self, other):
        if not isinstance(other, FermionicOperator):
            raise NotImplementedError
        if isinstance(other, AnnihilateFermion):
            if other.is_below_fermi or self.is_below_fermi:
                return KroneckerDelta(self.state, other.state)
            if other.is_above_fermi or self.is_above_fermi:
                return S.Zero
            return KroneckerDelta(self.state, other.state) * \
                    KroneckerDelta(other.state, Dummy('p', above_fermi=True))
        return S.Zero

    @property
    def q_creator(self):
        """Does the operator create a quasi-particle

        Will return 1 if the creator can act above the fermi level (i.e.
        creating a particle).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import Fd
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')
        >>> Fd(p).q_creator
        1
        >>> Fd(h).q_creator
        0
        >>> Fd(n).q_creator
        1

        """
        if self.is_above_fermi is not False:
            return 1
        return 0

    @property
    def q_annihilator(self):
        """Does the operator destroy a quasi-particle

        Will return -1 if the creator can act below the fermi level (i.e.
        creating a hole).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import Fd
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> n = symbols('n')
        >>> Fd(p).q_annihilator
        0
        >>> Fd(h).q_annihilator
        -1
        >>> Fd(n).q_annihilator
        -1

        """
        if self.is_below_fermi is not False:
            return -1
        return 0

B = AnnihilateBoson
Bd = CreateBoson
F = AnnihilateFermion
Fd = CreateFermion

#-----------------------------------------------------------------------------
# Fock States
#-----------------------------------------------------------------------------

class FockState(State):
    """Base class for many particle Fock state."""

    _label_separator = ','

    def __len__(self):
        return len(self.occupation)

    def __getitem__(self, i):
        i = int(i)
        return self.occupation[i]

    @classmethod
    def _eval_hilbert_space(cls, label):
        return FockSpace()

    @property
    def label(self):
        return self.args[0]

    @property
    def occupation(self):
        return self.label

class BosonState(FockState):
    """Base class for bosonic Fock state."""

    def _eval_innerproduct_FockStateBosonBra(self, bra, **hints):
        occ_bra = bra.occupation
        occ_ket = self.occupation
        result = KroneckerDelta(len(occ_bra), len(occ_ket))
        for i, j in zip(occ_bra, occ_ket):
            result *= KroneckerDelta(i, j)
            if result == 0:
                break
        return result

    #def _represent_VarBosonicBasis(self, basis, **options):
    #    return

    def up(self, state):
        """Creates a particle at a given state.

        Examples
        ========

        >>> from sympy.physics.quantum.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        <1,2|
        >>> b.up(1)
        <1,3|

        """
        occ = list(self.occupation)
        if len(self) <= state:
            add_occ = [S.Zero] * int(state - len(self) + 1)
            occ.extend(add_occ)
        occ[state] += 1
        return self.__class__(occ)

    def down(self, state):
        """Annihilates a particle at a given state.

        Examples
        ========

        >>> from sympy.physics.quantum.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        <1,2|
        >>> b.down(1)
        <1,1|

        """
        occ = list(self.occupation)
        if len(occ) <= state:
            return S.Zero
        if occ[state] == 0:
            return S.Zero
        occ[state] -= 1
        return self.__class__(occ)

class FermionState(FockState):
    """Base class for fermionic Fock state."""

    def __new__(cls, occupation, fermi_level=0):
        occupation = map(sympify, occupation)
        if len(occupation) > 1:
            try:
                occupation, sign = _sort_anticommuting(occupation, key=cls._sort_key)
            except ViolationOfPauliPrinciple:
                return S.Zero
            sign = (S.NegativeOne)**sign
        else:
            sign = 1

        holes = cls._count_holes(occupation, fermi_level)
        if holes > fermi_level:
            return S.Zero

        return sign * FockState.__new__(cls, occupation, fermi_level)

    @property
    def fermi_level(self):
        return self.args[1]

    @classmethod
    def _sort_key(cls, state):
        # Fermion sort key for occupations
        h = hash(state)
        if state.assumptions0.get('above_fermi'):
            i = 0
        elif state.assumptions0.get('below_fermi'):
            i = 1
        else:
            i = 2
        if isinstance(state, Dummy):
            i += 10
        return i, h

    @classmethod
    def _count_holes(cls, occupation, fermi_level):
        """ Counts the number of excitations that are considered holes."""
        return len([i for i in occupation if cls._only_below_fermi(i, fermi_level)])

    def _eval_innerproduct_FockStateBosonBra(self, bra, **hints):
        occ_bra = bra.occupation
        occ_ket = self.occupation
        result = KroneckerDelta(len(occ_bra), len(occ_ket))
        for i, j in zip(occ_bra, occ_ket):
            result *= KroneckerDelta(i, j)
            if result == 0:
                break
        return result

    def up(self, state):
        """Creates a particle at a given state.

        If the excitation is created above the fermi level, we try to create a
        particle, and if the excitiation is below the fermi level, we remove a
        hole.

        If the excitation index cannot be determined to be above or below the
        fermi level, this given a factor of ``KroneckerDelta(p,i)`` where ``i``
        is a new symbol, potentially with fermi level restrictions to either be
        above or below the fermi level.

        Examples
        ========

        A creator acting on the vacuum state above the fermi level creates an
        excitation:

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import FKet
        >>> a = symbols('a', above_fermi=True)
        >>> FKet([]).up(a)
        |a>

        A creator acting on the vacuum state below the fermi level destroys the
        state:

        >>> i = symbols('i', below_fermi=True)
        >>> FKet([]).up(i)
        0

        A general creator acting on the vacuum state gives a delta function,
        with the dummy index either above or below the fermi level.

        >>> p = symbols('p')
        >>> FKet([]).up(p)
        |p>
        >>> FKet([], 3).up(p)
        KroneckerDelta(p, _a)*|p>

        """
        present = state in self.occupation

        if self._only_below_fermi(state, self.fermi_level):
            if present:
                # Create hole excitation
                return self._remove_orbit(state)
            else:
                # No hole, so excitation still present below fermi level
                return S.Zero
        elif self._only_above_fermi(state, self.fermi_level):
            if present:
                # Particle excitation already exsits above fermi level
                return S.Zero
            else:
                # Create particle excitation
                return self._add_orbit(state)
        else:
            if present:
                # Non-zero state corresponds to raising a hole
                hole = Dummy("i",below_fermi=True)
                return KroneckerDelta(state,hole)*self._remove_orbit(state)
            else:
                # Non-zero state corresponds to creating a particle
                particle = Dummy("a",above_fermi=True)
                return KroneckerDelta(state,particle)*self._add_orbit(state)


    def down(self, state):
        """Annihilates a particle at a given state.

        If the excitation is annihilated above the fermi level, we try to
        annihilate a particle, and if the excitiation is below the fermi level,
        we create a hole.

        If the excitation index cannot be determined to be above or below the
        fermi level, this given a factor of ``KroneckerDelta(p,i)`` where ``i``
        is a new symbol, potentially with fermi level restrictions to either be
        above or below the fermi level.

        Examples
        ========

        An annihilator acting on a vacuum above the fermi level destroys the
        state:

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import FKet
        >>> a = symbols('a', above_fermi=True)
        >>> FKet([]).down(a)
        0

        An annihilator acting below the fermi level will not vanish, provided the fermi level is greater than 0:

        >>> i = symbols('i', below_fermi=True)
        >>> FKet([]).down(i)
        0
        >>> FKet([], fermi_level=4).down(i)
        |i>

        When the position of the excitation cannot be determined, this returns
        a delta function, either above of below the fermi level:

        >>> p = symbols('p')
        >>> FKet([]).down(p)
        0
        >>> FKet([], 4).down(p)
        KroneckerDelta(p, _a)*|p>

        """
        present = state in self.occupation

        if self._only_above_fermi(state, self.fermi_level):
            if present:
                return self._remove_orbit(state)
            else:
                return S.Zero
        elif self._only_below_fermi(state, self.fermi_level):
            if present:
                return S.Zero
            else:
                return self._add_orbit(state)
        else:
            if present:
                hole = Dummy("i", below_fermi=True)
                d = KroneckerDelta(state, hole)
                s = self._remove_orbit(state)
            else:
                particle = Dummy("a", above_fermi=True)
                d = KroneckerDelta(state, particle)
                s = self._add_orbit(state)
            return d * s

    def _remove_orbit(self, state):
        new_occ = list(self.occupation)
        i = new_occ.index(state)
        new_occ.remove(state)
        if i % 2:
            coeff = -1
        else:
            coeff = 1
        return coeff * self.__class__(new_occ, self.fermi_level)

    def _add_orbit(self, state):
        return self.__class__((state,) + self.occupation, self.fermi_level)

    @classmethod
    def _only_below_fermi(cls, state, fermi_level):
        if state.is_number:
            return state <= fermi_level
        if state.assumptions0.get('below_fermi'):
            return True
        return False

    @classmethod
    def _only_above_fermi(cls, state, fermi_level):
        if state.is_number:
            return state > fermi_level
        if state.assumptions0.get('above_fermi'):
            return True
        return fermi_level == 0

class FockStateBosonKet(BosonState, Ket):
    """Many particle boson state."""
    @classmethod
    def dual_class(self):
        return FockStateBosonBra

class FockStateBosonBra(BosonState, Bra):
    """Many particle boson state."""
    @classmethod
    def dual_class(self):
        return FockStateBosonKet

class FockStateFermionKet(FermionState, Ket):
    """Mant particle fermion state."""
    @classmethod
    def dual_class(self):
        return FockStateFermionBra

class FockStateFermionBra(FermionState, Bra):
    """Mant particle fermion state."""
    @classmethod
    def dual_class(self):
        return FockStateFermionKet

BBra = FockStateBosonBra
BKet = FockStateBosonKet
FBra = FockStateFermionBra
FKet = FockStateFermionKet

# TODO Allow NO to take boson operators
class NO(Expr):
    """Normal ordering function of given operators

    Represents the normal ordering of a product of operators. A normal ordered
    product of operators has all the creation operators on the left and all the
    annihilation operators on the right.

    The normal ordering implementation currently assumes all operators
    anticummute and have vanishing contractions. This allows immediate
    reordering to canonical form.

    Examples
    ========

    >>> from sympy.abc import p, q
    >>> from sympy.physics.quantum.secondquant import NO, F, Fd
    >>> NO(Fd(p) * F(q))
    NO(a+(p)*a(q))
    >>> NO(F(q) * Fd(p))
    -NO(a+(p)*a(q))

    See Also
    ========

    wicks: Generate normal ordered equivalent of an expression

    """

    nargs = 1
    is_commutative = False

    def __new__(cls, arg):
        arg = sympify(arg).expand()

        # {ab + cd} = {ab} + {cd}
        if arg.is_Add:
            return Add(*[ cls(term) for term in arg.args])

        if arg.is_Mul:
            c_part, nc_part = arg.args_cnc()
            if c_part:
                coeff = Mul(*c_part)
            else:
                coeff = S.One
            if not nc_part:
                return coeff

            # {ab{cd}} = {abcd}
            if [arg for arg in nc_part if isinstance(arg, NO)]:
                new_args = []
                for arg in nc_part:
                    if isinstance(arg, NO):
                        new_args.extend(arg.args)
                    else:
                        new_args.append(arg)
                return coeff * cls(Mul(*new_args))

            try:
                new_args, sign = _sort_anticommuting(nc_part)
            except ViolationOfPauliPrinciple:
                return S.Zero

            if sign:
                sign = S.NegativeOne**sign
                return sign * coeff * cls(Mul(*new_args))

            # Since sign == 0 didn't need reordering
            if coeff != S.One:
                return coeff * cls(Mul(*new_args))
            return Expr.__new__(cls, Mul(*new_args))

        if isinstance(arg, NO):
            return arg

        # if object was not Mul or Add, normal ordering does not apply
        return arg

    @property
    def has_q_creators(self):
        """Does the normal ordering have quasiparticle creators

        Return 0 if the leftmost argument of the first argument does not create
        a quasiparticle, else 1 if it is above the fermi level (particle) or -1
        if it is below the fermi level (hole).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import NO, F, Fd
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> NO(Fd(p) * Fd(h)).has_q_creators
        1
        >>> NO(F(h) * F(p)).has_q_creators
        -1
        >>> NO(Fd(h) * F(p)).has_q_creators
        0

        """
        return self.args[0].args[0].q_creator

    @property
    def has_q_annihilators(self):
        """Does the normal ordering have a quasiparticle annihilator

        Return 0 if the rightmost argument of the first argument does not
        destroy a quasiparticle, else 1 if it is above the fermi level
        (particle) or -1 if it is below the fermi level (hole).

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.quantum.secondquant import NO, Fd, F
        >>> p = symbols('p', above_fermi=True)
        >>> h = symbols('h', below_fermi=True)
        >>> NO(Fd(p) * Fd(h)).has_q_annihilators
        -1
        >>> NO(F(h) * F(p)).has_q_annihilators
        1
        >>> NO(F(h) * Fd(p)).has_q_annihilators
        0

        """
        return self.args[0].args[-1].q_annihilator

    def _remove_brackets(self):
        """Returns sorted expression without normal order brackets

        The returned expression has the property that no nonzero contractions
        exist.
        """
        # check if any creator is also an annihilator
        pass

def contraction(a, b):
    """
    Calculates contraction of Fermionic operators a and b.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.secondquant import F, Fd, contraction
    >>> p, q = symbols('p,q')
    >>> a, b = symbols('a,b', above_fermi=True)
    >>> i, j = symbols('i,j', below_fermi=True)

    A contraction is non-zero only if a quasi-creator is to the right of a
    quasi-annihilator:

    >>> contraction(F(a),Fd(b))
    KroneckerDelta(a, b)
    >>> contraction(Fd(i),F(j))
    KroneckerDelta(i, j)

    For general indices a non-zero result restricts the indices to below/above
    the fermi surface:

    >>> contraction(Fd(p),F(q))
    KroneckerDelta(p, q)*KroneckerDelta(q, _p)
    >>> contraction(F(p),Fd(q))
    KroneckerDelta(p, q)*KroneckerDelta(q, _p)

    Two creators or two annihilators always vanishes:

    >>> contraction(F(p),F(q))
    0
    >>> contraction(Fd(p),Fd(q))
    0

    """
    try:
        r = a._apply_contraction(b)
    except NotImplementedError:
        t = ( isinstance(i,FermionicOperator) for i in (a,b) )
        raise ContractionAppliesOnlyToFermions(*t)
    return r

def wicks(e, **kw_args):
    """Determines the normal ordered equivalent of an expression using Wicks
    Theorem

    Examples
    ========

    >>> from sympy import symbols, Function
    >>> from sympy.physics.quantum.secondquant import wicks, F, Fd, NO
    >>> p, q, r = symbols('p q r')
    >>> wicks(Fd(p) * F(q))

    By default, the expression is expanded:

    >>> wicks(F(p)*(F(q)+F(r)))

    With the keyword 'keep_only_fully_contracted=True', only fully contracted
    terms are returned.

    By request, the result can be simplified in the following order:
      1. KroneckerDelta functions are evaluated
      2. Dummy variables are substituted consistently across terms

    """
    if not e:
        return S.Zero

    opts = {'simplify_kronecker_deltas': False,
            'expand': True,
            'simplify_dummies': False,
            'keep_only_fully_contracted': False }
    opts.update(kw_args)

    # Check if we already have normal order
    if isinstance(e, NO) or isinstance(e, FermionicOperator):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e

    e = e.doit(wicks=True)

    e = e.expand()
    if isinstance(e, Add):
        if opts['simplify_dummies']:
            return substitute_dummies(Add(*[ wicks(term, **kw_args) for term in e.args]))
        else:
            return Add(*[ wicks(term, **kw_args) for term in e.args])

    # For Mul-objects we can actually do something
    if isinstance(e, Mul):
        # we dont want to mess around with commuting part of Mul
        # so we factorize it out before starting recursion
        c_part, nc_part = e.args_cnc()
        n = len(nc_parts)

        if n == 0:
            result = e
        elif n == 1:
            if opts['keep_opts_fully_contracted']:
                return S.Zero
            else:
                result = e
        else:
            result = _get_contractions(nc_part,
                    keep_only_fully_contracted=opts['keep_only_fully_contracted'] )
            result *= Mul(*c_part)

        if opts['expand']:
            result = result.expand()
        if opts['simplify_kronecker_deltas']:
            result = evaluate_deltas(result)

        return result
    # There was nothing to do
    return e

def _fermionic_key(state):
    # Sort key for fermionic operators
    h = hash(state)
    label = str(state.label)

    if state.q_creator:
        return 1, label, h
    if state.q_annihilator:
        return 4, label, h
    if isinstance(state, AnnihilatorOpBase):
        return 3, label, h
    if isinstance(state, CreatorOpBase):
        return 2, label, h

def _sort_anticommuting(occupation, key=_fermionic_key):
    verified = False
    sign = 0
    end = len(occupation) - 1

    keys = list(map(key, occupation))
    key_to_state = dict(zip(keys, occupation))

    while not verified:
        verified = True
        for i in range(end):
            l = keys[i]
            r = keys[i+1]
            if l == r:
                raise ViolationOfPauliPrinciple([l, r])
            if l > r:
                verified = False
                keys[i:i+2] = [r, l]
                sign += 1
        end -= 1
    occupation = [ key_to_state[k] for k in keys ]
    return occupation, sign
