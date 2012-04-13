"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory of
Many-Particle Systems."
"""
from sympy import Dummy, Expr, S, sqrt, sympify
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
            return self._apply_sqop(ket, **options)

class FermionicOperator(object):
    """Base class for fermion operators."""

    op_label = 'a'

    def _apply_operator_FockStateFermionKet(self, ket, **options):
        if not self.is_symbolic:
            return self._apply_sqop(ket, **options)

class AnnihilatorOpBase(SecondQuantOpBase):
    """Base class for annihilation operators."""

    def _apply_sqop(self, ket, **options):
        state = self.state
        if len(ket) <= state:
            return S.Zero
        a = sqrt(ket.occupation[state])
        return a * ket.down(state)

class CreatorOpBase(SecondQuantOpBase):
    """Base class for creation operators."""

    op_decorator = '+'
    op_udecorator = u'\u2020'
    op_latexdecorator = r'\dag'

    def _apply_sqop(self, ket, **options):
        state = self.state
        if len(ket) <= state:
            a = 1
        else:
            a = sqrt(ket.occupation[state]+1)
        return a * ket.up(state)

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

class CreateFermion(FermionicOperator, CreatorOpBase):
    """Operator for annihilating fermion states."""

    def _eval_commutator_FermionicOpBase(self, other):
        return wicks(self * other) - wicks(other * self)

    def _eval_dagger(self):
        return AnnihilateFermion(self.state)

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

def _fermion_key(state):
    h = hash(state)
    if index.assumptions0.get('above_fermi'):
        i = 0
    elif index.assumptions0.get('below_fermi'):
        i = 1
    else:
        i = 2

    if isinstance(index, Dummy):
        i += 10

    return i, h

class FermionState(FockState):
    """Base class for fermionic Fock state."""

    def __new__(cls, occupation, fermi_level=0):
        occupation = map(sympify, occupation)
        if len(occupation) > 1:
            try:
                occupation, sign = cls._sort_occupation(occupations)
            except ViolationOfPauliPrinciple:
                return S.Zero
        else:
            sign = 1

        return sign * FockState.__new__(cls, occupation, fermi_level)

    @property
    def fermi_level(self):
        return self.args[1]

    @classmethod
    def _sort_occupation(cls, occupation, key=_fermion_key):
        verified = False
        sign = 1
        end = len(occupation) - 1
        rev = range(len(occupation)-3, -1, -1)

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
                    sign *= -1
            end -= 1
        occupation = [ key_to_state[k] for k in keys ]
        return occupation, sign

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

        if self._only_below_fermi(state):
            if present:
                # Create hole excitation
                return self._remove_orbit(state)
            else:
                # No hole, so excitation still present below fermi level
                return S.Zero
        elif self._only_above_fermi(state):
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

        if self._only_above_fermi(state):
            if present:
                return ket._remove_orbit(i)
            else:
                return S.Zero
        elif self._only_below_fermi(state):
            if present:
                return S.Zero
            else:
                return self._add_orbit(state)
        else:
            if present:
                hole = Dummy("i", below_fermi=True)
                d = KroneckerDelta(state, hole)
                s = ket._remove_orbit(state)
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

    def _only_below_fermi(self, state):
        if state.is_number:
            return state <= self.fermi_level
        if state.assumptions0.get('below_fermi'):
            return True
        return False

    def _only_above_fermi(self, state):
        if state.is_number:
            return i > self.fermi_level
        if state.assumptions0.get('above_fermi'):
            return True
        return self.fermi_level == 0

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

class NO(Expr):
    # TODO
    pass

def contraction(a, b):
    # TODO
    pass

def wicks(e, **kw_args):
    # TODO
    pass
