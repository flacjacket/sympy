from sympy import Expr, S, sqrt, sympify
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
        """
        Returns True if the state is a symbol (as opposed to a number).

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> p = Symbol('p')
        >>> F(p).is_symbolic
        True
        >>> F(1).is_symbolic
        False

        """
        return self.state.is_Integer is not True

    @property
    def state(self):
        """Returns the state index related to the operator."""
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
        return r'%s\left(%s\right)' % (op, label)

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
    """Operator for annihilating boson states."""

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
    def occupation(self):
        return self.label[0]

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
        occ = list(self.occupation)
        if len(occ) <= state:
            add_occ = [S.Zero] * (state - len(occ) + 1)
            occ.extend(add_occ)
        occ[state] += 1
        return self.__class__(occ)

    def down(self, state):
        occ = list(self.occupation)
        if len(occ) <= state:
            return S.Zero
        if occ[state] == 0:
            return S.Zero
        occ[state] -= 1
        return self.__class__(occ)

class FermionState(FockState):
    """Base class for fermionic Fock state."""

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
        return
        # TODO
        present = i in self.args[0]

        if self._only_above_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif self._only_below_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        else:
            if present:
                hole = Dummy("i",below_fermi=True)
                return KroneckerDelta(i,hole)*self._remove_orbit(i)
            else:
                particle = Dummy("a",above_fermi=True)
                return KroneckerDelta(i,particle)*self._add_orbit(i)


    def down(self, state):
        return
        # TODO
        state = self.state
        present = state in ket.occupation
        
        if ket._only_above_fermi(state):
            if present:
                return ket._remove_orbit(i)
            else:
                return S.Zero
        elif ket._only_below_fermi(state):
            if present:
                return S.Zero
            else:
                return self._add_orbit(state)
        else:
            if present:
                hole = Dummy("i", below_fermi=True)
                d = KroneckerDelta(state, hole)
                s = ket._add_orbit(state)
            else:
                particle = Dummy("a", above_fermi=True)
                d = KroneckerDelta(state, particle)
                s = ket._remove_orbit(state)
            return d * s

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

class FockStateFermionKet(BosonState, Ket):
    """Mant particle fermion state."""
    @classmethod
    def dual_class(self):
        return FockStateFermionBra

class FockStateFermionBra(BosonState, Bra):
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
