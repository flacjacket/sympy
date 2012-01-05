# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

# This suite checks the latex output of the quantum mechanics module and its
# rendering in the IPython Qt console and HTML notebook. All tests run by
# test_printing.py should appear here.

# To update this file, change the HTML notebook, run all cells and save. Then,
# from the notebook, export the file to a .py file for the Qt console.

# To view the tests in the Qt console, run 'ipython-qtconsole --profile=sympy'
# and execute '%run 'examples/notebooks/quantum printing tester.py''

# <codecell>

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct

from sympy import Derivative, Function, Interval, Matrix, oo, Pow, S, Symbol, symbols

from IPython.core.display import display

# <codecell>

#anticommutator
A = Operator('A')
B = Operator('B')
ac = AntiCommutator(A,B)
ac_tall = AntiCommutator(A**2,B)

# <codecell>

display(ac)

# <codecell>

display(ac_tall)

# <codecell>

#cg
cg = CG(1,2,3,4,5,6)
wigner3j = Wigner3j(1,2,3,4,5,6)

# <codecell>

display(cg)

# <codecell>

display(wigner3j)

# <codecell>

#commutator
A = Operator('A')
B = Operator('B')
c = Commutator(A,B)
c_tall = Commutator(A**2,B)

# <codecell>

display(c)

# <codecell>

display(c_tall)

# <codecell>

#constants

# <codecell>

display(hbar)

# <codecell>

#dagger
x = symbols('x')
expr = Dagger(x)

# <codecell>

display(expr)

# <codecell>

#gate
a,b,c,d = symbols('a b c d')
uMat = Matrix([[a,b],[c,d]])
q = Qubit(1,0,1,0,1)
g1 = IdentityGate(2)
g2 = CGate((3,0), XGate(1))
g3 = CNotGate(1,0)
g4 = UGate((0,), uMat)

# <codecell>

display(g1)

# <codecell>

display(g1*q)

# <codecell>

display(g2)

# <codecell>

display(g3)

# <codecell>

display(g4)

# <codecell>

#hilbert
h1 = HilbertSpace()
h2 = ComplexSpace(2)
h3 = FockSpace()
h4 = L2(Interval(0,oo))

# <codecell>

display(h1)

# <codecell>

display(h2)

# <codecell>

display(h3)

# <codecell>

display(h4)

# <codecell>

display(h1+h2)

# <codecell>

display(h1*h2)

# <codecell>

display(h1**2)

# <codecell>

#innerproduct
x = symbols('x')
ip1 = InnerProduct(Bra(), Ket())
ip2 = InnerProduct(TimeDepBra(), TimeDepKet())
ip3 = InnerProduct(JzBra(1,1), JzKet(1,1))
ip4 = InnerProduct(JzBraCoupled(1,1,1,1), JzKetCoupled(1,1,1,1))
ip_tall1 = InnerProduct(Bra(x/2), Ket(x/2))
ip_tall2 = InnerProduct(Bra(x), Ket(x/2))
ip_tall3 = InnerProduct(Bra(x/2), Ket(x))

# <codecell>

display(ip1)

# <codecell>

display(ip2)

# <codecell>

display(ip3)

# <codecell>

display(ip4)

# <codecell>

display(ip_tall1)

# <codecell>

display(ip_tall2)

# <codecell>

display(ip_tall3)

# <codecell>

#operator
a = Operator('A')
b = Operator('B',Symbol('t'),S(1)/2)
inv = a.inv()
f = Function('f')
x = symbols('x')
d = DifferentialOperator(Derivative(f(x),x),f(x))
op = OuterProduct(Ket(), Bra())

# <codecell>

display(a)

# <codecell>

display(inv)

# <codecell>

display(d)

# <codecell>

display(b)

# <codecell>

display(op)

# <codecell>

#qexpr
q = QExpr('q')

# <codecell>

display(q)

# <codecell>

#qubit
q1 = Qubit('0101')
q2 = IntQubit(8)

# <codecell>

display(q1)

# <codecell>

display(q2)

# <codecell>

#spin
ket = JzKet(1,1)
bra = JzBra(1,1)
cket = JzKetCoupled(1,1,1,1)
cbra = JzBraCoupled(1,1,1,1)
rot = Rotation(1,2,3)
bigd = WignerD(1,2,3,4,5,6)
smalld = WignerD(1,2,3,0,4,0)

# <codecell>

display(J2)

# <codecell>

display(Jz)

# <codecell>

display(ket)

# <codecell>

display(bra)

# <codecell>

display(cket)

# <codecell>

display(cbra)

# <codecell>

display(rot)

# <codecell>

display(bigd)

# <codecell>

display(smalld)

# <codecell>

#state
x = symbols('x')
bra = Bra()
ket = Ket()
bra_tall = Bra(x/2)
ket_tall = Ket(x/2)
tbra = TimeDepBra()
tket = TimeDepKet()

# <codecell>

display(bra)

# <codecell>

display(ket)

# <codecell>

display(bra_tall)

# <codecell>

display(ket_tall)

# <codecell>

display(tbra)

# <codecell>

display(tket)

# <codecell>

#tensorproduct
tp = TensorProduct(JzKet(1,1), JzKet(1,0))

# <codecell>

display(tp)

# <codecell>

#big expr
f = Function('f')
x = symbols('x')
e1 = Dagger(AntiCommutator(Operator('A')+Operator('B'),Pow(DifferentialOperator(Derivative(f(x), x), f(x)),3))*TensorProduct(Jz**2,Operator('A')+Operator('B')))*(JzBra(1,0)+JzBra(1,1))*(JzKet(0,0)+JzKet(1,-1))
e2 = Commutator(Jz**2,Operator('A')+Operator('B'))*AntiCommutator(Dagger(Operator('C')*Operator('D')),Operator('E').inv()**2)*Dagger(Commutator(Jz,J2))
e3 = Wigner3j(1,2,3,4,5,6)*TensorProduct(Commutator(Operator('A')+Dagger(Operator('B')),Operator('C')+Operator('D')),Jz-J2)*Dagger(OuterProduct(Dagger(JzBra(1,1)),JzBra(1,0)))*TensorProduct(JzKetCoupled(1,1,1,1)*JzKetCoupled(1,0,1,1),JzKetCoupled(1,-1,1,1))
e4 = (ComplexSpace(1)*ComplexSpace(2)+FockSpace()**2)*(L2(Interval(0,oo))+HilbertSpace())

# <codecell>

display(e1)

# <codecell>

display(e2)

# <codecell>

display(e3)

# <codecell>

display(e4)

