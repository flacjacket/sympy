from sympy.physics.quantum.secondquant import (
    B, Bd, CreateBoson, AnnihilateBoson, BosonicOperator, BBra, BKet,
    BosonState, F, Fd, CreateFermion, AnnihilateFermion, FBra, FKet,
    FermionicOperator, FermionState, NO, wicks, contraction, evaluate_deltas
)
from sympy.physics.quantum import (
    Commutator, Dagger, InnerProduct, qapply
)

from sympy import Dummy, expand, KroneckerDelta, sqrt, symbols
from sympy.utilities.pytest import XFAIL

h1, h2, h3 = symbols('h1 h2 h3',below_fermi=True)
p1, p2, p3 = symbols('p1 p2 p3',above_fermi=True)
n, m = symbols('n m')

def test_dagger():
    # Operators
    assert Dagger(B(0)) == Bd(0)
    assert Dagger(Bd(0)) == B(0)
    assert Dagger(B(n)) == Bd(n)
    assert Dagger(Bd(n)) == B(n)
    assert Dagger(F(0)) == Fd(0)
    assert Dagger(Fd(0)) == F(0)
    assert Dagger(F(n)) == Fd(n)
    assert Dagger(Fd(n)) == F(n)
    # States
    assert Dagger(BKet([n])) == BBra([n])
    assert Dagger(BBra([n])) == BKet([n])
    assert Dagger(FKet([n])) == FBra([n])
    assert Dagger(FBra([n])) == FKet([n])

def test_commutator():
    # Boson commutators
    assert Commutator(B(0), Bd(0)).doit() == 1
    assert Commutator(Bd(0), B(0)).doit() == -1
    assert Commutator(B(p1), Bd(p2)).doit() == KroneckerDelta(p1, p2)
    assert Commutator(Bd(p2), B(p1)).doit() == -KroneckerDelta(p1, p2)
    assert Commutator(B(0), B(1)).doit() == 0
    assert Commutator(Bd(0), Bd(1)).doit() == 0
    assert Commutator(B(p1), B(p2)).doit() == 0
    assert Commutator(Bd(p1), Bd(p2)).doit() == 0
    # Fermion commutator
    n, m = symbols("n,m", above_fermi=True)
    c = Commutator(F(m), Fd(m)).doit()
    assert c == 1 - 2*NO(Fd(m)*F(m))
    c = Commutator(Fd(m), F(m)).doit().expand()
    assert c == -1 + 2*NO(Fd(m)*F(m))

    C = Commutator
    X,Y,Z = symbols('X,Y,Z',commutative=False)
    assert C(C(X,Y),Z) != 0
    assert C(C(X,Z),Y) != 0
    assert C(Y,C(X,Z)) != 0

    i,j,k,l = symbols('i,j,k,l',below_fermi=True)
    a,b,c,d = symbols('a,b,c,d',above_fermi=True)
    p,q,r,s = symbols('p,q,r,s')
    D=KroneckerDelta

    assert C(Fd(a),F(i)).doit() == -2*NO(F(i)*Fd(a))
    # TODO: use wicks theorem to do commutator
    #assert C(Fd(j),NO(Fd(a)*F(i))).doit() == -D(j,i)*Fd(a)
    #assert C(Fd(a)*F(i),Fd(b)*F(j)).doit() == 0

def test_boson_operators():
    # Creation operators
    assert isinstance(Bd(0), CreateBoson)
    assert qapply(Bd(0) * BKet([0])) == BKet([1])
    assert qapply(Bd(0) * BKet([n])) == sqrt(n+1) * BKet([n+1])
    assert qapply(Bd(n) * BKet([0])) == Bd(n) * BKet([0])
    assert qapply(Bd(n) * BKet([n])) == Bd(n) * BKet([n])
    # Annihilation operators
    assert isinstance(B(0), AnnihilateBoson)
    assert qapply(B(0) * BKet([0])) == 0
    assert qapply(B(0) * BKet([1])) == BKet([0])
    assert qapply(B(0) * BKet([n])) == sqrt(n) * BKet([n-1])
    assert qapply(B(n) * BKet([0])) == B(n) * BKet([0])
    assert qapply(B(n) * BKet([n])) == B(n) * BKet([n])
    # Complex boson operators
    o = Bd(0) * B(0) * Bd(1) * B(0)
    e = qapply(o * BKet([n, m]))
    answer = sqrt(n)*sqrt(m+1)*(-1+n)*BKet([-1+n, 1+m])
    assert expand(e) == expand(answer)

def test_fermion_operators():
    # Creation operators
    assert isinstance(Fd(1), CreateFermion)
    assert qapply(Fd(1) * FKet([n])) == FKet([1, n])
    assert qapply(Fd(1) * FKet([n])) == -FKet([n, 1])
    assert qapply(Fd(n) * FKet([])) == FKet([n])
    assert qapply(Fd(p1) * FKet([])) == FKet([p1])
    assert qapply(Fd(h1) * FKet([])) == 0
    assert qapply(Fd(h1) * FKet([h1, h2], 4)) == FKet([h2], 4)
    assert qapply(Fd(h2) * FKet([h1, h2], 4)) == -FKet([h1], 4)
    assert qapply(Fd(h3) * FKet([h1, h2], 4)) == 0
    assert qapply(Fd(p1) * FKet([h1, h2, h3], 4)) == FKet([p1, h1, h2, h3], 4)
    # Annihilation operators
    assert isinstance(F(1), AnnihilateFermion)
    assert qapply(F(1) * FKet([1, n])) == FKet([n])
    assert qapply(F(1) * FKet([n, 1])) == -FKet([n])
    assert qapply(F(n) * FKet([n])) == FKet([])
    assert qapply(F(h1) * FKet([h1, h2, h3], 4)) == 0
    assert qapply(F(h1) * FKet([h2, h3], 4)) == FKet([h1, h2, h3], 4)
    assert qapply(F(h1) * FKet([h2, h3], 2)) == 0
    assert qapply(F(p1) * FKet([h1, p2, h2], 4)) == 0
    assert qapply(F(p3) * FKet([p1, p2], 4)) == 0
    assert qapply(F(p1) * FKet([p1, p2], 4)) == FKet([p2], 4)

def test_fermi_level():
    assert F(p1).is_below_fermi is False
    assert F(h1).is_below_fermi
    assert F(n).is_below_fermi is None
    assert F(p1).is_above_fermi
    assert F(h1).is_above_fermi is False
    assert F(n).is_above_fermi is None

def test_q_particle():
    # Create quasiparticle
    assert F(p1).q_creator == 0
    assert Fd(p1).q_creator == 1
    assert F(h1).q_creator == -1
    assert Fd(h1).q_creator == 0
    assert F(n).q_creator == -1
    assert Fd(n).q_creator == 1
    # Annihilate quasiparticles
    assert F(p1).q_annihilator == 1
    assert Fd(p1).q_annihilator == 0
    assert F(h1).q_annihilator == 0
    assert Fd(h1).q_annihilator == -1
    assert F(n).q_annihilator == 1
    assert Fd(n).q_annihilator == -1

def test_number_operator():
    n = symbols("n")
    # Boson number operator
    nb = Bd(0)*B(0)
    assert qapply(nb * BKet([n])) == n * BKet([n])
    assert qapply(nb * BKet([1, 2])) == BKet([1, 2])
    # Fermion number operator
    # TODO

def test_basic_state():
    i, n, m = symbols('i n m')
    s = BosonState([0, 1, 2, 3, 4])
    assert len(s) == 5
    assert s.occupation == tuple(range(5))
    assert s.up(0) == BosonState([1,1,2,3,4])
    assert s.down(4) == BosonState([0,1,2,3,3])
    for i in range(5):
        assert s.up(i).down(i) == s
    assert s.down(0) == 0
    for i in range(5):
        assert s[i] == i
    s = BosonState([n, m])
    assert s.down(0) == BosonState([n-1, m])
    assert s.up(0) == BosonState([n+1, m])

@XFAIL
def Xtest_move1():
    i, j = symbols('i,j')
    A, C = symbols('A,C', cls=Function)
    o = A(i)*C(j)
    # This almost works, but has a minus sign wrong
    assert move(o, 0, 1) == KroneckerDelta(i, j) + C(j)*A(i)

@XFAIL
def Xtest_move2():
    i, j = symbols('i,j')
    A, C = symbols('A,C', cls=Function)
    o = C(j)*A(i)
    # This almost works, but has a minus sign wrong
    assert move(o, 0, 1) == -KroneckerDelta(i, j) + A(i)*C(j)

def test_inner_product():
    i, j, k, l = symbols('i,j,k,l')
    s1 = BBra([0])
    s2 = BKet([1])
    assert InnerProduct(s1, Dagger(s1)).doit() == 1
    assert InnerProduct(s1, s2).doit() == 0
    s1 = BBra([i, j])
    s2 = BKet([k, l])
    r = InnerProduct(s1, s2).doit()
    assert r == KroneckerDelta(i, k)*KroneckerDelta(j, l)

def test_symbolic_matrix_elements():
    n, m = symbols('n m')
    s1 = BBra([n])
    s2 = BKet([m])
    o = B(0)
    e = qapply(s1 * o * s2)
    assert e == sqrt(m)*KroneckerDelta(n, m-1)

"""def test_PermutationOperator():
    p,q,r,s = symbols('p,q,r,s')
    f,g,h,i = map(Function, 'fghi')
    P = PermutationOperator
    assert P(p,q).get_permuted(f(p)*g(q)) == -f(q)*g(p)
    assert P(p,q).get_permuted(f(p, q)) == -f(q, p)
    assert P(p,q).get_permuted(f(p)) == f(p)
    expr = (f(p)*g(q)*h(r)*i(s)
        - f(q)*g(p)*h(r)*i(s)
        - f(p)*g(q)*h(s)*i(r)
        + f(q)*g(p)*h(s)*i(r))
    perms = [P(p,q),P(r,s)]
    assert (simplify_index_permutations(expr,perms) ==
        P(p,q)*P(r,s)*f(p)*g(q)*h(r)*i(s))"""

"""def test_index_permutations_with_dummies():
    a,b,c,d = symbols('a b c d')
    p,q,r,s = symbols('p q r s', cls=Dummy)
    f,g = map(Function, 'fg')
    P = PermutationOperator

    # No dummy substitution necessary
    expr = f(a, b, p, q) - f(b, a, p, q)
    assert simplify_index_permutations(
            expr, [P(a, b)]) == P(a, b)*f(a, b, p, q)

    # Cases where dummy substitution is needed
    expected = P(a, b)*substitute_dummies(f(a, b, p, q))

    expr = f(a, b, p, q) - f(b, a, q, p)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    expr = f(a, b, q, p) - f(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    # A case where nothing can be done
    expr = f(a, b, q, p) - g(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expr == result"""

"""def test_matrix_elements():
    b = VarBosonicBasis(5)
    o = B(0)
    m = matrix_rep(o, b)
    for i in range(4):
        assert m[i, i+1] == sqrt(i+1)
    o = Bd(0)
    m = matrix_rep(o, b)
    for i in range(4):
        assert m[i+1, i] == sqrt(i+1)"""

"""def test_sho():
    n, m = symbols('n,m')
    h_n = Bd(n)*B(n)*(n + Rational(1, 2))
    H = Sum(h_n, (n, 0, 5))
    o = H.doit(deep = False)
    b = FixedBosonicBasis(2, 6)
    m = matrix_rep(o, b)
    # We need to double check these energy values to make sure that they
    # are correct and have the proper degeneracies!
    diag = [1, 2, 3, 3, 4, 5, 4, 5, 6, 7, 5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11]
    for i in range(len(diag)):
        assert diag[i] == m[i, i]"""

def test_wicks():
    p,q,r,s = symbols('p,q,r,s',above_fermi=True)

    # Testing for particles only

    str = F(p)*Fd(q)
    assert wicks(str) == NO(F(p)*Fd(q)) + KroneckerDelta(p,q)
    str = Fd(p)*F(q)
    assert wicks(str) == NO(Fd(p)*F(q))


    str = F(p)*Fd(q)*F(r)*Fd(s)
    nstr= wicks(str)
    fasit = NO(
    KroneckerDelta(p, q)*KroneckerDelta(r, s)
    + KroneckerDelta(p, q)*AnnihilateFermion(r)*CreateFermion(s)
    + KroneckerDelta(r, s)*AnnihilateFermion(p)*CreateFermion(q)
    - KroneckerDelta(p, s)*AnnihilateFermion(r)*CreateFermion(q)
    - AnnihilateFermion(p)*AnnihilateFermion(r)*CreateFermion(q)*CreateFermion(s))
    assert nstr == fasit

    assert (p*q*nstr).expand() == wicks(p*q*str)
    assert (nstr*p*q*2).expand() == wicks(str*p*q*2)


    # Testing CC equations particles and holes
    i,j,k,l = symbols('i j k l',below_fermi=True,cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True,cls=Dummy)
    p,q,r,s = symbols('p q r s',cls=Dummy)

    assert (wicks(F(a)*NO(F(i)*F(j))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*Fd(b)) +
            KroneckerDelta(a,b)*NO(F(i)*F(j)))
    assert (wicks(F(a)*NO(F(i)*F(j)*F(k))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*F(k)*Fd(b)) -
            KroneckerDelta(a,b)*NO(F(i)*F(j)*F(k)))


    expr = wicks(Fd(i)*NO(Fd(j)*F(k))*F(l))
    assert (expr ==
           -KroneckerDelta(i,k)*NO(Fd(j)*F(l)) -
            KroneckerDelta(j,l)*NO(Fd(i)*F(k)) -
            KroneckerDelta(i,k)*KroneckerDelta(j,l)+
            KroneckerDelta(i,l)*NO(Fd(j)*F(k)) +
            NO(Fd(i)*Fd(j)*F(k)*F(l)))
    expr = wicks(F(a)*NO(F(b)*Fd(c))*Fd(d))
    assert (expr ==
           -KroneckerDelta(a,c)*NO(F(b)*Fd(d)) -
            KroneckerDelta(b,d)*NO(F(a)*Fd(c)) -
            KroneckerDelta(a,c)*KroneckerDelta(b,d)+
            KroneckerDelta(a,d)*NO(F(b)*Fd(c)) +
            NO(F(a)*F(b)*Fd(c)*Fd(d)))

def test_NO():
    i,j,k,l = symbols('i j k l',below_fermi=True)
    a,b,c,d = symbols('a b c d',above_fermi=True)
    p,q,r,s = symbols('p q r s', cls=Dummy)

    assert (NO(Fd(p)*F(q) + Fd(a)*F(b))==
       NO(Fd(p)*F(q)) + NO(Fd(a)*F(b)))
    assert (NO(Fd(i)*NO(F(j)*Fd(a))) ==
       NO(Fd(i)*F(j)*Fd(a)))
    assert NO(1) == 1
    assert NO(i) == i
    assert (NO(Fd(a)*Fd(b)*(F(c)+F(d))) ==
               NO(Fd(a)*Fd(b)*F(c)) +
               NO(Fd(a)*Fd(b)*F(d)))

    assert NO(Fd(a)*F(b))._remove_brackets()==Fd(a)*F(b)
    assert NO(F(j)*Fd(i))._remove_brackets()==F(j)*Fd(i)

    assert (NO(Fd(p)*F(q)).subs(Fd(p),Fd(a)+Fd(i)) ==
            NO(Fd(a)*F(q)) + NO(Fd(i)*F(q)))
    assert (NO(Fd(p)*F(q)).subs(F(q),F(a)+F(i)) ==
            NO(Fd(p)*F(a)) + NO(Fd(p)*F(i)))


    expr = NO(Fd(p)*F(q))._remove_brackets()
    assert wicks(expr) == NO(expr)

    assert NO(Fd(a)*F(b)) == - NO(F(b)*Fd(a))

"""def test_sorting():
    i,j = symbols('i,j',below_fermi=True)
    a,b = symbols('a,b',above_fermi=True)
    p,q = symbols('p,q')

    # p, q
    assert _sort_anticommuting_fermions([Fd(p), F(q)]) == ([Fd(p), F(q)], 0)
    assert _sort_anticommuting_fermions([F(p), Fd(q)]) == ([Fd(q), F(p)], 1)

    # i, p
    assert _sort_anticommuting_fermions([F(p), Fd(i)]) == ([F(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), F(p)]) == ([F(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([Fd(p), Fd(i)]) == ([Fd(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(p)]) == ([Fd(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(p), F(i)]) == ([F(i), F(p)], 1)
    assert _sort_anticommuting_fermions([F(i), F(p)]) == ([F(i), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), F(i)]) == ([F(i), Fd(p)], 1)
    assert _sort_anticommuting_fermions([F(i), Fd(p)]) == ([F(i), Fd(p)], 0)

    # a, p
    assert _sort_anticommuting_fermions([F(p), Fd(a)]) == ([Fd(a), F(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), F(p)]) == ([Fd(a), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), Fd(a)]) == ([Fd(a), Fd(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(p)]) == ([Fd(a), Fd(p)], 0)
    assert _sort_anticommuting_fermions([F(p), F(a)]) == ([F(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), F(p)]) == ([F(p), F(a)], 1)
    assert _sort_anticommuting_fermions([Fd(p), F(a)]) == ([Fd(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), Fd(p)]) == ([Fd(p), F(a)], 1)

    # i, a
    assert _sort_anticommuting_fermions([F(i), Fd(j)]) == ([F(i), Fd(j)], 0)
    assert _sort_anticommuting_fermions([Fd(j), F(i)]) == ([F(i), Fd(j)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(i)]) == ([Fd(a), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(a)]) == ([Fd(a), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(a), F(i)]) == ([F(i), F(a)], 1)
    assert _sort_anticommuting_fermions([F(i), F(a)]) == ([F(i), F(a)], 0)"""

def test_contraction():
    i,j,k,l = symbols('i,j,k,l',below_fermi=True)
    a,b,c,d = symbols('a,b,c,d',above_fermi=True)
    p,q,r,s = symbols('p,q,r,s')
    assert contraction(Fd(i),F(j)) == KroneckerDelta(i,j)
    assert contraction(F(a),Fd(b)) == KroneckerDelta(a,b)
    assert contraction(F(a),Fd(i)) == 0
    assert contraction(Fd(a),F(i)) == 0
    assert contraction(F(i),Fd(a)) == 0
    assert contraction(Fd(i),F(a)) == 0
    assert contraction(Fd(i),F(p)) == KroneckerDelta(p,i)
    restr = evaluate_deltas(contraction(Fd(p),F(q)))
    assert restr.is_only_below_fermi
    restr = evaluate_deltas(contraction(F(p),Fd(q)))
    assert restr.is_only_above_fermi

"""def test_Tensors():
    i,j,k,l = symbols('i j k l',below_fermi=True,cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True,cls=Dummy)
    p,q,r,s = symbols('p q r s')

    AT= AntiSymmetricTensor
    assert AT('t',(a,b),(i,j)) == -AT('t',(b,a),(i,j))
    assert AT('t',(a,b),(i,j)) ==  AT('t',(b,a),(j,i))
    assert AT('t',(a,b),(i,j)) == -AT('t',(a,b),(j,i))
    assert AT('t',(a,a),(i,j)) == 0
    assert AT('t',(a,b),(i,i)) == 0
    assert AT('t',(a,b,c),(i,j)) == -AT('t',(b,a,c),(i,j))
    assert AT('t',(a,b,c),(i,j,k)) == AT('t',(b,a,c),(i,k,j))

    tabij = AT('t',(a,b),(i,j))
    assert tabij.has(a)
    assert tabij.has(b)
    assert tabij.has(i)
    assert tabij.has(j)
    assert tabij.subs(b,c) == AT('t',(a,c),(i,j))
    assert (2*tabij).subs(i,c) == 2*AT('t',(a,b),(c,j))

    assert AT('t', (a, a), (i, j)).subs(a, b) == AT('t', (b, b), (i, j))
    assert AT('t', (a, i), (a, j)).subs(a, b) == AT('t', (b, i), (b, j))"""

"""def test_fully_contracted():
    i,j,k,l = symbols('i j k l',below_fermi=True)
    a,b,c,d = symbols('a b c d',above_fermi=True)
    p,q,r,s = symbols('p q r s', cls=Dummy)

    Fock = (AntiSymmetricTensor('f',(p,),(q,))*
            NO(Fd(p)*F(q)))
    V = (AntiSymmetricTensor('v',(p,q),(r,s))*
            NO(Fd(p)*Fd(q)*F(s)*F(r)))/4

    Fai=wicks(NO(Fd(i)*F(a))*Fock,
            keep_only_fully_contracted=True,
            simplify_kronecker_deltas=True)
    assert Fai == AntiSymmetricTensor('f',(a,),(i,))
    Vabij=wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*V,
            keep_only_fully_contracted=True,
            simplify_kronecker_deltas=True)
    assert Vabij==AntiSymmetricTensor('v',(a,b),(i,j))"""

"""def test_substitute_dummies_without_dummies():
    i,j = symbols('i,j')
    assert substitute_dummies(att(i, j) + 2) == att(i, j) + 2
    assert substitute_dummies(att(i, j) + 1) == att(i, j) + 1"""

"""def test_substitute_dummies_NO_operator():
    i,j = symbols('i j', cls=Dummy)
    assert substitute_dummies(att(i, j)*NO(Fd(i)*F(j))
                - att(j, i)*NO(Fd(j)*F(i))) == 0

def test_substitute_dummies_SQ_operator():
    i,j = symbols('i j', cls=Dummy)
    assert substitute_dummies(att(i, j)*Fd(i)*F(j)
                - att(j, i)*Fd(j)*F(i)) == 0

def test_substitute_dummies_new_indices():
    i,j = symbols('i j',below_fermi=True, cls=Dummy)
    a,b = symbols('a b',above_fermi=True, cls=Dummy)
    p,q = symbols('p q', cls=Dummy)
    f = Function('f')
    assert substitute_dummies(f(i,a,p) - f(j,b,q), new_indices=True) == 0

def test_substitute_dummies_substitution_order():
    i,j,k,l = symbols('i j k l',below_fermi=True, cls=Dummy)
    f = Function('f')
    from sympy.utilities.iterables import variations
    for permut in variations([i,j,k,l], 4):
        assert substitute_dummies(f(*permut) - f(i,j,k,l)) == 0

def test_dummy_order_inner_outer_lines_VT1T1T1():
    ii = symbols('i',below_fermi=True)
    aa = symbols('a',above_fermi=True)
    k, l = symbols('k l',below_fermi=True, cls=Dummy)
    c, d = symbols('c d',above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # Coupled-Cluster T1 terms with V*T1*T1*T1
    # t^{a}_{k} t^{c}_{i} t^{d}_{l} v^{lk}_{dc}
    exprs = [
            # permut v and t <=> swapping internal lines, equivalent
            # irrespective of symmetries in v
            v(k, l, c, d)*t(c, ii)*t(d, l)*t(aa, k),
            v(l, k, c, d)*t(c, ii)*t(d, k)*t(aa, l),
            v(k, l, d, c)*t(d, ii)*t(c, l)*t(aa, k),
            v(l, k, d, c)*t(d, ii)*t(c, k)*t(aa, l),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_dummy_order_inner_outer_lines_VT1T1T1T1():
    ii,jj = symbols('i j',below_fermi=True)
    aa,bb = symbols('a b',above_fermi=True)
    k, l = symbols('k l',below_fermi=True, cls=Dummy)
    c, d = symbols('c d',above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    exprs = [
            # permut t <=> swapping external lines, not equivalent
            # except if v has certain symmetries.
            v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            v(k, l, c, d)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),
            v(k, l, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),
            v(k, l, c, d)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [
            # permut v <=> swapping external lines, not equivalent
            # except if v has certain symmetries.
            #
            # Note that in contrast to above, these permutations have identical
            # dummy order.  That is because the proximity to external indices
            # has higher influence on the canonical dummy ordering than the
            # position of a dummy on the factors.  In fact, the terms here are
            # similar in structure as the result of the dummy substitions above.
            v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            v(l, k, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            v(k, l, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            v(l, k, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [
            # permut t and v <=> swapping internal lines, equivalent.
            # Canonical dummy order is different, and a consistent
            # substitution reveals the equivalence.
            v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
            v(k, l, d, c)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),
            v(l, k, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),
            v(l, k, d, c)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_equivalent_internal_lines_VT1T1():
    i,j,k,l = symbols('i j k l',below_fermi=True, cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [ # permute v.  Different dummy order. Not equivalent.
            v(i, j, a, b)*t(a, i)*t(b, j),
            v(j, i, a, b)*t(a, i)*t(b, j),
            v(i, j, b, a)*t(a, i)*t(b, j),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v.  Different dummy order. Equivalent
            v(i, j, a, b)*t(a, i)*t(b, j),
            v(j, i, b, a)*t(a, i)*t(b, j),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    exprs = [ # permute t.  Same dummy order, not equivalent.
            v(i, j, a, b)*t(a, i)*t(b, j),
            v(i, j, a, b)*t(b, i)*t(a, j),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v and t.  Different dummy order, equivalent
            v(i, j, a, b)*t(a, i)*t(b, j),
            v(j, i, a, b)*t(a, j)*t(b, i),
            v(i, j, b, a)*t(b, i)*t(a, j),
            v(j, i, b, a)*t(b, j)*t(a, i),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_equivalent_internal_lines_VT2conjT2():
    # this diagram requires special handling in TCE
    i,j,k,l,m,n = symbols('i j k l m n',below_fermi=True, cls=Dummy)
    a,b,c,d,e,f = symbols('a b c d e f',above_fermi=True, cls=Dummy)
    p1,p2,p3,p4 = symbols('p1 p2 p3 p4',above_fermi=True, cls=Dummy)
    h1,h2,h3,h4 = symbols('h1 h2 h3 h4',below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # v(abcd)t(abij)t(ijcd)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(i, j, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(j, i, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # v(abcd)t(abij)t(jicd)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(j, i, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(i, j, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)

def test_equivalent_internal_lines_VT2conjT2_ambiguous_order():
    # These diagrams invokes _determine_ambiguous() because the
    # dummies can not be ordered unambiguously by the key alone
    i,j,k,l,m,n = symbols('i j k l m n',below_fermi=True, cls=Dummy)
    a,b,c,d,e,f = symbols('a b c d e f',above_fermi=True, cls=Dummy)
    p1,p2,p3,p4 = symbols('p1 p2 p3 p4',above_fermi=True, cls=Dummy)
    h1,h2,h3,h4 = symbols('h1 h2 h3 h4',below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # v(abcd)t(abij)t(cdij)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(p3, p4, i, j)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(p3, p4, i, j)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)

def test_equivalent_internal_lines_VT2():
    i,j,k,l = symbols('i j k l',below_fermi=True, cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies
    exprs = [
            # permute v. Same dummy order, not equivalent.
            #
            # This test show that the dummy order may not be sensitive to all
            # index permutations.  The following expressions have identical
            # structure as the resulting terms from of the dummy subsitutions
            # in the test above.  Here, all expressions have the same dummy
            # order, so they cannot be simplified by means of dummy
            # substitution.  In order to simplify further, it is necessary to
            # exploit symmetries in the objects, for instance if t or v is
            # antisymmetric.
            v(i, j, a, b)*t(a, b, i, j),
            v(j, i, a, b)*t(a, b, i, j),
            v(i, j, b, a)*t(a, b, i, j),
            v(j, i, b, a)*t(a, b, i, j),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [
            # permute t.
            v(i, j, a, b)*t(a, b, i, j),
            v(i, j, a, b)*t(b, a, i, j),
            v(i, j, a, b)*t(a, b, j, i),
            v(i, j, a, b)*t(b, a, j, i),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v and t.  Relabelling of dummies should be equivalent.
            v(i, j, a, b)*t(a, b, i, j),
            v(j, i, a, b)*t(a, b, j, i),
            v(i, j, b, a)*t(b, a, i, j),
            v(j, i, b, a)*t(b, a, j, i),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_internal_external_VT2T2():
    ii, jj = symbols('i j',below_fermi=True)
    aa, bb = symbols('a b',above_fermi=True)
    k, l = symbols('k l'  ,below_fermi=True, cls=Dummy)
    c, d = symbols('c d'  ,above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [
            v(k,l,c,d)*t(aa, c, ii, k)*t(bb, d, jj, l),
            v(l,k,c,d)*t(aa, c, ii, l)*t(bb, d, jj, k),
            v(k,l,d,c)*t(aa, d, ii, k)*t(bb, c, jj, l),
            v(l,k,d,c)*t(aa, d, ii, l)*t(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
            v(k,l,c,d)*t(aa, c, ii, k)*t(d, bb, jj, l),
            v(l,k,c,d)*t(aa, c, ii, l)*t(d, bb, jj, k),
            v(k,l,d,c)*t(aa, d, ii, k)*t(c, bb, jj, l),
            v(l,k,d,c)*t(aa, d, ii, l)*t(c, bb, jj, k),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
            v(k,l,c,d)*t(c, aa, ii, k)*t(bb, d, jj, l),
            v(l,k,c,d)*t(c, aa, ii, l)*t(bb, d, jj, k),
            v(k,l,d,c)*t(d, aa, ii, k)*t(bb, c, jj, l),
            v(l,k,d,c)*t(d, aa, ii, l)*t(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_internal_external_pqrs():
    ii, jj = symbols('i j')
    aa, bb = symbols('a b')
    k, l = symbols('k l'  , cls=Dummy)
    c, d = symbols('c d'  , cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [
            v(k,l,c,d)*t(aa, c, ii, k)*t(bb, d, jj, l),
            v(l,k,c,d)*t(aa, c, ii, l)*t(bb, d, jj, k),
            v(k,l,d,c)*t(aa, d, ii, k)*t(bb, c, jj, l),
            v(l,k,d,c)*t(aa, d, ii, l)*t(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_dummy_order_well_defined():
    aa, bb = symbols('a b', above_fermi=True)
    k, l, m = symbols('k l m', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)

    A = Function('A')
    B = Function('B')
    C = Function('C')
    dums = _get_ordered_dummies

    # We go through all key components in the order of increasing priority,
    # and consider only fully orderable expressions.  Non-orderable expressions
    # are tested elsewhere.

    # pos in first factor determines sort order
    assert dums(A(k, l)*B(l, k)) == [k, l]
    assert dums(A(l, k)*B(l, k)) == [l, k]
    assert dums(A(k, l)*B(k, l)) == [k, l]
    assert dums(A(l, k)*B(k, l)) == [l, k]

    # factors involving the index
    assert dums(A(k, l)*B(l, m)*C(k, m)) == [l, k, m]
    assert dums(A(k, l)*B(l, m)*C(m, k)) == [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(k, m)) == [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(m, k)) == [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(k, m)) == [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(m, k)) == [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(k, m)) == [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(m, k)) == [l, k, m]

    # same, but with factor order determined by non-dummies
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]

    # index range
    assert dums(A(p, c, k)*B(p, c, k)) == [k, c, p]
    assert dums(A(p, k, c)*B(p, c, k)) == [k, c, p]
    assert dums(A(c, k, p)*B(p, c, k)) == [k, c, p]
    assert dums(A(c, p, k)*B(p, c, k)) == [k, c, p]
    assert dums(A(k, c, p)*B(p, c, k)) == [k, c, p]
    assert dums(A(k, p, c)*B(p, c, k)) == [k, c, p]
    assert dums(B(p, c, k)*A(p, c, k)) == [k, c, p]
    assert dums(B(p, k, c)*A(p, c, k)) == [k, c, p]
    assert dums(B(c, k, p)*A(p, c, k)) == [k, c, p]
    assert dums(B(c, p, k)*A(p, c, k)) == [k, c, p]
    assert dums(B(k, c, p)*A(p, c, k)) == [k, c, p]
    assert dums(B(k, p, c)*A(p, c, k)) == [k, c, p]

def test_dummy_order_ambiguous():
    aa, bb = symbols('a b', above_fermi=True)
    i, j, k, l, m = symbols('i j k l m', below_fermi=True, cls=Dummy)
    a, b, c, d, e = symbols('a b c d e', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)
    p1,p2,p3,p4 = symbols('p1 p2 p3 p4',above_fermi=True, cls=Dummy)
    p5,p6,p7,p8 = symbols('p5 p6 p7 p8',above_fermi=True, cls=Dummy)
    h1,h2,h3,h4 = symbols('h1 h2 h3 h4',below_fermi=True, cls=Dummy)
    h5,h6,h7,h8 = symbols('h5 h6 h7 h8',below_fermi=True, cls=Dummy)

    A = Function('A')
    B = Function('B')

    from sympy.utilities.iterables import variations

    # A*A*A*A*B  --  ordering of p5 and p4 is used to figure out the rest
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*B(p5, p4)
    permutator = variations([a,b,c,d,e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # A*A*A*A*A  --  an arbitrary index is assigned and the rest are figured out
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*A(p5, p4)
    permutator = variations([a,b,c,d,e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # A*A*A  --  ordering of p5 and p4 is used to figure out the rest
    template = A(p1, p2, p4, p1)*A(p2, p3, p3, p5)*A(p5, p4)
    permutator = variations([a,b,c,d,e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

def atv(*args):
    return AntiSymmetricTensor('v', args[:2], args[2:] )
def att(*args):
    if len(args) == 4:
        return AntiSymmetricTensor('t', args[:2], args[2:] )
    elif len(args) == 2:
        return AntiSymmetricTensor('t', (args[0],), (args[1],))

def test_dummy_order_inner_outer_lines_VT1T1T1_AT():
    ii = symbols('i',below_fermi=True)
    aa = symbols('a',above_fermi=True)
    k, l = symbols('k l',below_fermi=True, cls=Dummy)
    c, d = symbols('c d',above_fermi=True, cls=Dummy)


    # Coupled-Cluster T1 terms with V*T1*T1*T1
    # t^{a}_{k} t^{c}_{i} t^{d}_{l} v^{lk}_{dc}
    exprs = [
            # permut v and t <=> swapping internal lines, equivalent
            # irrespective of symmetries in v
            atv(k, l, c, d)*att(c, ii)*att(d, l)*att(aa, k),
            atv(l, k, c, d)*att(c, ii)*att(d, k)*att(aa, l),
            atv(k, l, d, c)*att(d, ii)*att(c, l)*att(aa, k),
            atv(l, k, d, c)*att(d, ii)*att(c, k)*att(aa, l),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_dummy_order_inner_outer_lines_VT1T1T1T1_AT():
    ii,jj = symbols('i j',below_fermi=True)
    aa,bb = symbols('a b',above_fermi=True)
    k, l = symbols('k l',below_fermi=True, cls=Dummy)
    c, d = symbols('c d',above_fermi=True, cls=Dummy)


    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    # non-equivalent substitutions (change of sign)
    exprs = [
            # permut t <=> swapping external lines
            atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
            atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(aa, k)*att(bb, l),
            atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(bb, k)*att(aa, l),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == -substitute_dummies(permut)

    # equivalent substitutions
    exprs = [
            atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
            # permut t <=> swapping external lines
            atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(bb, k)*att(aa, l),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_equivalent_internal_lines_VT1T1_AT():
    i,j,k,l = symbols('i j k l',below_fermi=True, cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True, cls=Dummy)


    exprs = [ # permute v.  Different dummy order. Not equivalent.
            atv(i, j, a, b)*att(a, i)*att(b, j),
            atv(j, i, a, b)*att(a, i)*att(b, j),
            atv(i, j, b, a)*att(a, i)*att(b, j),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v.  Different dummy order. Equivalent
            atv(i, j, a, b)*att(a, i)*att(b, j),
            atv(j, i, b, a)*att(a, i)*att(b, j),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    exprs = [ # permute t.  Same dummy order, not equivalent.
            atv(i, j, a, b)*att(a, i)*att(b, j),
            atv(i, j, a, b)*att(b, i)*att(a, j),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v and t.  Different dummy order, equivalent
            atv(i, j, a, b)*att(a, i)*att(b, j),
            atv(j, i, a, b)*att(a, j)*att(b, i),
            atv(i, j, b, a)*att(b, i)*att(a, j),
            atv(j, i, b, a)*att(b, j)*att(a, i),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_equivalent_internal_lines_VT2conjT2_AT():
    # this diagram requires special handling in TCE
    i,j,k,l,m,n = symbols('i j k l m n',below_fermi=True, cls=Dummy)
    a,b,c,d,e,f = symbols('a b c d e f',above_fermi=True, cls=Dummy)
    p1,p2,p3,p4 = symbols('p1 p2 p3 p4',above_fermi=True, cls=Dummy)
    h1,h2,h3,h4 = symbols('h1 h2 h3 h4',below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations


    # atv(abcd)att(abij)att(ijcd)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(i, j, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(j, i, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # atv(abcd)att(abij)att(jicd)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(j, i, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(i, j, p3, p4)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

def test_equivalent_internal_lines_VT2conjT2_ambiguous_order_AT():
    # These diagrams invokes _determine_ambiguous() because the
    # dummies can not be ordered unambiguously by the key alone
    i,j,k,l,m,n = symbols('i j k l m n',below_fermi=True, cls=Dummy)
    a,b,c,d,e,f = symbols('a b c d e f',above_fermi=True, cls=Dummy)
    p1,p2,p3,p4 = symbols('p1 p2 p3 p4',above_fermi=True, cls=Dummy)
    h1,h2,h3,h4 = symbols('h1 h2 h3 h4',below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations


    # atv(abcd)att(abij)att(cdij)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(p3, p4, i, j)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(p3, p4, i, j)
    permutator = variations([a,b,c,d], 4)
    base = template.subs(zip([p1, p2, p3, p4], permutator.next()))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

def test_equivalent_internal_lines_VT2_AT():
    i,j,k,l = symbols('i j k l',below_fermi=True, cls=Dummy)
    a,b,c,d = symbols('a b c d',above_fermi=True, cls=Dummy)

    exprs = [
            # permute v. Same dummy order, not equivalent.
            atv(i, j, a, b)*att(a, b, i, j),
            atv(j, i, a, b)*att(a, b, i, j),
            atv(i, j, b, a)*att(a, b, i, j),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [
            # permute t.
            atv(i, j, a, b)*att(a, b, i, j),
            atv(i, j, a, b)*att(b, a, i, j),
            atv(i, j, a, b)*att(a, b, j, i),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [ # permute v and t.  Relabelling of dummies should be equivalent.
            atv(i, j, a, b)*att(a, b, i, j),
            atv(j, i, a, b)*att(a, b, j, i),
            atv(i, j, b, a)*att(b, a, i, j),
            atv(j, i, b, a)*att(b, a, j, i),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_internal_external_VT2T2_AT():
    ii, jj = symbols('i j',below_fermi=True)
    aa, bb = symbols('a b',above_fermi=True)
    k, l = symbols('k l'  ,below_fermi=True, cls=Dummy)
    c, d = symbols('c d'  ,above_fermi=True, cls=Dummy)

    exprs = [
            atv(k,l,c,d)*att(aa, c, ii, k)*att(bb, d, jj, l),
            atv(l,k,c,d)*att(aa, c, ii, l)*att(bb, d, jj, k),
            atv(k,l,d,c)*att(aa, d, ii, k)*att(bb, c, jj, l),
            atv(l,k,d,c)*att(aa, d, ii, l)*att(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
            atv(k,l,c,d)*att(aa, c, ii, k)*att(d, bb, jj, l),
            atv(l,k,c,d)*att(aa, c, ii, l)*att(d, bb, jj, k),
            atv(k,l,d,c)*att(aa, d, ii, k)*att(c, bb, jj, l),
            atv(l,k,d,c)*att(aa, d, ii, l)*att(c, bb, jj, k),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
            atv(k,l,c,d)*att(c, aa, ii, k)*att(bb, d, jj, l),
            atv(l,k,c,d)*att(c, aa, ii, l)*att(bb, d, jj, k),
            atv(k,l,d,c)*att(d, aa, ii, k)*att(bb, c, jj, l),
            atv(l,k,d,c)*att(d, aa, ii, l)*att(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

def test_internal_external_pqrs_AT():
    ii, jj = symbols('i j')
    aa, bb = symbols('a b')
    k, l = symbols('k l'  , cls=Dummy)
    c, d = symbols('c d'  , cls=Dummy)


    exprs = [
            atv(k,l,c,d)*att(aa, c, ii, k)*att(bb, d, jj, l),
            atv(l,k,c,d)*att(aa, c, ii, l)*att(bb, d, jj, k),
            atv(k,l,d,c)*att(aa, d, ii, k)*att(bb, c, jj, l),
            atv(l,k,d,c)*att(aa, d, ii, l)*att(bb, c, jj, k),
            ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)"""
