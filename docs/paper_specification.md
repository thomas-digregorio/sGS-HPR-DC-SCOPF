# Paper specification

## Scope and status

This document is the implementation-facing specification extracted from:

> Qi Wang, Guojun Zhang, Yue Yang, Chao Ren, Wenchuan Wu, Xinyuan Zhao,
> Mikael Skoglund, and Defeng Sun, "An Efficient GPU-based Halpern
> Accelerating Algorithm for Large-scale DC Optimal Power Flow."

The source is a 17-page prepublication manuscript whose header still reads
"JOURNAL OF LATEX CLASS FILES, VOL. XX, NO. XX, XX XXXX." All 17 rendered
pages were reviewed, and equations were cross-checked against layout-preserving
PDF text extraction. Equations (1)-(55), Algorithms 1-2, and Tables I-IV are
indexed below.

The formulation in Section II is a multi-period, base-case, shift-factor
DCOPF. Despite the paper's occasional use of "security-constrained DCOPF," the
published model contains no contingency index, outage model, LODF, or explicit
post-contingency constraint. It must not be represented as an N-1 SCOPF model.

## 1. Notation, dimensions, and stacking

### 1.1 Index sets and cardinalities

| Paper symbol | Meaning | Cardinality |
|---|---|---:|
| \(\mathbb G\) | conventional generators | \(N_G\) |
| \(\mathbb{RG}\) | renewable generators | \(N_{RG}\) |
| \(\mathbb E\) | energy storage systems (ESSs) | \(N_{ESS}\) |
| \(\mathbb D\) | loads | not named |
| \(\mathbb L\) | branches/lines | \(N_L\) |
| \(\mathbb T\) | time periods | \(T\) |

Table I additionally reports the bus count, but the paper does not introduce a
bus-count symbol in the mathematical model.

The paper uses \(A^*\) for transpose/adjoint, \(\langle\cdot,\cdot\rangle\)
for the Euclidean inner product, \(\|\cdot\|\) for its induced norm,
\(\lambda_1(AA^*)\) for the largest eigenvalue, and \(\otimes\) for the
Kronecker product.

### 1.2 Decision variables

The following stacking is implied by Eq. (55), although the paper never writes
the full vector \(x\) explicitly:

\[
x =
\left(
  p_G,\ p_{RG},\ p_{ESS}^{dc},\ p_{ESS}^{ch},\ r^u,\ r^d
\right).
\]

Each block is stacked over all periods and its associated devices.

| Variable | Meaning and sign convention | Dimension |
|---|---|---:|
| \(p_{G,i}(t)\) | conventional generation injected into the grid | \(TN_G\) |
| \(p_{RG,i}(t)\) | renewable generation injected into the grid | \(TN_{RG}\) |
| \(p_{ESS,i}^{dc}(t)\) | ESS discharge, a positive grid injection | \(TN_{ESS}\) |
| \(p_{ESS,i}^{ch}(t)\) | ESS charge, a positive grid withdrawal | \(TN_{ESS}\) |
| \(r_i^u(t)\) | upward spinning reserve | \(TN_G\) |
| \(r_i^d(t)\) | downward spinning reserve | \(TN_G\) |

Therefore,

\[
\boxed{n=T\left(3N_G+N_{RG}+2N_{ESS}\right)}.
\]

The ESS energy state is not introduced as a separate variable. It is an affine
cumulative sum of charge and discharge decisions in Eq. (8).

### 1.3 Important parameters

| Symbol | Meaning |
|---|---|
| \(P_{D,i}(t)\) | active load at bus \(i\), treated as fixed data |
| \(SF_{j-i}\) | shift factor from a net injection at bus \(i\) to line \(j\) |
| \(\bar P_{L,j}\) | symmetric line-flow magnitude limit |
| \(\underline P_{G,i},\bar P_{G,i}\) | minimum and maximum conventional output |
| \(\underline P_{RG,i},\bar P_{RG,i}\) | minimum and maximum renewable output |
| \(RU_i,RD_i\) | upward and downward generator ramp rates |
| \(SRU(t),SRD(t)\) | system-wide upward and downward reserve requirements |
| \(\Delta t\) | duration of one optimization interval |
| \(E_{ESS,i}(0)\) | initial ESS energy |
| \(\underline E_{ESS,i},\bar E_{ESS,i}\) | ESS energy bounds |
| \(\eta_{ESS,i}^{ch},\eta_{ESS,i}^{dc}\) | charge and discharge efficiencies |
| \(\bar P_{ESS,i}^{ch},\bar P_{ESS,i}^{dc}\) | charge and discharge power limits |
| \(a_{1,i},a_{0,i}\) | linear and constant conventional generation-cost terms |
| \(\sigma_{RG},\sigma_{ESS}\) | renewable-curtailment and ESS-loss penalties |

The same \(RU_i,RD_i\) parameters are used in reserve capacity Eq. (3) and
inter-period ramping Eq. (6).

## 2. DCOPF formulation: Eqs. (1)-(16)

### 2.1 Operational constraints: Eqs. (1)-(10)

**Eq. (1), system-wide power balance, \(T\) equalities**

\[
\sum_{i\in\mathbb G}p_{G,i}(t)
+\sum_{i\in\mathbb{RG}}p_{RG,i}(t)
+\sum_{i\in\mathbb E}
  \left(p_{ESS,i}^{dc}(t)-p_{ESS,i}^{ch}(t)\right)
=\sum_{i\in\mathbb D}P_{D,i}(t),
\quad \forall t\in\mathbb T.
\tag{1}
\]

Generation and discharge have positive balance coefficients; charging and
load are withdrawals. These rows form the first \(T\) rows of \(A_1x=b_1\).

**Eq. (2), base-case shift-factor line limits, \(2TN_L\) scalar
inequalities after splitting**

\[
\begin{aligned}
-\bar P_{L,j}\le{}&
\sum_{i\in\mathbb G}SF_{j-i}p_{G,i}(t)
+\sum_{i\in\mathbb{RG}}SF_{j-i}p_{RG,i}(t)\\
&+\sum_{i\in\mathbb E}SF_{j-i}
  \left(p_{ESS,i}^{dc}(t)-p_{ESS,i}^{ch}(t)\right)
-\sum_{i\in\mathbb D}SF_{j-i}P_{D,i}(t)
\le\bar P_{L,j},\\
&\hspace{7cm}\forall j\in\mathbb L,\ \forall t\in\mathbb T .
\end{aligned}
\tag{2}
\]

This is a base-case two-sided thermal constraint. It contributes two rows per
line-period pair to \(A_2x\ge b_2\).

**Eq. (3), reserve box bounds**

\[
0\le r_i^u(t)\le RU_i\Delta t,\qquad
0\le r_i^d(t)\le RD_i\Delta t .
\tag{3}
\]

These are variable bounds in \(C\), not rows of \(A_2\).

**Eq. (4), reserve headroom/footroom, \(2TN_G\) inequalities**

\[
r_i^u(t)\le\bar P_{G,i}-p_{G,i}(t),\qquad
r_i^d(t)\le p_{G,i}(t)-\underline P_{G,i}.
\tag{4}
\]

**Eq. (5), system reserve requirements, \(2T\) inequalities**

\[
\sum_{i\in\mathbb G}r_i^u(t)\ge SRU(t),\qquad
\sum_{i\in\mathbb G}r_i^d(t)\ge SRD(t),
\quad\forall t\in\mathbb T .
\tag{5}
\]

**Eq. (6), generator ramping, \(2(T-1)N_G\) scalar inequalities**

\[
-RD_i\Delta t
\le p_{G,i}(t)-p_{G,i}(t-1)
\le RU_i\Delta t,
\quad \forall t\in\mathbb T\setminus\{1\}.
\tag{6}
\]

**Eq. (7), renewable output bounds**

\[
\underline P_{RG,i}\le p_{RG,i}(t)\le\bar P_{RG,i},
\quad\forall i\in\mathbb{RG},\ \forall t\in\mathbb T .
\tag{7}
\]

These are variable bounds in \(C\).

**Eq. (8), cumulative ESS energy bounds, \(2TN_{ESS}\) scalar
inequalities**

\[
\begin{aligned}
\underline E_{ESS,i}\le E_{ESS,i}(0)
+\sum_{\tau=1}^{t}
\left(
 p_{ESS,i}^{ch}(\tau)\eta_{ESS,i}^{ch}
-\frac{p_{ESS,i}^{dc}(\tau)}{\eta_{ESS,i}^{dc}}
\right)\Delta t
\le\bar E_{ESS,i},\\
\forall i\in\mathbb E,\ \forall t\in\mathbb T .
\end{aligned}
\tag{8}
\]

Charging increases stored energy by \(\eta^{ch}p^{ch}\Delta t\);
discharging reduces it by \(p^{dc}\Delta t/\eta^{dc}\). These two-sided
constraints are in \(A_2x\ge b_2\).

**Eq. (9), terminal ESS energy equality, \(N_{ESS}\) equalities**

\[
\sum_{\tau=1}^{T}
\left(
 p_{ESS,i}^{ch}(\tau)\eta_{ESS,i}^{ch}
-\frac{p_{ESS,i}^{dc}(\tau)}{\eta_{ESS,i}^{dc}}
\right)\Delta t=0 .
\tag{9}
\]

Thus terminal energy equals initial energy. These rows form the last
\(N_{ESS}\) rows of \(A_1x=b_1\).

**Eq. (10), ESS charge/discharge power bounds**

\[
0\le p_{ESS,i}^{ch}(t)\le\bar P_{ESS,i}^{ch},\qquad
0\le p_{ESS,i}^{dc}(t)\le\bar P_{ESS,i}^{dc}.
\tag{10}
\]

These are variable bounds in \(C\). The model has no binary variable or
constraint preventing simultaneous charge and discharge.

### 2.2 Objective: Eqs. (11)-(14)

**Eq. (11), total objective**

\[
\min\left\{
\sum_{t\in\mathbb T}
\left[
\sum_{i\in\mathbb G}C_i(p_{G,i}(t))
+\sum_{i\in\mathbb{RG}}C_i(p_{RG,i}(t))
+\sum_{i\in\mathbb E}
 C_i(p_{ESS,i}^{dc}(t),p_{ESS,i}^{ch}(t))
\right]\right\}.
\tag{11}
\]

**Eq. (12), linear conventional generation cost**

\[
C_i(p_{G,i}(t))=a_{1,i}p_{G,i}(t)+a_{0,i}.
\tag{12}
\]

**Eq. (13), renewable-curtailment penalty**

\[
C_i(p_{RG,i}(t))
=\sigma_{RG}\left(\bar P_{RG,i}-p_{RG,i}(t)\right).
\tag{13}
\]

**Eq. (14), ESS conversion-loss penalty**

\[
\begin{aligned}
C_i(p_{ESS,i}^{dc}(t),p_{ESS,i}^{ch}(t))
=\sigma_{ESS}\bigg[
 p_{ESS,i}^{dc}(t)
 \left(\frac{1}{\eta_{ESS,i}^{dc}}-1\right)
+p_{ESS,i}^{ch}(t)
 \left(1-\eta_{ESS,i}^{ch}\right)
\bigg].
\end{aligned}
\tag{14}
\]

For the inferred variable order, the linear coefficient vector has blocks

\[
c=\left(
 a_1,\ -\sigma_{RG}\mathbf 1,\
 \sigma_{ESS}(\eta_{dc}^{-1}-\mathbf1),\
 \sigma_{ESS}(\mathbf1-\eta_{ch}),\
 0,\ 0
\right),
\]

repeated over periods and devices as appropriate. The terms
\(\sum_{t,i\in\mathbb G}a_{0,i}\) and
\(\sum_{t,i\in\mathbb{RG}}\sigma_{RG}\bar P_{RG,i}\) are objective
constants. They do not affect the optimizer and therefore are absent from
\(\langle c,x\rangle\), but they must be restored if reproducing absolute
objective values.

### 2.3 Canonical primal and dual: Eqs. (15)-(16)

**Eq. (15), primal LP**

\[
\begin{aligned}
\min_{x\in\mathbb R^n}\quad &\langle c,x\rangle\\
\text{s.t.}\quad&A_1x=b_1,\\
&A_2x\ge b_2,\\
&x\in C,
\end{aligned}
\qquad
C=\{x\in\mathbb R^n\mid l\le x\le u\}.
\tag{15}
\]

Dimensions are

\[
A_1\in\mathbb R^{m_1\times n},\quad
A_2\in\mathbb R^{m_2\times n},\quad
A=\begin{bmatrix}A_1\\A_2\end{bmatrix}\in\mathbb R^{m\times n},
\quad m=m_1+m_2,
\]

\[
b_1\in\mathbb R^{m_1},\quad b_2\in\mathbb R^{m_2},\quad
b=\begin{bmatrix}b_1\\b_2\end{bmatrix},\quad c,l,u\in\mathbb R^n .
\]

**Eq. (16), dual LP**

\[
\begin{aligned}
\min_{y\in\mathbb R^m,\ z\in\mathbb R^n}\quad&
-\langle b,y\rangle+\delta_D(y)+\delta_C^*(-z)\\
\text{s.t.}\quad&A^*y+z=c,
\end{aligned}
\tag{16}
\]

where

\[
D=\mathbb R^{m_1}\times\mathbb R_+^{m_2},\qquad
y=(y_1,y_2),\quad
y_1\in\mathbb R^{m_1},\quad y_2\in\mathbb R_+^{m_2}.
\]

Thus equality multipliers are free, inequality multipliers are nonnegative,
and the paper's canonical inequality direction is always \(A_2x\ge b_2\).

### 2.4 Exact \(A_1,A_2,C,b_1,b_2,c\) assignment

The equality right-hand side is

\[
b_1=
\begin{bmatrix}
(\sum_{i\in\mathbb D}P_{D,i}(t))_{t=1}^{T}\\
0_{N_{ESS}}
\end{bmatrix},
\qquad m_1=T+N_{ESS}.
\]

For a uniform \(A_2x\ge b_2\) implementation, split every double inequality.
Let

\[
h_{j,t}(x)=
\sum_{i\in\mathbb G}SF_{j-i}p_{G,i}(t)
+\sum_{i\in\mathbb{RG}}SF_{j-i}p_{RG,i}(t)
+\sum_{i\in\mathbb E}SF_{j-i}
  (p_{ESS,i}^{dc}(t)-p_{ESS,i}^{ch}(t))
\]

and \(d^{SF}_{j,t}=\sum_{i\in\mathbb D}SF_{j-i}P_{D,i}(t)\).
The canonical row mapping is:

| Source | \(A_2x\ge b_2\) rows | Row count |
|---|---|---:|
| Eq. (2) | \(h_{j,t}(x)\ge d^{SF}_{j,t}-\bar P_{L,j}\); \(-h_{j,t}(x)\ge-d^{SF}_{j,t}-\bar P_{L,j}\) | \(2TN_L\) |
| Eq. (4), up | \(-p_{G,i}(t)-r_i^u(t)\ge-\bar P_{G,i}\) | \(TN_G\) |
| Eq. (4), down | \(p_{G,i}(t)-r_i^d(t)\ge\underline P_{G,i}\) | \(TN_G\) |
| Eq. (5) | \(\sum_i r_i^u(t)\ge SRU(t)\); \(\sum_i r_i^d(t)\ge SRD(t)\) | \(2T\) |
| Eq. (6) | \(\Delta p_{G,i}(t)\ge-RD_i\Delta t\); \(-\Delta p_{G,i}(t)\ge-RU_i\Delta t\) | \(2(T-1)N_G\) |
| Eq. (8) | \(q_{i,t}(x)\ge\underline E_{ESS,i}-E_{ESS,i}(0)\); \(-q_{i,t}(x)\ge E_{ESS,i}(0)-\bar E_{ESS,i}\) | \(2TN_{ESS}\) |

Here

\[
\Delta p_{G,i}(t)=p_{G,i}(t)-p_{G,i}(t-1),\qquad
q_{i,t}(x)=\sum_{\tau=1}^{t}
\left(\eta_i^{ch}p_i^{ch}(\tau)-p_i^{dc}(\tau)/\eta_i^{dc}\right)\Delta t .
\]

Equations (3), (7), and (10), together with the conventional-output bounds
used in Eq. (4), define \(l,u\) and hence \(C\). Equations (1) and (9)
define \(A_1,b_1\). Equations (2), (4)-(6), and (8) define \(A_2,b_2\).
Equations (11)-(14) define \(c\) plus the objective constant.

## 3. HPR and sGS-HPR: Eqs. (17)-(55)

### 3.1 Augmented Lagrangian and Algorithm 1: Eq. (17)

For \(w=(y,z,x)\in\mathcal W=\mathbb R^m\times\mathbb R^n\times
\mathbb R^n\), Eq. (17) is the augmented Lagrangian of dual Eq. (16):

\[
\begin{aligned}
L_\sigma(y,z;x)
:={}&-\langle b,y\rangle+\delta_D(y)+\delta_C^*(-z)
+\langle x,A^*y+z-c\rangle\\
&+\frac{\sigma}{2}\|A^*y+z-c\|^2,\qquad \sigma>0 .
\end{aligned}
\tag{17}
\]

**Algorithm 1: HPR for Eq. (16)**

1. Choose \(\sigma>0\), a self-adjoint positive-semidefinite
   \(\mathcal T_1:\mathbb R^m\to\mathbb R^m\) such that
   \(\mathcal T_1+AA^*\succ0\), and
   \(w^0=(y^0,z^0,x^0)\). Write
   \(\bar w^k=(\bar y^k,\bar z^k,\bar x^k)\).
2. For \(k=0,1,\ldots\):
   1. \(\bar z^{k+1}=\arg\min_zL_\sigma(y^k,z;x^k)\).
   2. \(\bar x^{k+1}=x^k+\sigma(A^*y^k+\bar z^{k+1}-c)\).
   3. \[
      \bar y^{k+1}=\arg\min_y
      \left\{
      L_\sigma(y,\bar z^{k+1};\bar x^{k+1})
      +\frac{\sigma}{2}\|y-y^k\|_{\mathcal T_1}^2
      \right\}.
      \]
   4. Reflect: \(\hat w^{k+1}=2\bar w^{k+1}-w^k\).
   5. Halpern update:
      \[
      w^{k+1}=\frac{1}{k+2}w^0+\frac{k+1}{k+2}\hat w^{k+1}.
      \]

The anchor \(w^0\) remains fixed throughout.

### 3.2 sGS decomposition and Algorithm 2: Eqs. (18)-(23)

**Eq. (18), block operator**

\[
\mathcal H=
\begin{bmatrix}
A_1A_1^*&A_1A_2^*\\
A_2A_1^*&A_2A_2^*+\mathcal S_2
\end{bmatrix},
\quad
\mathcal S_2:\mathbb R^{m_2}\to\mathbb R^{m_2}\succeq0,
\quad A_2A_2^*+\mathcal S_2\succ0 .
\tag{18}
\]

**Eq. (19), symmetric Gauss-Seidel splitting**

\[
\mathcal H=\mathcal U_{\mathcal H}^*+
\mathcal D_{\mathcal H}+\mathcal U_{\mathcal H},
\tag{19}
\]

\[
\mathcal U_{\mathcal H}=
\begin{bmatrix}0&A_1A_2^*\\0&0\end{bmatrix},
\qquad
\mathcal D_{\mathcal H}=
\begin{bmatrix}A_1A_1^*&0\\0&A_2A_2^*+\mathcal S_2\end{bmatrix}.
\]

**Eq. (20), sGS operator**

\[
\operatorname{sGS}(\mathcal H)
=\mathcal U_{\mathcal H}^*\mathcal D_{\mathcal H}^{-1}
\mathcal U_{\mathcal H}.
\tag{20}
\]

**Eq. (21), paper's semi-proximal operator**

\[
\mathcal T_1=\operatorname{sGS}(\mathcal H)
+\operatorname{diag}(0_{m_1},\mathcal S_2).
\tag{21}
\]

**Eq. (22), exact three-sweep \(y\) solution**

\[
\left\{
\begin{aligned}
\bar y_1^{k+1/2}
&=\arg\min_{y_1\in\mathbb R^{m_1}}
L_\sigma(y_1,y_2^k,\bar z^{k+1};\bar x^{k+1}),\\
\bar y_2^{k+1}
&=\arg\min_{y_2\in\mathbb R^{m_2}}
\left\{
L_\sigma(\bar y_1^{k+1/2},y_2,\bar z^{k+1};\bar x^{k+1})
+\frac{\sigma}{2}\|y_2-y_2^k\|_{\mathcal S_2}^2
\right\},\\
\bar y_1^{k+1}
&=\arg\min_{y_1\in\mathbb R^{m_1}}
L_\sigma(y_1,\bar y_2^{k+1},\bar z^{k+1};\bar x^{k+1}).
\end{aligned}
\right.
\tag{22}
\]

**Eq. (23), definiteness consequence**

\[
\mathcal T_1+AA^*\succ0.
\tag{23}
\]

**Algorithm 2: sGS-HPR for the DCOPF**

1. Choose \(\sigma>0\) and
   \(w^0=(y^0,z^0,x^0)\in\mathbb R^m\times\mathbb R^n\times\mathbb R^n\).
2. For \(k=0,1,\ldots\):
   1. \(\bar z^{k+1}=\arg\min_z
      L_\sigma(y_1^k,y_2^k,z;x^k)\).
   2. \[
      \bar x^{k+1}=x^k+\sigma
      (A_1^*y_1^k+A_2^*y_2^k+\bar z^{k+1}-c).
      \]
   3. Perform the Eq. (22) sweeps in this exact order:
      \(\bar y_1^{k+1/2}\), then \(\bar y_2^{k+1}\), then
      \(\bar y_1^{k+1}\).
   4. Set
      \(\bar w^{k+1}=(\bar y_1^{k+1},\bar y_2^{k+1},
      \bar z^{k+1},\bar x^{k+1})\) and
      \(\hat w^{k+1}=2\bar w^{k+1}-w^k\).
   5. Set
      \(w^{k+1}=(k+2)^{-1}w^0+(k+1)(k+2)^{-1}\hat w^{k+1}\).

The two \(y_1\) solves are both required; the first uses \(y_2^k\), and the
second uses the newly projected \(\bar y_2^{k+1}\).

### 3.3 KKT system, monotone operator, and convergence: Eqs. (24)-(30)

**Eq. (24), KKT system**

\[
0\in Ax^*-b+\mathcal N_D(y^*),\qquad
0\in z^*+\mathcal N_C(x^*),\qquad
A^*y^*+z^*-c=0.
\tag{24}
\]

Assumption 1 is existence of a triple \((y^*,z^*,x^*)\) satisfying Eq. (24).

**Eq. (25), maximal monotone operator**

\[
\mathcal T w=
\begin{pmatrix}
-b+\mathcal N_D(y)+Ax\\
-\partial\delta_C^*(-z)+x\\
c-A^*y-z
\end{pmatrix},
\quad
w=(y,z,x)\in\mathbb R^m\times\mathbb R^n\times\mathbb R^n.
\tag{25}
\]

**Eq. (26), degenerate preconditioner**

\[
\mathcal M=
\begin{bmatrix}
\sigma(AA^*+\mathcal T_1)&0&A\\
0&0&0\\
A^*&0&\sigma^{-1}I_n
\end{bmatrix}.
\tag{26}
\]

Thus \(\mathcal M\) acts on \(\mathbb R^{m+2n}\).

**Eq. (27), accelerated degenerate proximal-point form**

\[
\bar w^{k+1}\in(\mathcal M+\mathcal T)^{-1}\mathcal Mw^k,\qquad
\hat w^{k+1}=2\bar w^{k+1}-w^k,\qquad
w^{k+1}=\frac{1}{k+2}w^0+\frac{k+1}{k+2}\hat w^{k+1}.
\tag{27}
\]

**Eq. (28), KKT residual mapping**

\[
\mathcal R(w)=
\begin{pmatrix}
y-\Pi_D(y-Ax+b)\\
x-\Pi_C(x-z)\\
c-A^*y-z
\end{pmatrix}
\in\mathbb R^{m+2n}.
\tag{28}
\]

\(\mathcal R(w)=0\) characterizes a primal-dual solution. The first block
combines equality and inequality feasibility, the second is the box
normal-cone condition, and the third is stationarity.

Let \(R_0=\|w^0-w^*\|_{\mathcal M}\), where \(w^*\in\mathcal T^{-1}(0)\).

**Eq. (29), Halpern fixed-point residual bound**

\[
\|w^k-\hat w^{k+1}\|_{\mathcal M}
\le\frac{2R_0}{k+1},\qquad k\ge0.
\tag{29}
\]

**Eq. (30), non-ergodic KKT residual bound**

\[
\|\mathcal R(\bar w^{k+1})\|
\le
\left(
\frac{\sigma(\|A^*\|+\|\sqrt{\mathcal T_1}\|)+1}{\sqrt\sigma}
\right)
\frac{R_0}{k+1}.
\tag{30}
\]

The paper therefore claims an \(O(1/k)\) non-ergodic KKT-residual rate.

### 3.4 Closed-form updates: Eqs. (31)-(50)

#### \(z\) and \(x\): Eqs. (31)-(34)

**Eq. (31)** expands the \(z\) subproblem:

\[
\bar z^{k+1}=\arg\min_z
\left\{
\delta_C^*(-z)+\langle x^k,A^*y^k+z-c\rangle
+\frac{\sigma}{2}\|A^*y^k+z-c\|^2
\right\}.
\tag{31}
\]

**Eq. (32)** is its optimality condition:

\[
0\in x^k+\sigma(A^*y^k+z-c)+\partial\delta_C^*(-z).
\tag{32}
\]

**Eq. (33)** gives

\[
\bar z^{k+1}
=\frac{1}{\sigma}
\left\{
\Pi_C[x^k+\sigma(A^*y^k-c)]
-[x^k+\sigma(A^*y^k-c)]
\right\}.
\tag{33}
\]

**Eq. (34)** gives

\[
\bar x^{k+1}
=x^k+\sigma(A^*y^k+\bar z^{k+1}-c).
\tag{34}
\]

Combining Eqs. (33)-(34) yields the useful identity

\[
\boxed{\bar x^{k+1}=\Pi_C[x^k+\sigma(A^*y^k-c)]}.
\]

#### \(y_1\): Eqs. (35)-(46)

**Eq. (35)** expands either equality-multiplier sweep (shown for the first):

\[
\begin{aligned}
\bar y_1^{k+1/2}
=\arg\min_{y_1}\bigg\{
-\langle b_1,y_1\rangle+\delta_{\mathbb R^{m_1}}(y_1)
+\langle\bar x^{k+1},A_1^*y_1\rangle\\
+\frac{\sigma}{2}
\|A_1^*y_1+A_2^*y_2^k+\bar z^{k+1}-c\|^2
\bigg\}.
\end{aligned}
\tag{35}
\]

**Eq. (36)** reduces the update to

\[
A_1A_1^*y_1=R_1,\qquad
R_1=\frac{1}{\sigma}
\left[
b_1-A_1\left(
\bar x^{k+1}
+\sigma(A_2^*y_2^k+\bar z^{k+1}-c)
\right)
\right].
\tag{36}
\]

For the second \(y_1\) sweep, replace \(y_2^k\) by
\(\bar y_2^{k+1}\).

**Eq. (37)** states the general block form

\[
A_1A_1^*=
\begin{bmatrix}
D_1&\mathbf1_{m_{11}}\otimes d^T\\
\mathbf1_{m_{11}}^T\otimes d&D_2
\end{bmatrix},
\tag{37}
\]

with \(m_{11}+m_{12}=m_1\),
\(D_1\in\mathbb R^{m_{11}\times m_{11}}\) and
\(D_2\in\mathbb R^{m_{12}\times m_{12}}\) diagonal, and
\(d\in\mathbb R^{m_{12}}\). For Eq. (55),
\(m_{11}=T\) and \(m_{12}=N_{ESS}\).

Writing \(y_1=(y_{11},y_{12})\) and \(R_1=(R_{11},R_{12})\),
**Eq. (38)** is

\[
\begin{bmatrix}
D_1&\mathbf1_{m_{11}}\otimes d^T\\
\mathbf1_{m_{11}}^T\otimes d&D_2
\end{bmatrix}
\begin{bmatrix}y_{11}\\y_{12}\end{bmatrix}
=
\begin{bmatrix}R_{11}\\R_{12}\end{bmatrix}.
\tag{38}
\]

**Eq. (39)** gives the paper's stated solution:

\[
\begin{aligned}
y_{11}
&=\widetilde y_{11}
-\frac{
D_1^{-1}\mathbf1_{m_{11}}\mathbf1_{m_{11}}^T\widetilde y_{11}
}{
(d^TD_2^{-1}d)^{-1}
+\mathbf1_{m_{11}}^TD_1^{-1}\mathbf1_{m_{11}}
},\\
y_{12}
&=D_2^{-1}
\left(R_{12}-(\mathbf1_{m_{11}}^T\otimes d)y_{11}\right).
\end{aligned}
\tag{39}
\]

**Eq. (40)** defines

\[
\widetilde y_{11}
=D_1^{-1}
\left[
R_{11}
-\left((\mathbf1_{m_{11}}\otimes d^T)D_2^{-1}\right)R_{12}
\right].
\tag{40}
\]

**Eq. (41)** is the Schur-complement elimination:

\[
\left[
D_1-(\mathbf1_{m_{11}}\otimes d^T)D_2^{-1}
(\mathbf1_{m_{11}}^T\otimes d)
\right]y_{11}
=\widetilde R_{11},
\tag{41}
\]

\[
\widetilde R_{11}
=R_{11}-(\mathbf1_{m_{11}}\otimes d^T)D_2^{-1}R_{12}.
\]

**Eq. (42)** uses the Kronecker structure:

\[
(\mathbf1_{m_{11}}\otimes d^T)D_2^{-1}
(\mathbf1_{m_{11}}^T\otimes d)
=
(d^TD_2^{-1}d)\,
\mathbf1_{m_{11}}\mathbf1_{m_{11}}^T
=\alpha\mathbf1_{m_{11}}\mathbf1_{m_{11}}^T,
\tag{42}
\]

where \(\alpha=d^TD_2^{-1}d\).

**Eq. (43)** becomes

\[
(D_1-\alpha\mathbf1_{m_{11}}\mathbf1_{m_{11}}^T)y_{11}
=\widetilde R_{11}.
\tag{43}
\]

**Eq. (44)** states

\[
\begin{aligned}
(D_1-\alpha\mathbf1\mathbf1^T)^{-1}
=D_1^{-1}
-\frac{
D_1^{-1}\mathbf1\mathbf1^TD_1^{-1}
}{
\alpha^{-1}+\mathbf1^TD_1^{-1}\mathbf1
}.
\end{aligned}
\tag{44}
\]

**Eq. (45)** applies Eq. (44) to \(\widetilde R_{11}\):

\[
\begin{aligned}
y_{11}
={}&
\left(
D_1^{-1}
-\frac{
D_1^{-1}\mathbf1_{m_{11}}\mathbf1_{m_{11}}^TD_1^{-1}
}{
(d^TD_2^{-1}d)^{-1}
+\mathbf1_{m_{11}}^TD_1^{-1}\mathbf1_{m_{11}}
}
\right)\\
&\quad\cdot
\left[
R_{11}
-\left((\mathbf1_{m_{11}}\otimes d^T)D_2^{-1}\right)R_{12}
\right].
\end{aligned}
\tag{45}
\]

**Eq. (46)** back-substitutes:

\[
D_2y_{12}=R_{12}-(\mathbf1_{m_{11}}^T\otimes d)y_{11},
\qquad
y_{12}=D_2^{-1}
\left(R_{12}-(\mathbf1_{m_{11}}^T\otimes d)y_{11}\right).
\tag{46}
\]

There is a material algebraic inconsistency in the manuscript:
Eq. (43) contains \(D_1-\alpha\mathbf1\mathbf1^T\), but Eq. (44) is the
Woodbury formula for a plus rank-one update. For the displayed minus update,
the standard identity is

\[
(D_1-\alpha uu^T)^{-1}
=D_1^{-1}
+\frac{D_1^{-1}uu^TD_1^{-1}}
{\alpha^{-1}-u^TD_1^{-1}u}.
\]

Accordingly, Eqs. (39), (44), and (45) must be validated against a direct
solve before implementation; their signs must not be copied silently.

Stage 4 resolved this gate numerically and algebraically. The implemented
Eq. (55) matrix gives the minus Schur complement in Eq. (43), so the valid
inverse uses the positive correction and difference denominator shown above.
The printed Eqs. (39), (44), and (45) fail the direct linear-system oracle and
remain recorded as a manuscript discrepancy.

#### \(y_2\): Eqs. (47)-(50)

**Eq. (47), diagonalizing proximal choice**

\[
\mathcal S_2=\lambda I_{m_2}-A_2A_2^*,
\tag{47}
\]

where the paper chooses \(\lambda=\lambda_1(A_2A_2^*)=\|A_2\|_2^2\).
Then \(A_2A_2^*+\mathcal S_2=\lambda I_{m_2}\).

**Eq. (48)** expands the projected inequality-multiplier subproblem:

\[
\begin{aligned}
\bar y_2^{k+1}=\arg\min_{y_2}\bigg\{
-\langle b_2,y_2\rangle+\delta_{\mathbb R_+^{m_2}}(y_2)
+\langle\bar x^{k+1},A_2^*y_2\rangle\\
+\frac{\sigma}{2}
\|A_1^*\bar y_1^{k+1/2}+A_2^*y_2+\bar z^{k+1}-c\|^2
+\frac{\sigma}{2}\|y_2-y_2^k\|_{\mathcal S_2}^2
\bigg\}.
\end{aligned}
\tag{48}
\]

**Eq. (49), optimality condition**

\[
\begin{aligned}
0\in{}&-b_2+\partial\delta_{\mathbb R_+^{m_2}}(y_2)
+A_2\bar x^{k+1}\\
&+\sigma A_2
\left(A_1^*\bar y_1^{k+1/2}+A_2^*y_2+\bar z^{k+1}-c\right)
+\sigma\mathcal S_2(y_2-y_2^k).
\end{aligned}
\tag{49}
\]

**Eq. (50), projected closed form**

\[
\bar y_2^{k+1}
=\Pi_{\mathbb R_+^{m_2}}
\left[
y_2^k+\frac{1}{\lambda}
\left(\frac{b_2}{\sigma}-A_2R_y\right)
\right],
\tag{50}
\]

\[
R_y=\frac{\bar x^{k+1}}{\sigma}
+A_1^*\bar y_1^{k+1/2}
+A_2^*y_2^k+\bar z^{k+1}-c.
\]

### 3.5 Complexity and stopping: Eqs. (51)-(54)

Corollary 1 states that \(A_1A_1^*y_1=R_1\) can be solved in
\(O(T(N_G+N_{RG}+N_{ESS}))\) flops. Corollary 2 states a large-system
per-iteration cost

\[
O\left(TN_L(N_G+N_{RG}+N_{ESS})\right).
\]

**Eq. (51), iteration bound**

\[
\left\lceil
\frac{1}{\epsilon}
\left(
\frac{\sigma(\|A^*\|+\|\sqrt{\mathcal T_1}\|)+1}{\sqrt\sigma}
R_0
\right)-1
\right\rceil
\tag{51}
\]

iterations suffice for \(\|\mathcal R(\bar w^k)\|\le\epsilon\).

**Eq. (52), overall flop bound**

\[
O\left[
\left(
\frac{\sigma(\|A^*\|+\|\sqrt{\mathcal T_1}\|)+1}{\sqrt\sigma}
R_0
\right)
\frac{TN_L(N_G+N_{RG}+N_{ESS})}{\epsilon}
\right].
\tag{52}
\]

The abstract/conclusion simplify this as \(O(N_Ln/\epsilon)\) for
large-scale DCOPF.

**Eq. (53), rearranged residual inequality**

\[
\frac{\sigma(\|A^*\|+\|\sqrt{\mathcal T_1}\|)+1}{\sqrt\sigma}
\frac{R_0}{k+1}\le\epsilon
\quad\Longrightarrow\quad
k\ge
\frac{1}{\epsilon}
\left(
\frac{\sigma(\|A^*\|+\|\sqrt{\mathcal T_1}\|)+1}{\sqrt\sigma}
R_0
\right)-1.
\tag{53}
\]

**Eq. (54), implemented stopping tests**

For \(\epsilon=5\times10^{-5}\), both sGS-HPR and HPR-LP terminate only
when all three conditions hold:

\[
\|\Pi_D(b-Ax)\|\le\epsilon(1+\|b\|),
\tag{54a}
\]

\[
\|x-\Pi_C(x-z)\|
\le\epsilon(1+\|x\|+\|z\|),
\tag{54b}
\]

\[
\|c-A^*y-z\|\le\epsilon(1+\|c\|).
\tag{54c}
\]

Because \(D=\mathbb R^{m_1}\times\mathbb R_+^{m_2}\), Eq. (54a)
contains the equality residual \(b_1-A_1x\) unchanged and the positive part
of \(b_2-A_2x\).

### 3.6 Explicit equality matrix: Eq. (55)

With the inferred variable order
\((p_G,p_{RG},p^{dc},p^{ch},r^u,r^d)\), Eq. (55) is

\[
A_1=
\begin{bmatrix}
I_T\otimes\mathbf1_{N_G}^T&
I_T\otimes\mathbf1_{N_{RG}}^T&
I_T\otimes\mathbf1_{N_{ESS}}^T&
-I_T\otimes\mathbf1_{N_{ESS}}^T&
0\\
0&0&
\mathbf1_T^T\otimes
\operatorname{diag}(-\Delta t/\eta^{dc})&
\mathbf1_T^T\otimes
\operatorname{diag}(\Delta t\,\eta^{ch})&
0
\end{bmatrix}.
\tag{55}
\]

The final zero block covers both reserve blocks and has \(2TN_G\) columns.
The first block row has \(T\) rows and implements Eq. (1). The second has
\(N_{ESS}\) rows and implements Eq. (9). Hence
\(A_1\in\mathbb R^{(T+N_{ESS})\times n}\).

For this explicit matrix, Eq. (37) specializes to

\[
D_1=(N_G+N_{RG}+2N_{ESS})I_T,
\]

\[
[D_2]_{ii}
=T\Delta t^2\left((\eta_i^{ch})^2+(\eta_i^{dc})^{-2}\right),
\qquad
d_i=-\Delta t\left(\eta_i^{ch}+(\eta_i^{dc})^{-1}\right).
\]

Appendix A is internally inconsistent with Eq. (55): its prose replaces
\(\mathbf1_T^T\) by \(I_T\), assigns \(TN_{ESS}\) rows to the second block,
and concludes \(A_1\) has \(T(1+N_{ESS})\) rows. Eq. (55), Section II-C
("Eq. (1) and Eq. (9)"), Eq. (9) itself, and every Table II constraint
dimension instead imply \(T+N_{ESS}\) equality rows. The implementation
must follow Eq. (55) and the table-consistent count, while retaining this
discrepancy as a paper erratum.

## 4. Independently derived dimensions

### 4.1 Constraint formula

The paper gives no explicit closed formula for \(m\), so derive it from its
row families. Box bounds in Eqs. (3), (7), and (10) belong to \(C\) and are
not counted as rows of \(A\).

| Family | Rows |
|---|---:|
| Eq. (1), balance | \(T\) |
| Eq. (9), terminal ESS | \(N_{ESS}\) |
| Eq. (2), two line-flow sides | \(2TN_L\) |
| Eq. (4), up/down reserve coupling | \(2TN_G\) |
| Eq. (5), up/down reserve requirement | \(2T\) |
| Eq. (6), two ramp sides | \(2(T-1)N_G\) |
| Eq. (8), two energy-bound sides | \(2TN_{ESS}\) |

Therefore,

\[
\boxed{m_1=T+N_{ESS}},
\]

\[
\boxed{
m_2=2TN_L+2TN_G+2T+2(T-1)N_G+2TN_{ESS}
},
\]

and

\[
\boxed{
m=2TN_L+(4T-2)N_G+(2T+1)N_{ESS}+3T
}.
\]

### 4.2 Table II check: case1354pegase, \(T=4\)

Table I gives
\(N_L=1{,}991,\ N_G=260,\ N_{RG}=136,\ N_{ESS}=68\).

\[
n=4(3\cdot260+136+2\cdot68)
=4(1{,}052)=\boxed{4{,}208}.
\]

\[
\begin{aligned}
m_1&=4+68=72,\\
m_2&=
\underbrace{2(4)(1{,}991)}_{15{,}928}
+\underbrace{2(4)(260)}_{2{,}080}
+\underbrace{2(4)}_{8}
+\underbrace{2(3)(260)}_{1{,}560}
+\underbrace{2(4)(68)}_{544}\\
&=20{,}120,\\
m&=72+20{,}120=\boxed{20{,}192}.
\end{aligned}
\]

Both independently reproduce Table II.

### 4.3 Table II check: case2868rte, \(T=16\)

Table I gives
\(N_L=3{,}808,\ N_G=600,\ N_{RG}=286,\ N_{ESS}=143\).

\[
n=16(3\cdot600+286+2\cdot143)
=16(2{,}372)=\boxed{37{,}952}.
\]

\[
\begin{aligned}
m_1&=16+143=159,\\
m_2&=
\underbrace{2(16)(3{,}808)}_{121{,}856}
+\underbrace{2(16)(600)}_{19{,}200}
+\underbrace{2(16)}_{32}
+\underbrace{2(15)(600)}_{18{,}000}
+\underbrace{2(16)(143)}_{4{,}576}\\
&=163{,}664,\\
m&=159+163{,}664=\boxed{163{,}823}.
\end{aligned}
\]

Both independently reproduce the case2868 T16 row of Table II.

### 4.4 Table II check: case9241pegase, \(T=6\)

Table I gives
\(N_L=16{,}049,\ N_G=1{,}445,\ N_{RG}=920,\ N_{ESS}=460\).

\[
n=6(3\cdot1{,}445+920+2\cdot460)
=6(6{,}175)=\boxed{37{,}050}.
\]

\[
\begin{aligned}
m_1&=6+460=466,\\
m_2&=
\underbrace{2(6)(16{,}049)}_{192{,}588}
+\underbrace{2(6)(1{,}445)}_{17{,}340}
+\underbrace{2(6)}_{12}
+\underbrace{2(5)(1{,}445)}_{14{,}450}
+\underbrace{2(6)(460)}_{5{,}520}\\
&=229{,}910,\\
m&=466+229{,}910=\boxed{230{,}376}.
\end{aligned}
\]

Both independently reproduce the case9241 T6 row of Table II.

## 5. Experimental specification

### 5.1 Hardware, software, storage, and kernel configuration

- GPU algorithms: NVIDIA A100-SXM4-80GB, CUDA 12.3.
- Gurobi: v12.0.0 on a separate workstation with an Intel i9-14900HX
  24-core CPU and 96 GB total memory; Gurobi memory capped at 80 GB.
- GPU competitors: cuPDLP [19] and HPR-LP [43]. No source revision or package
  version is stated for either.
- Matrices: compressed sparse row (CSR).
- SpMV: cuSPARSE `CUSPARSE_SPMV_CSR_ALG2`, selected for deterministic results.
- Custom vector kernels: \(\lceil n/256\rceil\) blocks and 256 threads per
  block.
- Time limit: 3,600 seconds for every case.
- Gurobi comparison: primal simplex, dual simplex, and barrier are each tested;
  the shortest time is reported. The text also describes Gurobi as operating
  in default multi-threaded mode.

### 5.2 Preconditioning, initialization, penalty, restart, and tolerance

- sGS-HPR uses 10 Ruiz-scaling iterations.
- It then uses the Pock-Chambolle bidiagonal preconditioner with \(\alpha=1\).
- The paper says \(b\) and \(c\) are normalized by division by
  \(\|b\|+1\) and \(\|c\|+1\), respectively.
- HPR-LP receives a "similar" preconditioner; cuPDLP uses its default
  preconditioning; Gurobi scaling-related parameters remain at defaults.
- sGS-HPR is cold-started with initial \(\sigma=1\); competitors are said to
  use the same initial values.
- Production tests use an adaptive \(\sigma\) update described only as derived
  from Theorem 1 and similar to HPR-LP [43]. The formula is not provided.
- Restart is checked every 100 iterations. The criterion and restart-state
  update are not provided; the paper refers to [29] and [43].
- sGS-HPR and HPR-LP use Eq. (54) on \(\bar w^k\) with
  \(\epsilon=5\times10^{-5}\).
- cuPDLP uses its default stopping criteria and default \(\sigma\) update.
- Gurobi `OptimalityTol`, `FeasibilityTol`, and `BarConvTol` are each set to
  \(5\times10^{-5}\).

#### Stage 5 sourced reconstruction

The manuscript specifies the preprocessing sequence and the 100-iteration
policy cadence, but not enough detail to recreate the control policy by itself.
Stage 5 therefore separates printed facts from the sourced reconstruction:

| Item | Implemented rule | Provenance |
|---|---|---|
| Ruiz scaling | 10 simultaneous row/column infinity-norm steps | DCOPF manuscript; formulas cross-checked against PDLP and HPR-LP |
| Pock-Chambolle | one simultaneous row/column L1 step, alpha = 1 | DCOPF manuscript, PDLP, and Pock-Chambolle |
| Vector normalization | after diagonal scaling, divide the complete scaled `b` and `c` vectors by `1 + norm(vector)` | DCOPF manuscript and HPR-LP |
| Restart | HPR-LP Eqs. (10)-(12), including the v0.1.0 forced first restart | HPR-LP v0.1.0, pinned at commit `1941fbcfbf2dae14e4a439b22f0ea1e1c05f4a29` |
| Policy cadence | inspect at iterations 100, 200, 300, and so on | DCOPF manuscript |
| Adaptive penalty | HPR-LP Eqs. (15)-(18), generalized to the sGS metric | published HPR-LP article |

The HPR-LP restart parameters are:

```text
sufficient-decay threshold   0.2
necessary-decay threshold    0.6
long-inner-loop fraction     0.2
```

Let `Qy(Δy)` denote the squared multiplier movement in the sGS metric. The
implemented adaptive update is:

```text
sigma_new = norm(Δx) / sqrt(Qy(Δy))
```

It is accepted only when both movement norms are strictly between `1e-16` and
`1e12`, and the normalized dual/primal infeasibility ratio is strictly between
`1e-8` and `1e8`. A failed guard resets sigma to 1, as specified by the
published HPR-LP rule.

The repository commit inspected for source drift,
`0f8f1501bcc7013b53fec6822e8da91929a39d2e`, postdates the published article
and contains additional penalty heuristics. Stage 5 deliberately implements
the published equations and the pinned v0.1.0 first-restart behavior instead
of silently adopting those later additions.

This is a sourced HPR-LP transfer with the DCOPF manuscript's 100-iteration
cadence. It is not claimed to be byte-for-byte equivalent to the unpublished
author implementation. The adaptive-without-restart case is a controlled
fixed-horizon ablation, not a paper algorithm.

### 5.3 Test systems

| Case | Buses | Branches \(N_L\) | Conventional \(N_G\) | RGs \(N_{RG}\) | ESSs \(N_{ESS}\) |
|---|---:|---:|---:|---:|---:|
| case1354pegase | 1,354 | 1,991 | 260 | 136 | 68 |
| case2868rte | 2,868 | 3,808 | 600 | 286 | 143 |
| case9241pegase | 9,241 | 16,049 | 1,445 | 920 | 460 |

The tested horizons are:

- case1354: \(T=4,16,48,96\);
- case2868: \(T=4,16,48,56,64,72,80,88,96\);
- case9241: \(T=4,6,16,24,32\).

### 5.4 Table II results (seconds)

| Case | \(m\) | \(n\) | nnz(\(A\)) | Gurobi | cuPDLP | HPR-LP | sGS-HPR | Avg. relative objective error vs Gurobi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case1354 T4 | 20,192 | 4,208 | 7,190,640 | 1.136 | 6.194 | 4.680 | 3.849 | 1.573E-05 |
| case1354 T16 | 82,124 | 16,832 | 28,791,792 | 1.945 | 11.679 | 8.338 | 4.113 | 4.817E-06 |
| case1354 T48 | 247,276 | 50,496 | 86,586,352 | 5.848 | 17.833 | 15.180 | 6.757 | 1.628E-04 |
| case1354 T96 | 495,004 | 100,992 | 173,800,432 | 15.231 | 29.671 | 19.340 | 7.857 | 1.569E-05 |
| case2868 T4 | 40,163 | 9,488 | 30,111,616 | 8.489 | 7.059 | 6.014 | 4.546 | 6.331E-05 |
| case2868 T16 | 163,823 | 37,952 | 120,508,576 | 34.659 | 17.136 | 13.717 | 7.695 | 2.990E-05 |
| case2868 T48 | 493,583 | 113,856 | 295,998,240 | 90.154 | 31.787 | 26.491 | 13.948 | 1.688E-04 |
| case2868 T56 | 576,023 | 132,832 | 345,459,808 | 104.152 | 34.624 | 28.879 | 15.988 | 1.076E-04 |
| case2868 T64 | 658,463 | 151,808 | 394,957,984 | 109.656 | 38.019 | 29.209 | 16.157 | 9.008E-05 |
| case2868 T72 | 740,903 | 170,784 | 444,492,768 | 115.025 | 43.451 | 31.857 | 16.808 | 4.795E-05 |
| case2868 T80 | 823,343 | 189,760 | 494,064,160 | 119.797 | 51.815 | 38.677 | 17.287 | 1.019E-04 |
| case2868 T88 | 905,783 | 208,736 | 543,672,160 | 124.926 | 62.112 | 47.456 | 17.538 | 6.694E-07 |
| case2868 T96 | 988,223 | 227,712 | 593,316,768 | 130.554 | 74.872 | 54.737 | 17.899 | 4.790E-08 |
| case9241 T4 | 152,774 | 24,700 | 373,238,888 | 66.904 | 43.702 | 23.110 | 8.040 | 2.635E-05 |
| case9241 T6 | 230,376 | 37,050 | 559,872,262 | 136.266 | 87.041 | 45.333 | 19.478 | 3.049E-05 |
| case9241 T16 | 618,386 | 98,800 | 1,493,149,532 | OOM | 351.764 | 177.678 | 59.784 | / |
| case9241 T24 | 928,794 | 148,200 | 2,239,903,828 | OOM | 854.565 | 430.475 | 138.550 | / |
| case9241 T32 | 1,239,202 | 197,600 | 2,986,775,884 | OOM | 1,492.22 | 719.247 | 227.682 | / |

The objective-error column is described as the average over five independent
runs. The paper does not define the relative-error denominator or explain why
independent runs differ despite deterministic SpMV.

### 5.5 Reported aggregate comparisons and sensitivity

- Table III contrasts update structures. It lists cuPDLP with
  \(\mathcal T_1=\lambda_1(AA^*)I_m-AA^*\), and marks relaxation,
  Halpern iteration, separate equality/inequality handling, and use of the
  DCOPF structure as absent. For sGS-HPR it lists Eq. (21) and marks all four
  properties as present.
- The paper reports "up to" or approximately \(6\times\) speedup over Gurobi
  for large instances. It does not report a single reproducible aggregation
  formula for this Gurobi claim.
- Using the shifted geometric mean
  \[
  \operatorname{SGM}_{10}
  =\left(\prod_{i=1}^n(t_i+10)\right)^{1/n}-10,
  \]
  it reports sGS-HPR speedups of \(2.12\times\) over HPR-LP and
  \(3.05\times\) over cuPDLP.
- Table IV reports SGM10 values: fixed \(\sigma=1\), 39.037; adaptive update
  initialized at \(\sigma=1\), 18.753; initialized at 0.1, 18.591; and
  initialized at 10, 18.926. The text describes the fixed-vs-adaptive
  improvement as \(2.08\times\).
- The text attributes roughly one second of overhead on small instances to
  GPU kernel launch and identifies nnz near \(10^8\) as the point beyond
  which the GPU method becomes advantageous.
- Gurobi reports OOM for the three case9241 rows whose nnz values are at or
  above approximately \(1.5\times10^9\).

## 6. Paper ambiguities and implementation cautions

These points are part of the specification because they prevent an unsupported
"exact reproduction" claim:

1. The paper gives case counts but not the source files, versions, hashes,
   renewable/ESS bus locations, profiles, reserve series, ramp inputs, ESS
   parameters, cost modifications, or construction scripts needed to recreate
   Table II matrices.
2. It gives neither the adaptive-\(\sigma\) formula nor the restart criterion;
   both are delegated to references [29] and [43]. Stage 5 reconstructs a
   sourced HPR-LP policy, but the exact DCOPF author code remains unavailable.
3. It does not state how \(\lambda_1(A_2A_2^*)\) is estimated, whether it is
   conservatively overestimated, or whether matrices and kernels use FP64,
   FP32, or mixed precision.
4. It does not identify the shift-factor slack/reference bus, treatment of
   transformers, out-of-service branches, islands, or numerical PTDF
   thresholding.
5. It does not define the exact timing boundary: preprocessing, scaling,
   matrix construction, host/device transfer, CUDA initialization, warm-up,
   kernel compilation, and residual checks are not separated.
6. It compares GPU methods and Gurobi on different hardware and reports the
   fastest of three Gurobi algorithms; this is a reported benchmark, not a
   hardware-controlled speedup experiment.
7. Eq. (15) drops the objective constants in Eqs. (12)-(13). They must be
   restored for absolute objective comparisons.
8. Equation (55) conflicts with Appendix A's row-count proof, as documented
   above. Eq. (55) and Table II support \(m_1=T+N_{ESS}\).
9. Equations (39)/(44)/(45) have a rank-one inverse sign inconsistency with
   Eq. (43). A direct linear solve is required as the reference oracle.
10. The manuscript does not print a separate conventional-generator box
    constraint. Nonnegative reserves plus Eq. (4) imply
    \(\underline P_G\le p_G\le\bar P_G\), while the compact-form prose says
    \(C\) contains all variable bounds. Whether those redundant \(p_G\) bounds
    are also placed explicitly in \(C\) should be resolved because the split
    changes the projection used by the algorithm even though it does not
    change the feasible set.
11. The paper describes a "security-constrained" model, but Eqs. (1)-(10)
    contain no N-1 contingency constraints. Any contingency extension is new
    research, not paper reproduction.
12. No public source-code URL or archived DCOPF implementation revision is
    supplied in the manuscript. The related HPR-LP source is public, but its
    exact revision in the authors' DCOPF runs is not identified.

Until these inputs and ambiguities are resolved, the defensible target is a
mathematical or structural reproduction, not an exact numerical recreation of
the paper's experiments.
