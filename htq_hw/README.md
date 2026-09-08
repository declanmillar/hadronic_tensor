# htq_hw

Self-contained Qiskit package for the Z2 hadronic-tensor hardware campaign
(Ns = 50 staggered sites = 100 system qubits + 1 ancilla).  The campaign target
is **ibm_phoenix** (Nighthawk r2, square lattice); Heron r3 (heavy-hex) is
supported throughout and is what the Ns = 58 cards are sized for, so nothing
here is specific to one device.  No dependency on the research code
(`htensor`); `tests/test_htq_hw_crosscheck.py` in the repository root proves
equality against it.  Nothing in this package submits to IBM unless
`submit --real --confirm <backend>` is invoked explicitly, and
`python -m htq_hw acceptance` (section 6) decides whether it may be run at all.

## 1. Physics in five lines

1. 1+1d Z2 lattice gauge theory with staggered fermions, Jordan-Wigner mapped:
   H = (g2/2) sum_links X_l - (m0/2) sum_n (-1)^n Z_n + (eta/4) sum_bonds (X_a X_b + Y_a Y_b) Z_l,
   (m0, g2, eta) = (0.7, 1.1, 1.3), periodic ring, the seam hop carries the fermion-parity sign.
2. Vector current: J0(v) = ((-1)^v - Z_v)/2 (charge), J1 on bond (a, b) = (eta/4)(Y_a X_b - X_a Y_b) Z_l;
   exactly conserved on the lattice (Ward identity), so the four components W^{mu nu} are consistent.
3. State: variational vacuum (2 layers) plus a local wavepacket block (L = 3, offsets -4..4) creating a
   boosted meson (k0 = 2pi/5, sigma_x = 0.75) centred on site 24; both trained at small volume and
   transferred verbatim (scalable circuits).
4. Observable: C^{mu nu}(x, t) = <psi| J^mu(x, t) J^nu(24, 0) |psi> by a Hadamard test: ancilla |+>,
   controlled J^nu insertion, second-order Trotter U(t) (dt = 0.5), read X_anc x J^mu(x) for all x
   at once; the four components come from J0/J1 insertion x J0/J1 probes.
5. Mitigation: every physics circuit is paired with a depth-matched mirror (same 2q skeleton, net
   identity), whose known t = 0 truth measures the per-site damping kappa_v; the calibrated slices
   feed the one-sided Fourier transform to W^{mu nu}(q0, q1).

## 2. Register and embeddings

Logical register (2Ns + 1 wires): matter site n -> wire 2n, link (n, n+1) -> wire 2n+1, seam link
(Ns-1, 0) -> wire 2Ns-1, ancilla -> wire 2Ns = 100.  Clbit i always holds logical wire i.

```
heavy-hex RING (Heron r3, FakeBoston/FakeKingston; 100-cycle found by face growth)

      s0 - l0 - s1 - l1 - s2 - ... - s49 - l49 -+      s = matter site, l = link
      |                                          |      consecutive wires on consecutive
      +------------------------------------------+      cycle qubits; hop gates need one
                     anc                                local SWAP (a and b straddle l)
                      |
             ... s23 - l23 - s24 - l24 - s25 ...        ancilla on the free third neighbour
                                                        of the hub carrying site 24

square-lattice LADDER (Nighthawk 12x10, 218 edges; 50-cycle of matter sites)

      l49  l0   l1   l2          pendant l_(n-1,n) hangs off site n
       |    |    |    |          rail edges (s_n, s_n+1) and pendants are all
      s0 - s1 - s2 - s3 - ...    coupling-map edges: routing-free, the layout is
       |                          preserved by every Trotter step (asserted)
      anc  (next to s24)
```

`choose_embedding` picks ring / ladder / grid-cycle / transpiler automatically and scores
candidates with the target's 2q and readout error rates.

## 3. Circuit families and counts

| family | insertion | c_a, id_a | gadget |
|---|---|---|---|
| j0 | J0 at site 24 | c_a = -1/2, id_a = +1/2 | one CZ(anc, wire 48) |
| j1p1 | Y_48 Z_49 X_50 | c_a = +eta/4, id_a = 0 | parity ladder: basis change, cx, cx, CZ(anc, 48), cx, cx, undo |
| j1p2 | X_48 Z_49 Y_50 | c_a = -eta/4, id_a = 0 | same skeleton as j1p1 |
| j0d (option) | J0 at site 25 | dither control | one CZ |

Each pub = prep + gadget + Trotter block (n = t/dt steps, or the mirror block with total angle
1e-8) + readout layer (Z: matter Z, links X; XYA: even matter Y, odd X, links Z; XYB: the
complement; ancilla X, or Y for the imaginary part).  The Trotter step is transpiled once with a
symbolic Parameter t, so physics and mirror have the identical two-qubit skeleton on the same
physical qubits by construction.

### Count table (offline, optimization_level=3, seed 7)

`python -m htq_hw report --targets fake:boston fake:nighthawk --ns 50 --steps 1 12` (and `--ns 58`).
Two-qubit counts after transpilation.  Ring = heavy-hex cycle on FakeBoston (Heron r3,
156q); ladder = Nighthawk 12x10 grid; grid = square-lattice 2Ns-cycle fallback.  The
Trotter step is transpiled with a symbolic `Parameter t` (physics `t -> n*dt`, mirror
`t -> 1e-8`, identical skeleton asserted).  Durations are `estimate_duration` against the
fake target; for the rzz basis the fake has no rzz, so durations come from a
`GenericBackendV2` twin and are placeholders.  Each 101-qubit O3 transpile took 0.1-3 s.

### Ns = 50 (100 system qubits + ancilla), steps 1 and 12, dt = 0.5

| target | embedding | basis | gadget | prep 2q (depth, us) | gadget 2q | step 2q (cz/rzz) | step 2q-depth | 12-step block 2q (depth, us) | prep+block12 2q (us) | layout |
|---|---|---|---|---|---|---|---|---|---|---|
| fake:boston | ring | cz | J0 | 872 (93, 12) | 1 | 534 (534/0) | 16 | 5484 (170, 21) | 6356 (33) | moved |
| fake:boston | ring | cz | J1a | 876 (93, 12) | 5 | 534 (534/0) | 16 | 5484 (170, 21) | 6360 (33) | moved |
| fake:boston | ring | rzz | J0 | 870 (93, 52) | 1 | 534 (186/348) | 16 | 5484 (170, 122) | 6354 (159) | moved |
| fake:boston | ring | rzz | J1a | 874 (93, 52) | 5 | 534 (186/348) | 16 | 5484 (170, 122) | 6358 (159) | moved |
| fake:nighthawk | ladder | cz | J0 | 603 (70, 8) | 1 | 300 (300/0) | 10 | 3600 (120, 14) | 4203 (22) | preserved |
| fake:nighthawk | ladder | cz | J1a | 607 (70, 8) | 5 | 300 (300/0) | 10 | 3600 (120, 14) | 4207 (22) | preserved |
| fake:nighthawk | ladder | rzz | J0 | 601 (70, 44) | 1 | 300 (150/150) | 10 | 3600 (120, 102) | 4201 (135) | preserved |
| fake:nighthawk | ladder | rzz | J1a | 605 (70, 44) | 5 | 300 (150/150) | 10 | 3600 (120, 102) | 4205 (135) | preserved |

### Ns = 58 (116 system qubits + ancilla), steps 1 and 12, dt = 0.5

| target | embedding | basis | gadget | prep 2q (depth, us) | gadget 2q | step 2q (cz/rzz) | step 2q-depth | 12-step block 2q (depth, us) | prep+block12 2q (us) | layout |
|---|---|---|---|---|---|---|---|---|---|---|
| fake:boston | ring | cz | J0 | 964 (93, 12) | 1 | 622 (622/0) | 16 | 6364 (170, 22) | 7328 (31) | moved |
| fake:boston | ring | cz | J1a | 968 (93, 12) | 5 | 622 (622/0) | 16 | 6364 (170, 22) | 7332 (31) | moved |
| fake:boston | ring | rzz | J0 | 962 (93, 56) | 1 | 622 (218/404) | 16 | 6364 (170, 130) | 7326 (165) | moved |
| fake:boston | ring | rzz | J1a | 966 (93, 56) | 5 | 622 (218/404) | 16 | 6364 (170, 130) | 7330 (165) | moved |
| fake:nighthawk | grid | cz | J0 | 918 (98, 11) | 1 | 569 (569/0) | 30 | 5874 (310, 33) | 6792 (45) | moved |
| fake:nighthawk | grid | cz | J1a | 922 (98, 11) | 5 | 569 (569/0) | 30 | 5874 (310, 33) | 6796 (45) | moved |
| fake:nighthawk | grid | rzz | J0 | 916 (98, 60) | 1 | 569 (241/328) | 30 | 5957 (369, 203) | 6873 (244) | moved |
| fake:nighthawk | grid | rzz | J1a | 920 (98, 60) | 5 | 569 (241/328) | 30 | 5957 (369, 203) | 6877 (244) | moved |

Expected (plan) vs observed at Ns = 50: prep 1024 -> 872 (ring); ring step 608 CZ -> 534 CZ
(`xy2cx` hop form), 458 rzz -> 534 (348 rzz + 186 cz) for a lone step but 457/step inside the
12-step block (5484); ladder step 300 -> 300 in both bases with the layout preserved.


## 4. Campaign: presets, jobs, shots (target ibm_phoenix, 3 h)

Target device: **ibm_phoenix** (IBM Nighthawk r2). Public information: 120
qubits on a square lattice with 218 couplers (the FakeNighthawk/FakeMiami
coupling map, so the ladder embedding of section 2 applies unchanged),
dissipative reset on every qubit, dynamic circuits, Heron-class two-qubit
error. Assumed until measured: per-shot time ~250 us (the 100k circuits/s
headline is not a per-shot number).  `--target phoenix` resolves to the real
device when the account can see it and otherwise to `fake:nighthawk`
labelled as its stand-in. What must be measured on the real target before
the plan is pinned: the per-shot time (`report`/`plan` print
estimate_duration of the deepest pub + reset + overhead and re-pin the shot
plan with it), dead or degraded couplers (the ladder needs 50 rail + 50
pendant + 1 ancilla edges; `choose_embedding` scores candidates with the
calibration data), fractional-gate (rzz) availability, and the
dynamic-circuit latency of a mid-circuit measure + reset (gauss-midcircuit).

Budget 3 h = 4.3e7 shots at 250 us; unit 1 pub x 6e4 shots = 15 s.  Presets
(`python -m htq_hw plan --target phoenix --preset relA-core prod-bridge vac-w00 qpdf-scan`),
composable; pub names carry a `preset.card:` prefix so several cards share
one campaign and `analyze --prefix` picks one card at a time:

| preset | card(s) | pubs | content |
|---|---|---|---|
| relA-core | relA_k1.26_s0.75_ns50 | 333 (72 stretch) | 12 slices t = 0.5..6.0 x 18 pubs (9 physics j0/j1p1/j1p2 x Z/XYA/XYB, 3 j0 mirrors, 3 j1p1 mirrors, 3 dither j0d = J0 at site 25 calibrated by the j0 mirror), 9 refs at t = 0, dt = 0.25 controls at t = 0.5 and 1.0 (18 each), stretch slices t = 6.5..8.0 (4 x 18) in their own jobs, ordered last |
| prod-bridge | prod_k1.26_s0.75_ns50 | 150 | J0 family only: j0 x 3 physics + 3 j0 mirrors + 3 dither per slice, 16 slices t = 0.5..8.0, refs j0 x 3 + j0d x 3 |
| vac-w00 | prod_vac_ns50, relA_vac_ns50 | 25 + 25 | vacuum-ansatz state (no block), j0 x Z physics + j0 mirror, 12 slices + 1 ref per card |
| qpdf-scan | six width cards + two vacuum cards | 8 x 20 | prep-only Wilson-line bilinears psi-bar(z) W psi(0), z = +-2m, m = 1..5 (section 4.1) |
| gauss-midcircuit | prod_k1.26_s0.75_ns50 | see section 4.2 | risk-gated mid-circuit Gauss syndromes |

Pairing (`group_jobs`, max 8 pubs): a calibration group = the physics pubs
of one (card, t, readout, dt) and their mirror(s); jobs never mix cards or
committed/stretch pubs; the t = 0 references of a card form one job; stretch
jobs come last so they can be dropped.

Shots (`shots_plan`, default with presets: equal 6e4 per pub, 2e4 for the
prep-only qPDF pubs, mirrors >= 3e4; `--weighting kappa` keeps the
1/kappa(t)^2 option): committed 128.5 min (relA-core 65.2, prod-bridge 37.5,
vac-w00 12.5, qpdf-scan 13.3), stretch 18.0 min, contingency 33.5 min at
250 us per shot, 4.16e7 shots.

### 4.1 qpdf-scan: readout bases and estimator

The bilinear for separation z (htensor/quasipdf.py `wilson_bilinear`) is
O_R = (X_a Z..Z X_b + Y_a Z..Z Y_b)/2, O_I = (X_a Z..Z Y_b - Y_a Z..Z X_b)/2
with a < b the two site qubits and Z on every qubit strictly between them
(links and matter).  All pairs are anchored at the centre site, so one
setting serves z = +2m and z = -2m together: setting `qPQm` measures the
centre qubit in P, the sites center +- 2m in Q and everything else in Z;
for each m the four settings XX, YY, XY, YX give
h(+2m) = (<qXX> + <qYY>)/2 + i(<qXY> - <qYX>)/2 and
h(-2m) = (<qXX> + <qYY>)/2 + i(<qYX> - <qXY>)/2 (the string product of the
+-1 outcomes between and including the two ends, `analyze.qpdf_term`).
20 prep-only pubs per card cover z in [-10, 10]; the same 20 on the
matching vacuum card give the connected (vacuum-subtracted) values.  The
reported quantity follows the reference convention of
scripts/quasipdf_analysis.py and data/qpdf_card_refs.npz:

    A(z) = <chi-dag(c+z) W chi(c)> = (C_R - i C_I) / 2   for z != 0,
    A(0) = <n(c)>  (the local density = J0 on the even centre site),

with C_R = <O_R>, C_I = <O_I> the Wilson-line bilinears, i.e.
A = conj(h)/2 for the raw h built above (`analyze.qpdf_amplitude`;
`analyze.qpdf_h_of_m` lays the values out on the same-sublattice grid
h(m) = A(2m), m = -5..5, matching the `<card>_h` keys of the reference
file).  A(0) comes from the Z-readout reference pub of the same card.  The
staggered taste phase (-1)^m and the window are applied downstream.
Eight cards (prod and relA at sigma_x = 0.75, 1.00, 1.50 plus the two
vacuum cards): 160 pubs, 13.3 min at 2e4 shots.

### 4.2 gauss-midcircuit (risk-gated)

`htq_hw/gauss.py`: j0 x Z family, 8 slices t = 0.5 .. 4.0 (16 pubs + 1 ref),
the five Gauss checks G_n = (-1)^n Z_n X_l- X_l+ of the width-2 patch around
the centre extracted after every second Trotter step onto ONE spare-qubit
ancilla adjacent to site 25 (the centre's free neighbour is the Hadamard
ancilla; the checks of sites 22, 23, 24, 26 reach the ancilla by rail relays
`relay_copy`, 1/4/10/22 CX for 0-3 hops), measured and reset for reuse:
8 CX per parity ladder + relay, 81 two-qubit gates per round, syndromes in
extra clbits (round-major, 5 per round).  On Gauss-law states the syndrome
bit equals n mod 2; the Aer dry run at Ns = 6 (statevector, mid-circuit
measure/reset) gives acceptance 1.000 in all rounds with the post-selected
readout equal to the plain circuit's, and a Pauli-frame injection lowers the
acceptance as expected.  Ns = 50: the ladder is re-chosen with the extra
ancilla (`choose_embedding(gauss_sites=[25])`, a notch-free two-site
segment is reserved and notches are placed by backtracking), the pubs are
transpiled at O1 with `routing_method="none"` (layout preserved, physics and
mirror skeletons identical): 1606 two-qubit gates for 2 steps + 1 round,
3649 for 8 steps + 4 rounds.  The FakeNighthawk target lists `measure` and
`reset` and accepts the dynamic pubs at validation (local Aer cannot hold
103 qubits, so the local-mode round trip is exercised on a small generic
square lattice).  RISKS for ibm_phoenix: mid-circuit reset latency and
measurement-induced dephasing of neighbours, and the +20 percent depth per
round; analysis reports acceptance vs depth and the calibrated slice with and
without post-selection (`gauss.acceptance`, `gauss.postselect`).

### 4.3 Calibration rule: kappa is per probe weight, never transferred

The mirror measures the damping of the *observable it is read with*, and
weight-3 (J1) probes damp faster than weight-1 (J0) probes in the same
circuit: an independent Ns = 16 study over 100 Pauli trajectories per family
gives ratios kappa_3 / kappa_1 = 0.93-0.97 at t = 0.5, 0.83-0.96 at t = 1
and 0.72-0.82 at t = 2 (errors 0.02-0.07), i.e.

    kappa_w ~ kappa_1 exp(-0.05 (w - 1) t).

**The ideal grid is only ever divided by at t = 0.**  The theta -> 0 mirror's
exact truth IS the t = 0 slice, so the calibration is
kappa(t) = sx_mirror(t) / sx_ideal(0) and beta(t) = (B_mirror(t) - 1/2) /
(B_ideal(0) - 1/2): only the t = 0 row of a grid enters a denominator.  Every
deeper row is used for validation alone (the `check` comparison, the
rehearsal's ideal reference, the wing-anchor target b_ideal(t) which is
subtracted, not divided).  A grid that stops short in time therefore limits
what can be *validated*, never what can be calibrated.

The analysis takes every probe's kappa from the mirror pub read in
that probe's own basis: J0 probes from the Z-readout mirror
(`_calib_J0`, kappa = sx_mirror / sx_ideal(0) with sx = xJ0 - id_b <x_anc>),
and the T1 / T2 probes from the XYA / XYB mirrors, per term
(`_calib_J1`, kappa_k = <x T_k>_mirror / <x T_k>_ideal(0)).  A J0-mirror
kappa is never applied to a J1 probe; the regression test
`test_kappa_never_transferred_between_probe_weights` injects 0.90 damping on
the Z readout and 0.60 on the XY readouts and requires each component to
recover its own value to 0.05 and to differ from the other by more than 0.15.

Corollary for the manifest: a slice that carries J1 physics pubs must carry
mirrors in all three readouts (the default), and `--j1-mirrors` additionally
gives the j1p1 family its own mirror so the insertion-dependent part of the
damping can be separated from the probe-dependent part.

## 5. Knobs IBM may tune, and invariants

Knobs (anything preserving the invariants below): transpiler seed, layout method, routing,
optimisation level, the choice of ring cycle / ladder rectangle, CZ vs fractional rzz basis (one
basis per session), dynamical-decoupling sequence, measurement twirling, shots per pub (keep the
1/kappa^2 shape), job packing (up to the pairing rule), qubit selection by calibration data.

Invariants:
- per-pub unitary equivalence to the logical circuit (`check` verifies to 5e-3 in expectation values);
- physics and mirror of a group have the identical two-qubit skeleton on the same physical qubits
  (`skeleton_block` hash in the job metadata);
- readout basis map and clbit -> logical map as recorded in `htq_job_<id>.json`; all 101 logical
  qubits measured in every pub;
- physics pubs and their mirror in one job;
- no dynamic circuits together with fractional gates; never mix CZ-basis and rzz-basis sessions in
  one estimator;
- mirrors >= 30000 shots.

### 5.1 What `submit --real` does with the allocation

The whole campaign is priced against `usage_remaining_seconds` **before job 0**: 112 jobs that each
fit individually can still overrun together, and the per-job estimate cannot see that.  Over budget
(more than `--guard`, default 85%, of what remains) nothing is submitted at all.  If the remaining
allocation cannot be read, that is not treated as "it fits": a real submission stops unless
`--no-strict-guard` says to proceed without the assurance.

The jobs then run inside one `Batch` with a single hoisted sampler, so they execute back to back
instead of re-queueing 112 times (`--no-batch` to disable).  If the per-job guard cancels a job, the
loop **stops** rather than sending the remaining jobs, and the cancelled job's metadata is written
anyway, so no spent job id is ever lost.  Resume from where it stopped with `--only-jobs`.

## 6. Acceptance gate

One command decides whether this package may be run or shipped, and writes the evidence:

```
PYTHONPATH=. python -m htq_hw --ns 50 acceptance --level fast \
    --target fake:nighthawk --record data/hw/acceptance.json      # minutes
PYTHONPATH=. python -m htq_hw --ns 50 acceptance --level full --require-real \
    --target ibm_phoenix --record data/hw/acceptance.json         # before any real run
```

Steps, each a PASS/WARN/FAIL line in the record:

| step | what it refuses to let through |
|---|---|
| env | an environment that does not match `requirements.txt` (the mirror skeleton is version sensitive) |
| git | (warning) a dirty tree: the bundle would not match any commit |
| target | with `--require-real`, a stand-in silently substituted for the named device |
| embedding | no ladder/ring on the *operational* graph, or redundancy < 2 (one dead qubit away from a transpiler fallback) |
| cards | a card whose ideal grids are missing, have no t = 0 row, or stop short of its slices with no wing surrogate |
| circuits | any of the 693 pubs failing to build, or a physics/mirror skeleton mismatch |
| plan | a shot plan over budget, or a mirror below the shot floor |
| check (full) | an ISA circuit whose logical expectation values disagree with the ideal grids above `--tol` |
| rehearse (full) | a card that produces no slices, or a noiseless kappa(centre) off 1 by more than `--kappa-tol` |

`--level fast` stops after `plan`; `--level full` adds `check` and a sampled `rehearse` + `analyze`
round trip at t = 0 and the shortest evolved slice of each card (`--accept-shots`, default 4000:
statistics do not matter here, the code path does).  `bundle` requires a full-level record whose
file hashes still match the tree, so a package cannot be shipped after an edit that was never
validated.

The unit suite underneath it:

```
PYTHONPATH=. python -m pytest htq_hw/tests -m "not slow"     # circuits, embeddings, manifest, plan,
                                                            # pairing, estimators, calibration,
                                                            # check/rehearse at Ns <= 6, local submit,
                                                            # the gate's own refusals
PYTHONPATH=. python -m pytest htq_hw/tests -m slow           # 101-qubit transpile bounds
PYTHONPATH=. python -m htq_hw --ns 50 check --target <backend> --times 0.5 1.0 --record check.json
PYTHONPATH=. python -m htq_hw --ns 50 rehearse --target <backend> --shots-scale 0.01 --record reh.json
```

`check` maps every ISA circuit back to the logical register through the recorded layouts and
compares Aer expectation values (statevector for Ns <= 10, MPS cap 512 from a prep MPS the package
builds once per card) with the ideal grids (`python -m htq_hw ideal`, hw_cal_grids key set).
`rehearse` samples the same ISA circuits, optionally with a Pauli-trajectory noise model
(`--noise 0.005 3e-4`: after each 2q gate with probability p2 a random Pauli on each of its
qubits, after each 1q gate with probability p1; a pub's shots are drawn in `--n-traj` batches
with a fresh trajectory each, since one fixed trajectory is a deterministic Pauli frame rather
than damping), writes fetch-format bits and runs the full analysis into slice files.

### 6.1 Reference-grid coverage (what the gate checks, card by card)

Every calibrated slice needs its card's ideal grids: the **t = 0 row** sets kappa and beta and is
mandatory, and the row at t sets the wing anchor.  Generated by `python -m htq_hw ideal` /
`scripts/hw_cal_grids.py`; installed under `cards/<card>/ideal_<family>.npz`.

| card | preset | ideal grids installed | slices | rows short | wing target |
|---|---|---|---|---|---|
| prod_k1.26_s0.75_ns50 | prod-bridge, qpdf-scan | j0 t<=8 (17), j0d t<=8 (17) | 0-8 (17) | 0 | - |
| prod_vac_ns50 | vac-w00, qpdf-scan | j0 t<=6 (13) | 0-6 (13) | 0 | - |
| relA_k1.26_s0.75_ns50 | relA-core, qpdf-scan | j0, j0d, j1p1, j1p2 all t<=3 (7) | 0-8 (17) | 10 | surrogate |
| relA_vac_ns50 | vac-w00, qpdf-scan | j0 t<=2 (4) | 0-6 (13) | 8 | surrogate |
| prod/relA _s1.00_, _s1.50_ | qpdf-scan | none needed (preparation only) | 1 t | - | - |

The relA grids stop at t = 3 because the MPS cost doubles every dt ~ 0.5 at that coupling; carrying
them to t = 8 is weeks of compute.  Only the **wing anchor** needs the deep rows, and the wing
signal is bulk vacuum breathing, so `data/wing_surrogate_relA.npz` (an exact small-ring calculation,
`scripts/wing_surrogate.py`) supplies it, validated against the Ns = 50 production wings to 2e-5
against a 2e-3 gate.  The surrogate carries its couplings and is refused for a card with a different
eta.  Without a surrogate the affected slices degrade to `wing_applied=False` (recorded, not silent)
and the gate reports them as warnings.

## 7. Report-back format

Per job: `data/hw/htq_job_<id>.json` (job id, backend, embedding kind and initial layout, pub
names, shots, per-pub basis map, clbit -> logical map, 2q counts, skeleton hashes, options) and
`data/hw/htq_bits_<id>.npz` (one uint8 array (shots, 101) per pub name, column i = logical qubit i;
`to_legacy_losch_bits` gives the reversed-column t0.5 / m0.5 view used by older poolers).  Analysis
output: `data/hw/slice_{comp}_t{T:.1f}.npz` with the tier-3 keys (times, probes, C, C_cal, C_err,
kappa_v, beta_v, b_cal, id_a, c_a, id_b, tier, backend, nshot, merged_jobs) plus component and t,
consumed by `scripts/hw_w_tensor.py`.  Please return the job JSON, the bits npz (or the job ids),
the calibration snapshot used for qubit selection, and the transpiled QPY if the skeleton hashes
differ from ours.

## 8. Device asks

- Mitigated depth: the t = 6 slice is ~5,500 two-qubit gates (4,203 on the ladder) after the
  872-gate preparation; we need kappa >= 0.4 there for the planned shots to resolve C at 3 sigma.
- Repetition rate: the 180-minute plan assumes 250 us per shot on ibm_phoenix.  At 4 ms it becomes
  ~39 h, so the plan re-pins itself from `estimate_duration` on the live backend (`_plan`), and a
  slow rate means a reduced manifest (`--times`, t <= 3) rather than an overrun.
- Session access to **ibm_phoenix** (Nighthawk r2) with the calibration snapshot at run time;
  a Heron r3 (ibm_boston class) is the fallback topology and is what the Ns = 58 cards are for.
- `rzz` (fractional gates) exposed on the session, if the CZ and fractional variants are both to
  be run; the package selects them with `--basis cz|rzz` and transpiles each separately.
- Dynamic-circuit + fractional-gate co-existence (for a future mid-circuit-reset variant); until
  then the campaign stays static and per-basis.

## 9. Running

**Step zero: the pinned environment.**  The physics/mirror skeleton assertion is version sensitive
(it fails on qiskit 2.4.1 and holds on the pinned 2.5.0), so install the pins before anything else.
Every command warns if the environment drifts, and `acceptance` fails on it.

```
python -m venv .venv && .venv/bin/pip install -r htq_hw/requirements.txt
```

**Accounts.**  Nothing here contacts IBM unless `submit --real` is used.  A real backend *name* is
only a Target for transpilation, and if the account cannot see that device the package substitutes
an offline stand-in and says so:  `ibm_phoenix not visible (...); using fake_nighthawk`.  Treat that
line as a failure whenever the result is meant to describe the real machine, which is what
`acceptance --require-real` enforces.  Save the credentials once with
`QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=..., instance=..., name=...)`;
they live in `~/.qiskit/qiskit-ibm.json` and never in this package.

```
cd <repo root>
PYTHONPATH=. python -m htq_hw audit
PYTHONPATH=. python -m htq_hw embed --targets fake:boston fake:nighthawk
PYTHONPATH=. python -m htq_hw report --targets fake:boston fake:nighthawk --ns 50 --steps 1 12
PYTHONPATH=. python -m htq_hw plan --target fake:boston --basis cz
PYTHONPATH=. python -m htq_hw --ns 50 ideal --times 0.5 1.0 ... --ideal "data/hw/ideal_{family}.npz"
PYTHONPATH=. python -m htq_hw --ns 50 check --target fake:boston --times 0.5 --ideal "data/hw/ideal_{family}.npz"
PYTHONPATH=. python -m htq_hw --ns 50 rehearse --target fake:boston --ideal "..." --out data/hw/rehearsal --noise 0.005 3e-4
PYTHONPATH=. python -m htq_hw submit --target fake:nighthawk --fetch        # local testing mode
PYTHONPATH=. python -m htq_hw submit --target ibm_phoenix --real --confirm ibm_phoenix   # spends the allocation
PYTHONPATH=. python -m htq_hw submit --target ibm_phoenix --real --confirm ibm_phoenix --only-jobs 7 8 9   # resume
PYTHONPATH=. python -m htq_hw fetch data/hw/htq_job_<id>.json
PYTHONPATH=. python -m htq_hw analyze data/hw/htq_bits_<id>.npz --prefix "relA-core.relA_k1.26_s0.75_ns50:"
PYTHONPATH=. python -m htq_hw bundle --acceptance data/hw/acceptance.json --report-json report.json
```

The campaign itself is composed from presets and its pubs are namespaced per card, so analysis
selects one card at a time:

```
PYTHONPATH=. python -m htq_hw --ns 50 plan --preset relA-core prod-bridge vac-w00 qpdf-scan
PYTHONPATH=. python -m htq_hw analyze data/hw/htq_bits_*.npz --list-prefixes
PYTHONPATH=. python -m htq_hw analyze data/hw/htq_bits_*.npz --preset relA-core --card relA_k1.26_s0.75_ns50
```

`submit --real` refuses unless the resolved backend *is* the named live device and `--confirm
<name>` repeats it; `submit` without `--real` refuses a live backend outright rather than
submitting while printing "local testing mode".

On a shared machine prefix the commands with `OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 nice -n 10`.
Targets: `fake:boston|kingston|fez|pittsburgh|marrakesh|torino|nighthawk|miami`, `grid:RxC`,
`heavyhex:d`, or a real IBM backend name.  Transpiled circuits are cached as QPY under `--cache`
(default `~/.cache/htq_hw`, `""` disables).

## 10. Provenance

Every ported function cites its `htensor` or `scripts/` source (`file:line`).  Card:
`cards/prod_k1.26_s0.75_ns50/card.json`, converted from `data/wp10reg_params_k1.26_L3.npz`
(block vector, offsets -4..4, L = 3, k0 = 2pi/5, sigma_x = 0.75, training fidelity 0.9936) plus
the Ns = 6 vacuum angles (`data/test/vac_prod_test.npz`).

## 11. Reference-grid provenance: how the 101-qubit circuits are simulated

The ideal grids (`ideal_{family}.npz` next to each card) and every `check`
are Aer matrix-product-state simulations of the actual circuits.  Two
choices matter for their accuracy, and both are recorded here so the
numbers can be reproduced.

**Chain order.**  The MPS chain folds the periodic ring in half,
`[0, 2Ns-1, 1, 2Ns-2, ...]`, with the ancilla wire inserted directly after
the centre site qubit (`sim.chain_order`, the order used by the reference
pipeline).  Every ring-neighbour pair, the seam included, is then at most
two chain sites apart, so the bond dimension stays small (max 96 for the
production prep, 121 for relA); in the plain ring order the seam gates drive
the bond dimension into the cap.

**Why Aer's own handling of non-adjacent gates truncates.**  In the folded
chain the two matter sites of a hop gate sit four sites apart.  Aer's MPS
applies such a gate by moving one qubit next to its partner site by site,
and each transient ordering exposes bipartitions whose Schmidt rank is far
above the state's natural one; with a bond cap those intermediate SVDs
truncate, and the discarded weight is irreversible.  The effect is large:
the same production prep gives `<H>` = -49.3540 (ring order, cap 512),
-49.3606 (folded order, Aer moves, cap 512, final max bond only 54, 35 min)
but -49.36904 when no qubit is ever moved (below).  At Ns = 12, where an
exact statevector exists, the moved simulation does not finish in five
minutes while the routed one reproduces the exact `<H>` to 3.5e-6.

**The routed alternative (`sim.route_to_chain`).**  Before simulation the
wire circuit is transpiled onto a *line* coupling map with Sabre, the chain
map as initial layout: every two-qubit gate becomes chain-local, the
inserted SWAPs are persistent (no move-back), and the final layout is
composed into the stored chain map so observables and measurements follow
their wires.  Noise (rehearsals) is applied to the wire circuit before the
simulation-only SWAPs.  The prep MPS then builds in 8 s instead of 35 min,
a full `check` of three families at t <= 1 takes 2.5 min per target, and
nothing is truncated below the cap.

**Convergence evidence (production card, prep `<H>`).**

| routing seed | cap | threshold | `<H>` | insert_1pt | max bond |
|---|---|---|---|---|---|
| 7 | 512 | 1e-10 | -49.3690386 | -0.0008676 | 96 |
| 11 | 512 | 1e-10 | -49.3690379 | -0.0008676 | 97 |
| 23 | 512 | 1e-10 | -49.3690382 | -0.0008694 | 97 |
| 7 | 1024 | 1e-12 | -49.3690368 | -0.0008678 | 124 |
| 11 | 2048 | 1e-14 | -49.3690368 | -0.0008678 | 163 |

Independent main-repo port of the same routing: -49.3690391.  Packet
energies above the vacuum (routed, converged to 2e-6): production
2.8564 (vacuum -52.225438; the ns = 12 certification value 2.856, the
earlier +0.015 was truncation), relA k0 = 2pi/5 3.1084 (vacuum -64.858350,
prep -61.749934; ns = 20 certification 3.105-3.108, the moved-qubit
reference gave 3.264), relA k0 = 0 2.8864 (prep -61.971950).

**Rehearsal noise model.**  `rehearse --noise p2 p1` draws each pub's shots
in `--n-traj` batches, one Pauli trajectory per batch (a single trajectory is
a frozen error frame, i.e. a deterministic sign pattern, not damping).  The
trajectory count enters the *calibration*, because the mirror's per-site
kappa is estimated from the same frozen frames.  Measured at Ns = 50,
p2 = 0.005, p1 = 3e-4 (fake:boston, rzz basis, W00 component at t = 1.0):
with 4 trajectories the kappa map has spread 0.135 over the ring (range
0.19 to 1.04) and the calibrated correlator misses the ideal grid by up to
0.19-0.25; with 32 trajectories the spread falls to 0.052 (range 0.62 to
0.96) and the miss to 0.07-0.09 at half the shots, with the J1-insertion
components at 0.006-0.025.  The noiseless rehearsal at 2e4 shots per pub
reproduces the ideal grids to 0.004-0.010 in every component
(chi2/dof 0.5-3.9, kappa 0.95-1.01).  Use at least a few dozen trajectories
when the rehearsal is meant to exercise the calibration rather than the
plumbing; hardware noise is genuinely stochastic and has no such artefact.

**Consequences.**  Grids produced with moved-qubit simulations carry
3-7e-3 errors on the packet-region T probes, which is exactly what `check`
reported against them; against routed grids the ISA circuits agree to
1.2e-4 (prep re-simulated from |0>) and 1-4e-5 (Trotter blocks and
mirrors) at t <= 1 and to 2e-4 at t = 2, 4, 6 on both the heavy-hex ring
and the Nighthawk ladder.  Convergence of the production setting (cap 512,
truncation 1e-10), production card, j0 family, over all 452 observables:
at t = 6 (12 Trotter steps) raising the cap to 1024 changes nothing above
1.2e-7, i.e. the bond dimension never reaches the cap; at t = 3 the full
scan gives (512, 1e-10) vs (1024, 1e-12) 6.6e-6 and (512, 1e-12) vs
(1024, 1e-12) 1.8e-7, so the truncation threshold, not the cap, sets the
error and 1e-10 is already two orders below the 2e-4 agreement between the
two independent pipelines.  The prep itself is stable to 2e-6 in <H> and
2e-7 in insert_1pt across caps 512/1024/2048 and thresholds
1e-10/1e-12/1e-14.  Tightening to 1e-12 costs 7x the runtime (5839 s vs
833 s for one t = 3 ancilla circuit) and buys nothing; the routed grids of this package and the main
repository's independent routed port agree to 1.3e-4 over all 13 times and
three families (differences grow from 1e-5 at t <= 1 to 1e-4 at t >= 3 and
come from the different gate decompositions of the two pipelines).
