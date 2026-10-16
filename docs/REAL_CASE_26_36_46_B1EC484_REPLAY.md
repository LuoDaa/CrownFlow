# CAD-1 real-case replay baseline — 26 / 36 / 46

Date: 2026-10-16

Baseline code: `b1ec4848aba0921a54c62e492a543bb11eee5f2d`

This document records the **local real-case replay supplied from the CrownFlow workstation** after synchronising `patient_envelope.py`.  It is not a synthetic-CI result and it must not be used to claim the cases are clinically accepted.  Its purpose is to pin the three observed failure classes so later deterministic geometry changes cannot silently regress them.

## Replay summary

| Case | FDI26-4 | FDI36-2 | FDI46-1 |
| --- | --- | --- | --- |
| Neighbor extraction | both selected, core 773 / 486 | both selected, core 586 / 613 before support eligibility block | both `UNSET`, core/ref null |
| Mode | `BOTH` | generation not reached | no `PROXIMAL` stage |
| Margin full loop | PASS, 100%, p95 0.0099 mm | blocked by support-loop selection | PASS, 100%, p95 0.0178 mm |
| Proximal | `CONSTRAINT_CONFLICT`; both `target_reached=False`; nearest 0.003 / 0.001 mm but only point contact | not reached | stage missing |
| Occlusal | `CONSTRAINT_CONFLICT`; median gap about +3.05 mm; penetration 0 | not reached | stage said `PASS_ENGINEERING`, but median gap about +2.04 mm; penetration 0 |
| Height loop | reported PASS after 10 passes | not reached | reported PASS |
| Shell | PASS; reported min 0.114 / p05 0.80 mm | not reached | `REVIEW_REQUIRED`; min 0.033 mm |
| Final | blocked: ~1098 self intersections | no generation | blocked: ~1743 self intersections |

## FDI26 patient envelope measurement

The six-direction patient cage was reconstructed from the replay and was internally plausible:

- MD interval `u`: approximately `-4.78 … +3.80 mm`
- BL interval `v`: approximately `-4.04 … +4.70 mm`
- `target_top_z`: approximately `5.717 mm`
- median Margin Z: approximately `-0.085 mm`
- resulting cage height: approximately `5.80 mm`

The crown maximum Z reached approximately the cage top, while the occlusal-domain median still remained about **3.05 mm** below the antagonist.  Therefore the failure was not simply “the box is too short”.  A single/global affine Z mapping can let one cusp reach the ceiling while the remaining occlusal surface stays far below a curved antagonist envelope.

Pinned interpretation: **global cage dimensions are necessary but not sufficient; coronal fitting needs a smooth local-column/height-field solve with Margin and established proximal contacts protected.**

## FDI36 pinned failure: short sliver chosen as Margin support loop

The support crop contained two boundary loops:

- true finish-line loop: about **332 points / 19.52 mm**
- short sliver loop: about **15 points / 1.15 mm**

The old selector minimised only candidate-loop → Margin error.  The sliver happened to have a floating-point-level maximum error around `7.38e-16`, compared with around `1.26e-15` for the full true ring, so the sliver won even though it represented only a tiny fraction of the finish line.  The following full-support validation then failed with `MARGIN_NOT_ON_SUPPORT`.

Pinned acceptance rule:

1. candidate loop → Margin error must pass;
2. **Margin → candidate loop full-circle coverage** must pass;
3. loop perimeter must be commensurate with the confirmed Margin perimeter;
4. sub-arc/sliver loops can never win because of smaller floating-point residuals.

## FDI46 pinned failure: visible neighbors degraded to UNSET

On the fresh replay both mesial and distal automatic neighbors were `UNSET` and their core/reference counts were null, so no proximal stage ran.  A previous build on the same scan had produced a confirmed neighbor on at least one side.

Pinned acceptance rule:

- `ABSENT` is allowed only when the bounded coronal search has **no geometric candidate**;
- if candidate faces exist but the connected core/reference construction fails, the result remains an explicit extraction failure with diagnostic counts;
- the automatic extractor must try a deterministic bounded connected-component fallback before returning `UNKNOWN`;
- diagnostics must include strict/relaxed candidate count, seed, core/reference size and fallback/failure reason.

## Occlusal gate semantics exposed by FDI26/46

A stage status of `PASS_ENGINEERING` is not enough.  FDI46 demonstrated a stage PASS while the median gap was still about **+2.04 mm**, and FDI26 demonstrated a height-loop PASS while the median gap was about **+3.05 mm**.

For CAD-1 the solver and the final joint recheck must distinguish:

- **solver execution succeeded**;
- **geometry target was reached**.

The minimum occlusal target gate must include real full-domain signed-distance statistics (median/coverage plus penetration), not only whether an optimisation pass completed.

## 1.0 invariant order

The real-case regressions are evaluated in this authority order:

1. **Margin** — establish and freeze the cervical root.
2. **Mesial/distal** — grow the body from the frozen root until a continuous proximal contact domain exists; do not move Margin.
3. **Occlusion** — fit the coronal surface to the antagonist envelope while keeping Margin and already-established proximal contact protected.
4. **Material shell** — repair only violations and re-run only the constraint domains actually touched/degraded.
5. Joint remeasurement — a stage is not accepted merely because its solver returned normally.

The library tooth remains a dimensionless anatomy/topology prior.  Patient geometry owns MD, BL, cervical boundary and coronal height.

## Next replay acceptance

The next fresh-case replay of 26 / 36 / 46 should record, at minimum:

- 36 reaches generation; the 19.52 mm finish-line loop wins over the 1.15 mm sliver;
- 46 has deterministic neighbor states (`CONFIRMED` or true `ABSENT`), not unexplained double `UNSET`;
- 26 occlusal median gap falls from ~3 mm into the CAD-1 target band without Margin drift or new severe self intersections;
- all three retain full-loop Margin fit;
- every confirmed proximal side has a connected contact domain, not 1–3-point touch;
- finalization self-intersection count must be zero before the case is called a 1.0 finished crown.
