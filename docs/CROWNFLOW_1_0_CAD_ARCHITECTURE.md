# CrownFlow 1.0 CAD Architecture

## 1. Product goal

CrownFlow 1.0 is a deterministic, non-deep-learning automatic single-crown design system.

The design principle is:

- the tooth library provides anatomical identity and cusp/fossa morphology;
- the preparation/abutment provides the internal surface, margin and insertion direction;
- adjacent teeth provide mesial/distal spatial constraints and contact shape constraints;
- the antagonist provides crown height, occlusal clearance and local occlusal relief constraints;
- the final crown is produced by constrained deformation of the selected library tooth, not by copying surrounding surfaces.

The 1.0 release must prefer a review-required result over silently producing a geometrically invalid crown.

## 2. Two product workflows

### 2.1 Existing-abutment crown: fast automatic workflow

User-facing flow:

1. Home / create order.
2. Upload bite-aligned upper and lower jaw STL/DCM.
3. Select FDI tooth position and `abutment` indication.
4. Select tooth library.
5. Enter design parameters.
6. Press `Generate crown`.
7. Review final crown and export.

Hidden deterministic preparation performed by the backend:

1. identify/extract the restorative abutment support;
2. detect abutment finish line / crown margin;
3. derive insertion axis from abutment geometry;
4. build patient frame;
5. accept the uploaded upper/lower relative pose as the static bite pose (`REGISTERED_EXPORT`, identity transform);
6. identify mesial/distal adjacent contact patches from the working-jaw scan;
7. place the selected template;
8. solve proximal constraints;
9. solve occlusal constraints;
10. generate intaglio from the prepared abutment;
11. construct final shell and cervical seam;
12. run geometry, collision, contact and minimum-thickness QC.

The operator is not asked to click through Margin / Axis / Neighbor confirmation pages for routine abutment orders. If any automatic step is ambiguous, the fast flow stops and exposes an advanced review path instead of inventing a result.

### 2.2 Natural preparation: explicit professional workflow

Keep the current reviewable flow:

1. Case.
2. Margin / insertion axis / patient frame review.
3. Template and parameters.
4. Automatic crown candidate.
5. Optional sculpt / local correction.
6. Finalize and QC.

Natural preparations remain review-heavy because finish-line detection, undercut handling and insertion direction are less deterministic than an existing restorative abutment.

## 3. Geometry truth model

The crown is defined by five independent geometry sources.

### 3.1 Margin

The confirmed margin is the authoritative cervical boundary. It must be revisioned and reused by placement, outer deformation, intaglio and final cervical stitching.

No later stage is allowed to silently move the authoritative margin.

### 3.2 Patient frame

Use an orthonormal local frame:

- `X`: mesial direction;
- `Y`: buccal direction;
- `Z`: occlusal / insertion direction.

All placement and regional decisions are evaluated in this frame rather than world XYZ.

For abutments the frame is generated automatically from margin + insertion direction + working-jaw geometry. For natural preparations it remains reviewable.

### 3.3 Template anatomy

The template is not a finished crown. It provides:

- cusp/fossa anatomy;
- marginal ridges;
- buccal/lingual convexity;
- general mesiodistal proportions.

Template anatomy is preserved as much as possible. Automatic fitting should use the minimum deformation required to satisfy patient constraints.

### 3.4 Adjacent teeth

Adjacent teeth do not generate an entire crown side. They define a target contact patch and surrounding clearance envelope.

The proximal solver therefore uses:

- a small contact target region;
- a smooth transition band around the contact region;
- preserved template anatomy outside the band.

Movement should be biased toward the patient-frame mesiodistal axis to avoid nearest-point spikes.

### 3.5 Antagonist

The antagonist does not become a negative mold for the occlusal surface. It supplies one-sided constraints:

- remove penetration;
- enforce target clearance;
- cap excessive crown height;
- preserve template cusp/fossa pattern wherever sufficient space already exists.

The occlusal solver should primarily move vertices along the insertion/occlusal direction and distribute local changes with a smooth deformation operator.

## 4. Crown semantic zones

A single semantic-zone model must be computed once for a placed template and reused by all solvers.

Required 1.0 zones:

- `CERVICAL`: protected margin/collar band;
- `MESIAL`: candidate mesial contact and transition region;
- `DISTAL`: candidate distal contact and transition region;
- `OCCLUSAL`: cusp/fossa/table region;
- `BODY`: all remaining axial anatomy.

Rules:

- zones are expressed in patient-frame/template-local coordinates;
- every vertex belongs to exactly one primary zone;
- cervical vertices are locked during automatic proximal/occlusal fitting;
- the same vertex IDs are reused during all fitting/recheck stages;
- later versions may add cusp/fossa/marginal-ridge subzones without changing the main orchestration contract.

## 5. Automatic outer-crown pipeline

### Stage A: template placement

Input:

- selected FDI/template;
- confirmed margin;
- insertion axis;
- patient frame.

Operations:

1. align template +Z with insertion axis;
2. align template mesial axis to patient-frame X;
3. translate cervical loop to the patient margin region;
4. anisotropically scale mesiodistal, buccolingual and crown-height dimensions within configured limits;
5. reject implausible scaling rather than forcing placement.

The placement stage must not perform patient-contact deformation.

### Stage B: semantic zones

Compute/freeze zones on the placed mesh. All subsequent automatic fitting consumes this shared zone object.

### Stage C: proximal adaptation

For each confirmed neighbor:

1. build a signed-distance/query surface from the confirmed/automatic neighbor patch;
2. measure the proximal zone;
3. select only vertices requiring movement toward the configured contact target;
4. create target seed displacements biased along +/- patient-frame X;
5. propagate displacements through a local transition band with Laplacian/local-band deformation;
6. line-search the deformation magnitude;
7. reject any step causing face inversion, self-intersection or unacceptable distortion;
8. remeasure contact after persistence/readback.

Do not pull the entire proximal wall onto the neighbor surface.

### Stage D: occlusal adaptation

Input is the antagonist in the confirmed static bite pose.

For the occlusal zone:

1. measure signed clearance;
2. activate penetration/clearance-deficit vertices only;
3. calculate a target displacement primarily along -Z of the patient frame;
4. protect accepted proximal contact bands;
5. solve local-band deformation;
6. line-search and geometry-gate;
7. independently remeasure both occlusal and proximal constraints.

If occlusal correction destroys proximal contact beyond tolerance, reject the correction or iterate with a smaller step.

### Stage E: staged recheck

Do not build one monolithic nonlinear optimizer for 1.0.

Use deterministic staged iteration:

`proximal -> occlusal -> joint recheck`, up to a small bounded iteration count.

The system must never report PASS when a required stage is `BLOCKED`, `CONSTRAINT_CONFLICT`, `INPUT_UNVERIFIED` or unmeasured.

## 6. Intaglio and final crown

The internal crown is generated after outer-crown fitting.

Required order:

1. use preparation/abutment mesh and confirmed insertion direction;
2. block out undercuts directionally along the insertion axis;
3. apply spatial cement-gap field;
4. preserve the authoritative margin transition;
5. construct intaglio cutter/inner surface;
6. perform outer-minus-intaglio construction;
7. rebuild/stitch a 3D cervical seam at the real margin;
8. verify manifold/watertight output.

The inner surface must follow support geometry; it must never be derived from the opposing/adjacent teeth.

## 7. Minimum thickness policy

Minimum thickness is a manufacturing constraint, not an excuse to globally inflate the crown.

For 1.0:

- measure wall thickness against the actual intaglio;
- allow only bounded local outer corrections;
- protect cervical and accepted contact/occlusal regions;
- use a region-aware thickness profile when available;
- if correction would destroy contact, occlusion, anatomy or margin integrity, return `REVIEW_REQUIRED` / `CONSTRAINT_CONFLICT` instead of forcing geometry.

## 8. Automatic abutment reference extraction

The abutment fast flow must automatically derive references from the uploaded case.

### Margin / axis

Use existing abutment support extraction and finish-line logic. Persist the resulting margin and axis as authoritative revisions even though the user is not asked to click a confirmation button.

### Bite

Uploaded upper/lower scans are assumed to already be bite-aligned. Persist an identity rigid transform with provenance `REGISTERED_EXPORT`. Do not run hidden ICP/automatic bite registration in 1.0.

### Neighbors

Automatically identify mesial/distal reference patches on the working-jaw scan by combining:

- patient-frame side (mesial/distal);
- distance from the crown/preparation center;
- connected-component/topology checks;
- exclusion of support/preparation overlap;
- local surface orientation and contact-candidate distance.

If a reliable patch cannot be isolated, fail the fast path and request advanced review.

## 9. Frontend architecture

### Abutment UI

Only expose four user concepts:

1. Order.
2. Template.
3. Parameters.
4. Generate / Review.

The order page can show compact automatic-state indicators (Margin detected, Axis ready, Neighbor references ready, Bite accepted) but they are diagnostics, not mandatory clicks.

### Natural-prep UI

Retain detailed Margin / Axis / frame / neighbor review tools.

### Common design/review page

Both flows share the same final viewer, contact/occlusal heat maps, sculpt tools, QC and export endpoints.

## 10. Backend ownership

Target responsibilities:

- `api/abutment_fast.py`: fast-flow orchestration endpoints and automatic reference preparation;
- `services/generation_workflow.py`: candidate creation entry point;
- `services/auto_fit.py` or one designated optimizer service: orchestration only;
- `geometry/crown_outer/zones.py`: shared semantic zones;
- `geometry/crown_outer/auto_fit_adapter.py`: low-level region fitting / local deformation / geometry gates;
- `services/auto_fit_templates.py` + `geometry/registration.py`: template source, placement and scale;
- `geometry/intaglio.py` / final pipeline: intaglio and final shell;
- contact/thickness modules: independent measurement and QC.

Do not keep two competing orchestration pipelines live. The final 1.0 branch should have one authoritative outer-design route.

## 11. Required outputs and audit data

Every automatic candidate should record:

- selected library and source template hash;
- margin/axis/frame revision IDs;
- upper/lower scan hashes and bite transform provenance;
- neighbor patch provenance;
- parameter profile revision;
- per-stage status;
- before/after proximal and occlusal measurements;
- maximum and p95 vertex displacement;
- geometry-gate results;
- final thickness/contact/occlusion QC;
- final STL/hash.

This makes bad geometry reproducible instead of only visually debuggable.

## 12. Test strategy

### Unit tests

- semantic-zone partition and rotation equivariance;
- proximal movement direction and cervical locking;
- occlusal one-sided correction;
- contact protection during occlusal fitting;
- face-flip/self-intersection rejection;
- abutment auto-frame generation;
- automatic neighbor patch extraction;
- directional blockout and gap field;
- minimum-thickness protection policy.

### Synthetic integration cases

Use simple synthetic crowns/neighbor/antagonist meshes to assert exact expected changes.

### Golden clinical-like cases

Maintain a small non-patient-sensitive test set with expected ranges for:

- margin error;
- proximal signed clearance/contact patch;
- occlusal signed clearance;
- maximum deformation;
- minimum wall thickness;
- watertight/manifold output.

A visual screenshot alone is never a PASS criterion.

## 13. CrownFlow 1.0 acceptance definition

An automatic abutment crown is considered a successful 1.0 candidate only when:

1. a valid abutment support, margin and insertion direction were derived;
2. a selected tooth-library template was placed with plausible scale;
3. both proximal sides are either measured against a reliable neighbor patch or explicitly classified absent;
4. static antagonist clearance is measured using the uploaded bite relationship;
5. proximal and occlusal solvers completed without geometry conflict;
6. intaglio and margin seam were constructed successfully;
7. final mesh is watertight/manifold and passes geometry gates;
8. minimum thickness is within configured engineering tolerance;
9. the generated crown can be reviewed and exported without additional mandatory setup clicks.

If any required item is uncertain, the result is `REVIEW_REQUIRED`, not a silent success.

## 14. Explicitly out of scope for 1.0

- deep-learning crown generation;
- dynamic articulation / excursion simulation;
- automatic bite registration when scans are not already aligned;
- multi-unit bridges;
- custom implant abutment generation;
- manufacturing certification or automatic clinical approval.

These can be added later without changing the core template + constrained-deformation architecture.

## 15. Implementation priority

P0:

1. finish abutment fast flow end-to-end;
2. stabilize semantic zones and one authoritative outer optimizer;
3. make proximal and occlusal deformation visibly effective on real cases;
4. guarantee intaglio/finalize/QC integration;
5. add integration tests for one abutment and one natural-prep flow.

P1:

1. improve neighbor instance extraction;
2. tooth-class-specific zone parameters;
3. contact-patch shaping rather than only target distance;
4. region-aware minimum-thickness profile;
5. better automatic template scale estimation.

P2:

1. richer anatomy subzones;
2. multiple template morph controls;
3. optional statistical/ML models after the deterministic CAD baseline is stable.
