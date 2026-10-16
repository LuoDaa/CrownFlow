import { api } from "../api";
import adapter from "./adapter";

// CrownFlow 1.0 test mode: AUTO generation is patient-space driven and the
// backend owns the irreducible input checks. Legacy template_allowed and
// automatic_mode frontend gates can stop a valid run before generation starts,
// so treat eligibility as diagnostic context only and always submit AUTO.
adapter.generateInitialCrown = async (designId, { kind = "AUTO" } = {}) => {
  const eligibility = await api.generationEligibility(designId);
  return api.generationCandidate(designId, {
    kind,
    expected_input_digest: eligibility.input_digest,
    confirmed: true,
  });
};
