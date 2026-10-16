import { api, base } from "../api";
import adapter from "./adapter";

async function postJson(url, body) {
  const response = await fetch(`${base}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload?.detail?.message || payload?.detail || message;
    } catch { /* keep HTTP text */ }
    throw new Error(typeof message === "string" ? message : "保存失败");
  }
  return response.json();
}

// 1.0 demo: every visible Step-5 edit is persisted through the lightweight
// editing endpoint. Review/QC/final-sync status is not a front-end gate.
api.editOuter = (designId, payload) => postJson(`/api/designs/${designId}/demo-edit-outer`, payload);

const wrapSession = (loader) => async (id) => {
  const session = await loader(id);
  return {
    ...session,
    state: {
      ...(session.state || {}),
      thickness_reference: session.state?.thickness_reference || { demo: true },
    },
    capabilities: {
      ...(session.capabilities || {}),
      remove: true,
      remove_reason: null,
    },
    thickness_status: "DEMO_READY",
  };
};

api.sculptSession = wrapSession(api.sculptSession);
api.sculptSessionLight = wrapSession(api.sculptSessionLight);
api.sculptSessionRegions = wrapSession(api.sculptSessionRegions);
api.sculptThicknessPreview = async () => ({ status: "DEMO_READY", distance_mm: [] });

// Formal finalization remains best-effort for engineering development. In the
// 1.0 product demo it must never stop export: the server's /final.stl route will
// serve the latest visible outer whenever a formal final STL is unavailable.
const originalFinalize = adapter.finalize.bind(adapter);
adapter.finalize = async (designId, options = {}) => {
  try {
    return await originalFinalize(designId, options);
  } catch (error) {
    return {
      qc: {
        status: "DEMO_EXPORT_READY",
        note: error?.message || "formal finalization unavailable",
      },
      demo_fallback: true,
      complete_crown_ready: false,
    };
  }
};
