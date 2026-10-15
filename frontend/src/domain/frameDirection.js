function normalize(v) {
  const n = Math.hypot(...(v || []));
  return v && v.length === 3 && Number.isFinite(n) && n > 1e-9 ? v.map((x) => x / n) : null;
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function suggestPatientFrame(marginPoints, insertionAxis) {
  const z = normalize(insertionAxis);
  if (!z) throw new Error("插入轴无效，无法推导横向方向");
  const points = (marginPoints || []).map((p) => [...p]);
  if (points.length < 3) throw new Error("边缘线点不足，无法推导近中方向");

  const origin = points.reduce((sum, p) => sum.map((v, i) => v + p[i] / points.length), [0, 0, 0]);
  const reference = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    .reduce((best, candidate) => Math.abs(candidate.reduce((s, v, i) => s + v * z[i], 0))
      < Math.abs(best.reduce((s, v, i) => s + v * z[i], 0)) ? candidate : best);
  const u = normalize(cross(z, reference));
  const v = cross(z, u);
  let uu = 0; let uv = 0; let vv = 0;
  for (const p of points) {
    const rel = p.map((value, i) => value - origin[i]);
    const x = rel.reduce((s, value, i) => s + value * u[i], 0);
    const y = rel.reduce((s, value, i) => s + value * v[i], 0);
    uu += x * x; uv += x * y; vv += y * y;
  }
  uu /= points.length; uv /= points.length; vv /= points.length;
  const gap = Math.hypot(uu - vv, 2 * uv);
  const largest = (uu + vv + gap) / 2;
  if (largest <= 1e-12 || gap <= largest * 0.03) {
    throw new Error("边缘线近远中跨度相近，无法稳定推导方向，请手动调整方向");
  }
  const angle = Math.atan2(2 * uv, uu - vv) / 2;
  const mesial = normalize(u.map((value, i) => value * Math.cos(angle) + v[i] * Math.sin(angle)));
  const buccal = normalize(cross(z, mesial));
  return { origin, mesial, buccal, occlusal: z, sign_ambiguous: true };
}
