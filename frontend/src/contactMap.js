export const CONTACT_INVALID_COLOR = '#9ba8ae';
export const CONTACT_BAND_MM = 0.02;

// Legend and mesh use the same signed-distance stops; target comes from the confirmed profile.
export function contactStops(target = 0) {
  target = Number.isFinite(target) && target >= 0 ? target : 0;
  return [[-.3, '#991b32'], [-.15, '#ee3d46'], [-.03, '#ff9b38'],
    ...(target > 0 ? [[0, '#f6df55']] : []), [target, '#30b98a'],
    [target + .08, '#38c3d9'], [Math.max(.2, target + .15), '#306bc9']];
}

export function contactColor(signed, valid, target = 0) {
  if (!valid || !Number.isFinite(signed)) return CONTACT_INVALID_COLOR;
  const stops = contactStops(target);
  if (signed <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [high, b] = stops[i], [low, a] = stops[i - 1];
    if (signed > high) continue;
    const t = (signed - low) / (high - low);
    const rgb = [1, 3, 5].map(k => Math.round(parseInt(a.slice(k, k + 2), 16) * (1 - t)
      + parseInt(b.slice(k, k + 2), 16) * t).toString(16).padStart(2, '0'));
    return '#' + rgb.join('');
  }
  return stops.at(-1)[1];
}

export function contactMetrics(contacts, target) {
  const signed = contacts?.signed_mm, valid = contacts?.valid;
  if (!signed || !valid || !Number.isFinite(target)) return null;
  const usable = signed.map((v, i) => valid[i] && Number.isFinite(v));
  const values = signed.filter((_, i) => usable[i]);
  if (!values.length) return null;
  let area = 0;
  for (const [a, b, c] of contacts.faces || []) {
    const pa = contacts.vertices[a], pb = contacts.vertices[b], pc = contacts.vertices[c];
    const ab = pb.map((v, k) => v - pa[k]), ac = pc.map((v, k) => v - pa[k]);
    const cross = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
    const count = [a, b, c].filter(i => usable[i] && Math.abs(signed[i] - target) <= CONTACT_BAND_MM).length;
    area += Math.hypot(...cross) / 2 * count / 3;
  }
  return { minimum: Math.min(...values), penetration: Math.max(0, -Math.min(...values)), area,
    validCount: values.length, totalCount: signed.length };
}
