// Crown Design 1.0 主工作流：步骤定义、进度状态与准入判据。
// 生成完整牙冠后直接进入微调画布；检查/导出不再作为独立必经页面。

export const S = {
  HOME: "home",
  CASE: "case",
  MARGIN: "margin",
  SETUP: "setup",
  DESIGN: "design",
  // 保留 legacy id，旧设计/旧链接仍可映射回 DESIGN，但不再出现在主流程。
  REVIEW: "review",
};

export const FLOW = [
  { id: S.HOME, n: 1, label: "首页", title: "数字牙冠设计" },
  { id: S.CASE, n: 2, label: "病例信息", title: "病例信息" },
  { id: S.MARGIN, n: 3, label: "Margin / Axis", title: "边缘线与插入轴" },
  { id: S.SETUP, n: 4, label: "参数 / 模板", title: "参数与牙冠模板" },
  { id: S.DESIGN, n: 5, label: "微调 / 导出", title: "牙冠微调与导出" },
];

export const ORDER = FLOW.map(step => step.id);

export function indexOf(id) {
  return ORDER.indexOf(id);
}
export function stepOf(id) {
  if (id === S.REVIEW) return FLOW.find(step => step.id === S.DESIGN) || FLOW[0];
  return FLOW.find(step => step.id === id) || FLOW[0];
}
export function prevStep(id) {
  const normalized = id === S.REVIEW ? S.DESIGN : id;
  const index = indexOf(normalized);
  return index > 0 ? ORDER[index - 1] : null;
}
export function nextStep(id) {
  const normalized = id === S.REVIEW ? S.DESIGN : id;
  const index = indexOf(normalized);
  return index >= 0 && index < ORDER.length - 1 ? ORDER[index + 1] : null;
}

/** 顶部 Stepper 的三种状态：已完成 / 当前 / 未开始。 */
export function stepState(id, current) {
  const normalizedCurrent = current === S.REVIEW ? S.DESIGN : current;
  const self = indexOf(id);
  const now = indexOf(normalizedCurrent);
  if (self < 0 || now < 0) return "todo";
  if (self === now) return "active";
  return self < now ? "done" : "todo";
}

/**
 * 1.0 生成优先：只有建立患者空间所必需的输入可以挡住生成前步骤。
 * DESIGN 不再以“已通过审核”为导航条件；生成出来就能看、能微调，导出时
 * 如 final 因微调失效则自动同步一次完整冠，不再暴露额外“最终检查流程”。
 */
export function gateFor(id, facts = {}) {
  const normalized = id === S.REVIEW ? S.DESIGN : id;
  const needs = {
    [S.CASE]: [],
    [S.MARGIN]: [["hasCase", "请先在首页新建或打开一个病例"]],
    [S.SETUP]: [["marginConfirmed", "请在第 3 步点击“确认当前 Margin”"], ["axisConfirmed", "请先确认插入轴"], ["frameConfirmed", "请在第 3 步确认近中/颊侧方向"]],
    [S.DESIGN]: [],
  }[normalized] || [];
  for (const [key, message] of needs) if (!facts[key]) return { ok: false, reason: message };
  return { ok: true, reason: null };
}

export function canVisit(id, facts, current) {
  const normalized = id === S.REVIEW ? S.DESIGN : id;
  const normalizedCurrent = current === S.REVIEW ? S.DESIGN : current;
  if (indexOf(normalized) <= indexOf(normalizedCurrent)) return { ok: true, reason: null };
  return gateFor(normalized, facts);
}

export function progressPercent(id) {
  const normalized = id === S.REVIEW ? S.DESIGN : id;
  const index = indexOf(normalized);
  return index < 0 ? 0 : Math.round((index / (ORDER.length - 1)) * 100);
}

/** 病例事实汇总：页面与 Stepper 共用，避免各页面各写一份布尔表达式。 */
export function factsFrom(state) {
  return {
    hasCase: !!state.caseId,
    hasScans: !!(state.assets?.upper || state.assets?.lower),
    hasSupport: !!state.assets?.support,
    hasDesign: !!state.design,
    marginConfirmed: !!state.confirmed?.margin,
    axisConfirmed: !!state.confirmed?.axis,
    frameConfirmed: !!state.confirmed?.frame,
    profileConfirmed: !!state.confirmed?.profile,
    hasOuter: !!state.hasOuter,
    hasFinal: !!state.hasFinal,
  };
}
