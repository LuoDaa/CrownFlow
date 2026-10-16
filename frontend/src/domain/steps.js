// Crown Design 6 步工作流：步骤定义、进度状态与准入判据。
// 单一真源：Stepper、底部导航、页面路由都从这里取，避免三处各写一遍顺序。

export const S = {
  HOME: "home",
  CASE: "case",
  MARGIN: "margin",
  SETUP: "setup",
  DESIGN: "design",
  REVIEW: "review",
};

export const FLOW = [
  { id: S.HOME, n: 1, label: "首页", title: "数字牙冠设计" },
  { id: S.CASE, n: 2, label: "病例信息", title: "病例信息" },
  { id: S.MARGIN, n: 3, label: "Margin / Axis", title: "边缘线与插入轴" },
  { id: S.SETUP, n: 4, label: "参数 / 模板", title: "参数与牙冠模板" },
  { id: S.DESIGN, n: 5, label: "牙冠设计", title: "牙冠设计" },
  { id: S.REVIEW, n: 6, label: "检查 / 导出", title: "检查与导出" },
];

export const ORDER = FLOW.map(step => step.id);

export function indexOf(id) {
  return ORDER.indexOf(id);
}
export function stepOf(id) {
  return FLOW.find(step => step.id === id) || FLOW[0];
}
export function prevStep(id) {
  const index = indexOf(id);
  return index > 0 ? ORDER[index - 1] : null;
}
export function nextStep(id) {
  const index = indexOf(id);
  return index >= 0 && index < ORDER.length - 1 ? ORDER[index + 1] : null;
}

/** 顶部 Stepper 的三种状态：已完成 / 当前 / 未开始。 */
export function stepState(id, current) {
  const self = indexOf(id);
  const now = indexOf(current);
  if (self < 0 || now < 0) return "todo";
  if (self === now) return "active";
  return self < now ? "done" : "todo";
}

/**
 * 1.0 生成优先：只有建立患者空间所必需的输入可以挡住生成前步骤。
 * DESIGN/REVIEW 不再以“已采纳 outer / 已通过审核”为导航条件——用户必须
 * 能进入画布查看本轮生成结果，再决定是否采纳、继续修复或导出。
 *
 * 质量审核仍由后端报告承担，但它只影响接受/制造语义，不影响预览。
 */
export function gateFor(id, facts = {}) {
  const needs = {
    [S.CASE]: [],
    [S.MARGIN]: [["hasCase", "请先在首页新建或打开一个病例"]],
    [S.SETUP]: [["marginConfirmed", "请在第 3 步点击“确认当前 Margin”"], ["axisConfirmed", "请先确认插入轴"], ["frameConfirmed", "请在第 3 步确认近中/颊侧方向"]],
    [S.DESIGN]: [],
    [S.REVIEW]: [],
  }[id] || [];
  for (const [key, message] of needs) if (!facts[key]) return { ok: false, reason: message };
  return { ok: true, reason: null };
}

export function canVisit(id, facts, current) {
  if (indexOf(id) <= indexOf(current)) return { ok: true, reason: null };
  return gateFor(id, facts);
}

export function progressPercent(id) {
  const index = indexOf(id);
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
