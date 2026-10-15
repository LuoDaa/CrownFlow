/** Crown Design 6 步工作流模型：顺序、进度、准入（与 tests/ 现有 node:assert 风格一致）。 */
import assert from 'node:assert/strict';
import { FLOW, S, ORDER, stepState, gateFor, canVisit, prevStep, nextStep, factsFrom, progressPercent } from '../src/domain/steps.js';

// 1) 固定 6 步且顺序就是新版工作流
assert.equal(FLOW.length, 6, '流程必须是 6 步');
assert.deepEqual(ORDER, ['home', 'case', 'margin', 'setup', 'design', 'review']);
assert.deepEqual(FLOW.map((step) => step.label), ['首页', '病例信息', 'Margin / Axis', '参数 / 模板', '牙冠设计', '检查 / 导出']);
assert.equal(nextStep(S.HOME), S.CASE);
assert.equal(prevStep(S.HOME), null);
assert.equal(nextStep(S.REVIEW), null);
assert.equal(prevStep(S.CASE), S.HOME);

// 2) 进度状态：已完成 / 当前 / 未开始
assert.equal(stepState(S.HOME, S.MARGIN), 'done');
assert.equal(stepState(S.MARGIN, S.MARGIN), 'active');
assert.equal(stepState(S.REVIEW, S.MARGIN), 'todo');

// 3) 准入：往前跳必须满足前置确认；回退永远允许
const nothing = factsFrom({});
assert.equal(gateFor(S.CASE, nothing).ok, true);
assert.equal(gateFor(S.MARGIN, nothing).ok, false);
assert.equal(gateFor(S.SETUP, { marginConfirmed: true }).ok, false, '轴未确认不能进参数页');
assert.equal(gateFor(S.SETUP, { marginConfirmed: true, axisConfirmed: true }).ok, false, '方向未确认不能选牙型');
assert.equal(gateFor(S.SETUP, { marginConfirmed: true, axisConfirmed: true, frameConfirmed: true }).ok, true);
assert.equal(gateFor(S.DESIGN, { profileConfirmed: true }).ok, false, '只应用参数不能跳过自动生成');
assert.equal(gateFor(S.DESIGN, { hasOuter: true }).ok, true);
assert.equal(gateFor(S.REVIEW, { hasOuter: true }).ok, true);
assert.equal(gateFor(S.REVIEW, {}).ok, false);
// 回退不看判据
assert.equal(canVisit(S.CASE, nothing, S.REVIEW).ok, true);
assert.equal(canVisit(S.REVIEW, nothing, S.CASE).ok, false);

// 4) facts 只认服务端事实，不从"设计行存在"推断已确认
const facts = factsFrom({
  caseId: 'c1',
  design: { id: 'd1' },
  assets: { upper: true, lower: true, support: true },
  confirmed: { margin: true, axis: false, profile: false },
  hasOuter: false,
  hasFinal: false,
});
assert.equal(facts.hasCase, true);
assert.equal(facts.hasDesign, true);
assert.equal(facts.marginConfirmed, true);
assert.equal(facts.axisConfirmed, false);
assert.equal(facts.hasOuter, false);

// 5) 进度百分比单调
assert.equal(progressPercent(S.HOME), 0);
assert.equal(progressPercent(S.REVIEW), 100);
assert.ok(progressPercent(S.DESIGN) > progressPercent(S.SETUP));
