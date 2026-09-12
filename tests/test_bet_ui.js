// Run with node tests/test_bet_ui.js; uses only Node's standard library.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
const ui = template.slice(template.indexOf('// ── BET UI'), template.indexOf('function openXrdFittingGuide'));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {value: '', textContent: '', innerHTML: '', style: {}, disabled: false,
      classList: {add: name => classes.add(name), remove: name => classes.delete(name),
        contains: name => classes.has(name), toggle: (name, on) => on ? classes.add(name) : classes.delete(name)},
      addEventListener() {}, scrollIntoView() {}});
  }
  return elements.get(id);
}
let requests = [], displayedStats;
let reply = () => Promise.resolve({ok: true, json: async () => ({sample_id: 'Synthetic',
  sample_mass_g: .1, mass_normalization: {source_mass_g: .1, note: 'Calculation mass: 0.1 g.'},
  surface_area_m2_g: 24, source_surface_area_m2_g: 22, plot_token: 'test',
  output_dir: 'test', plot_path: 'plot.png', summary_path: 'summary.xlsx'})});
const context = vm.createContext({
  document: {getElementById: element}, betSelectedFile: null, currentBetResult: null,
  charSetFile: () => true, charSig: String, charFormat: String, escHtml: String, escPath: String,
  charStats: (_, stats) => { displayedStats = stats; }, charNumber: id => Number(element(id).value),
  FormData: class { constructor() { this.fields = {}; } append(k, v) { this.fields[k] = v; } },
  fetch: (...args) => { requests.push(args); return reply(); },
});
vm.runInContext(ui, context);

(async () => {
  const firstFile = {name: 'first.xlsx'};
  context.selectBETFile(firstFile);
  await context.processBET();
  assert.equal(element('bet_sample_mass_g').value, '', 'File mass must not become an override');
  assert.equal(displayedStats[0].label, 'Calculated BET area');
  assert.ok(!displayedStats.some(stat => stat.label.includes('reported')));
  assert.equal(context.betFitIsStale(), false);
  assert.match(element('bet-btn-process').innerHTML, /Recalculate BET/);

  element('bet_sample_mass_g').value = '.2';
  context.updateBETFitState();
  assert.equal(context.betFitIsStale(), true);
  assert.ok(element('bet-fit-stale').classList.contains('visible'));
  assert.equal(element('bet-regenerate').disabled, true);
  await context.regenerateBETPlot();
  assert.equal(requests.length, 1, 'Stale fits cannot be regenerated as if current');
  await context.processBET();
  assert.equal(requests[1][1].body.fields.sample_mass_g, '.2');
  assert.equal(context.betFitIsStale(), false);

  element('bet_plot_x_min').value = '.05';
  assert.equal(context.betFitIsStale(), false, 'Cosmetic axis limits do not change fit inputs');
  element('bet_p_min').value = '.05';
  assert.equal(context.betFitIsStale(), true);

  // Editing a field while the request is pending must leave a stale notice.
  let resolve;
  const defaultReply = reply;
  reply = () => new Promise(done => { resolve = done; });
  const pending = context.processBET();
  element('bet_p_max').value = '.3';
  resolve(await defaultReply());
  await pending;
  assert.equal(context.betFitIsStale(), true);

  // Selecting another file cannot inherit the previous sample's mass override.
  const secondFile = {name: 'second.xlsx'};
  context.selectBETFile(secondFile);
  assert.equal(element('bet_sample_mass_g').value, '');
  assert.equal(context.currentBetResult, null);
  assert.ok(!element('bet-results').classList.contains('visible'));

  // An old response arriving after file selection must not restore old results.
  const oldRequest = context.processBET();
  context.selectBETFile(firstFile);
  resolve(await defaultReply());
  await oldRequest;
  assert.equal(context.currentBetResult, null);
  assert.ok(!element('bet-results').classList.contains('visible'));
  console.log('BET UI: recalculation, stale settings, fixed report, and file switching passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
