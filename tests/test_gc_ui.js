// Run with node tests/test_gc_ui.js; uses only Node's standard library.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
const editor = template.slice(template.indexOf('function checkGcPlotRenderer'),
  template.indexOf('function resetGcPlotSettings'));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value: '', checked: false, disabled: false, src: ''});
  return elements.get(id);
}
let responseSettings, status, displayedSettings;
const context = vm.createContext({
  document: {getElementById: element, querySelectorAll: () => []},
  renderGcPlotColors: settings => { displayedSettings = settings; },
  renderGcDownloadButtons() {}, setGcPlotStatus: message => { status = message; },
  currentGcResult: {plot_token: 'synthetic-session', plot_path: 'previous.png'},
  fetch: async () => ({ok: true, json: async () => ({plot_settings: responseSettings,
    plot_path: 'regenerated.png', plot_b64: 'synthetic-image'})}),
});
vm.runInContext(editor, context);

(async () => {
  const defaults = {title: 'Sample', show_title: true, tick_font_size: 24,
    legend_font_size: 24, title_font_size: 34, axis_font_size: 28};
  context.applyGcPlotSettings(defaults);
  element('gc_plot_show_title').checked = false;
  context.updateGcTitleControls();
  assert.equal(element('gc_plot_title').disabled, true);
  assert.equal(context.collectGcPlotSettings().show_title, false);

  responseSettings = {...defaults, show_title: false};
  await context.regenerateGcPlot();
  assert.equal(element('gc_plot_show_title').checked, false);
  assert.equal(context.collectGcPlotSettings().legend_font_size, 24);
  assert.match(status, /Plot regenerated/);

  // An old server must not silently recheck the title or replace the preview.
  responseSettings = {title: 'Sample', tick_font_size: 16, legend_font_size: 16};
  element('result-plot').src = 'current-preview';
  await context.regenerateGcPlot();
  assert.match(status, /server is older/);
  assert.equal(element('gc_plot_show_title').checked, false);
  assert.equal(element('result-plot').src, 'current-preview');
  assert.equal(displayedSettings.show_title, false);
  assert.throws(() => context.checkGcPlotRenderer(responseSettings), /server is older/);

  // Legacy saved presets still load; compatibility checking is for API results.
  context.applyGcPlotSettings(responseSettings);
  assert.equal(element('gc_plot_show_title').checked, true);
  assert.equal(context.collectGcPlotSettings().legend_font_size, 16);
  console.log('GC UI: title regeneration, 24 px controls, and stale-server detection passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
