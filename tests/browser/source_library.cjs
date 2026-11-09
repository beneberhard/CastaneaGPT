// Run against the application's emitted scripts/data; no browser or model calls.
// Usage: node tests/browser/source_library.cjs /tmp/map.html /tmp/sources.json
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.handlers = {}; this.value = ''; this.style = {}; this.open = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this.ownText = ''; }
  set textContent(text) { this.ownText = text; this.children = []; }
  get textContent() { return (this.ownText || '') + this.children.map(child => child.textContent || '').join(''); }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  focus() {}
}
const html = fs.readFileSync(process.argv[2], 'utf8');
const sources = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const elements = {};
const document = {
  getElementById: id => elements[id] ||= new Element(),
  createElement: tag => new Element(tag),
  createTextNode: text => ({textContent: text}),
};
const context = vm.createContext({document, fetch: async () => ({ok: true, json: async () => sources}), console});
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
const library = scripts.find(script => script.includes("const dialog = document.getElementById('sourceLibrary')"));
assert.ok(library, 'library script must be present in the served map');
vm.runInContext(library, context);
(async () => {
  await elements.openSourceLibrary.handlers.click();
  assert.equal(elements.sourceLibrary.open, true);
  assert.equal(elements.sourceLibraryList.children.length, 69);
  assert.match(elements.sourceLibraryList.children[0].textContent, /^Agnoletti/);
  elements.sourceLibrarySearch.value = 'Mariotti';
  elements.sourceLibrarySearch.handlers.input();
  assert.ok(elements.sourceLibraryList.children.length > 0);
  assert.ok(elements.sourceLibraryList.children.length < 69);
  assert.ok(elements.sourceLibraryList.children.every(item => item.textContent.toLowerCase().includes('mariotti')));
  elements.closeSourceLibrary.handlers.click();
  assert.equal(elements.sourceLibrary.open, false);
  const main = scripts[0];
  const start = main.indexOf('function renderCompletedAnswer(');
  const end = main.indexOf('document.getElementById("askBtn")', start);
  vm.runInContext(main.slice(start, end), context);
  const answer = new Element();
  context.renderCompletedAnswer(answer, 'Evidence <script>alert(1)</script> (Mariotti, 2009)\nhttps://doi.org/10.4129/CNS2008.113\nNext line');
  assert.match(answer.textContent, /<script>alert\(1\)<\/script>/);
  const links = answer.children.filter(child => child.tag === 'a');
  assert.equal(links.length, 1);
  assert.equal(links[0].href, 'https://doi.org/10.4129/CNS2008.113');
  assert.equal(links[0].rel, 'noopener noreferrer');
  console.log('Source library checks passed: open, 69 sources, alphabetic order, search, close, safe publication links.');
})().catch(error => { console.error(error); process.exit(1); });
