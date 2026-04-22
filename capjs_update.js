#!/usr/bin/env node
'use strict';

const path = require('path');

// ─── Browser Path Resolution ───────────────────────────────────────────────
// When running as .exe (pkg), resolve browsers/ relative to the .exe location
// When running via `node`, resolve relative to this source file
const browsersPath = process.pkg
  ? path.join(path.dirname(process.execPath), 'browsers')
  : path.join(__dirname, '..', 'browsers');

process.env.PLAYWRIGHT_BROWSERS_PATH = browsersPath;

// ─── NOW import Playwright (after env var is set) ──────────────────────────
const { chromium } = require('playwright');

// ─── Your existing capture logic below ─────────────────────────────────────

(async () => {
  // Example — replace with your actual capture logic
  const browser = await chromium.launch({
    headless: true,
    executablePath: getChromiumPath(browsersPath)
  });

  const page = await browser.newPage();

  // Accept URL from CLI args or config.json
  const targetUrl = process.argv[2] || require('../config.json').url;

  console.log(`Capturing: ${targetUrl}`);
  await page.goto(targetUrl, { waitUntil: 'networkidle' });

  // Save HTML capture
  const html = await page.content();
  const fs = require('fs');
  const outputPath = path.join(path.dirname(process.execPath), 'output.html');
  fs.writeFileSync(outputPath, html, 'utf-8');

  console.log(`✅ Captured HTML saved to: ${outputPath}`);
  await browser.close();
})();

// ─── Helper: Find exact chromium executable inside browsers/ ───────────────
function getChromiumPath(browsersBase) {
  const fs = require('fs');

  // Walk browsers/ to find chrome.exe
  const chromiumDir = fs.readdirSync(browsersBase)
    .find(d => d.startsWith('chromium'));

  if (!chromiumDir) throw new Error('Chromium not found in browsers/ folder');

  return path.join(browsersBase, chromiumDir, 'chrome-win', 'chrome.exe');
}
