---
layout: default
title: Evo-Architect Dashboard
---

<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .dashboard { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .card { border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; margin: 10px 0; background: #fff; }
  .card h3 { margin-top: 0; color: #24292e; }
  .metric { display: inline-block; margin: 10px 20px 10px 0; }
  .metric .value { font-size: 2em; font-weight: bold; color: #0366d6; }
  .metric .label { font-size: 0.9em; color: #586069; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e1e4e8; }
  th { background: #f6f8fa; font-weight: 600; }
  .status-merged { color: #22863a; }
  .status-discarded { color: #cb2431; }
  #qd-grid { display: grid; grid-template-columns: repeat(3, 80px); gap: 4px; margin: 10px 0; }
  .qd-cell { width: 80px; height: 60px; border: 1px solid #e1e4e8; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.8em; }
  .qd-cell.filled { background: #dcffe4; border-color: #34d058; }
  .qd-cell.empty { background: #f6f8fa; }
  #playground { border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; margin: 10px 0; background: #fafbfc; }
  #playground textarea { width: 100%; height: 80px; font-family: monospace; padding: 8px; border: 1px solid #d1d5da; border-radius: 4px; }
  #playground .output { background: #fff; border: 1px solid #d1d5da; border-radius: 4px; padding: 12px; margin-top: 10px; min-height: 40px; font-family: monospace; white-space: pre-wrap; }
  canvas { max-width: 100%; height: 300px; }
</style>

<div class="dashboard">

# 🧬 Evo-Architect Dashboard

**Continuous Dynamic Liquid Engine (CDLE)** — Live evolution status

---

<div class="card">
<h3>📊 Current Status</h3>
<div id="status-metrics">
  <div class="metric"><div class="value" id="generation">—</div><div class="label">Generation</div></div>
  <div class="metric"><div class="value" id="best-lpw">—</div><div class="label">Best Loss/Watt</div></div>
  <div class="metric"><div class="value" id="best-val-loss">—</div><div class="label">Best Val Loss</div></div>
  <div class="metric"><div class="value" id="qd-coverage">—</div><div class="label">QD Coverage</div></div>
</div>
</div>

<div class="card">
<h3>🏆 Evolution Leaderboard</h3>
<table id="leaderboard">
  <thead>
    <tr>
      <th>Gen</th><th>Verdict</th><th>Val Loss</th><th>Loss/Watt</th><th>Params</th><th>Stability</th>
    </tr>
  </thead>
  <tbody id="leaderboard-body">
    <tr><td colspan="6"><em>Loading data...</em></td></tr>
  </tbody>
</table>
</div>

<div class="card">
<h3>📈 Evolution Plot</h3>
<canvas id="evolution-chart"></canvas>
</div>

<div class="card">
<h3>🗺️ QD Archive (MAP-Elites)</h3>
<p>Species diversity across complexity (rows) × sparsity (columns):</p>
<div id="qd-grid"></div>
</div>

<div class="card" id="playground">
<h3>🎮 Model Playground</h3>
<p>Try byte-level text generation (demo — runs in browser with mock inference):</p>
<textarea id="input-text" placeholder="Type some text here...">Once upon a time</textarea>
<button onclick="generateText()">Generate →</button>
<div class="output" id="output-text">Click "Generate" to see mock output...</div>
<p style="font-size: 0.8em; color: #586069;">
  Note: This is a static demo. For real inference, download the best model checkpoint
  and run locally with <code>python models/cdle_base.py</code>.
</p>
</div>

<div class="card">
<h3>🔗 Links</h3>
<ul>
  <li><a href="https://github.com/Rishu2023/Evo-Architect">📦 Repository</a></li>
  <li><a href="https://github.com/Rishu2023/Evo-Architect/actions">⚡ Actions (CI/CD)</a></li>
  <li><a href="https://github.com/Rishu2023/Evo-Architect/issues/new?template=steer-evolution.md">🎯 Steer Evolution (Issue)</a></li>
</ul>
</div>

</div>

<script>
// ===== Dashboard Logic =====

// Load evolutionary memory data
async function loadData() {
  try {
    const memResp = await fetch('evolutionary_memory.json');
    if (memResp.ok) {
      const memory = await memResp.json();
      updateStatus(memory);
      updateLeaderboard(memory);
      updateChart(memory);
    }
  } catch (e) { console.log('Memory data not yet available:', e); }

  try {
    const qdResp = await fetch('qd_population.json');
    if (qdResp.ok) {
      const qd = await qdResp.json();
      updateQDGrid(qd);
    }
  } catch (e) { console.log('QD data not yet available:', e); }
}

function updateStatus(memory) {
  document.getElementById('generation').textContent = memory.generation || 0;
  document.getElementById('best-lpw').textContent =
    memory.best_loss_per_watt ? memory.best_loss_per_watt.toFixed(6) : '—';
  document.getElementById('best-val-loss').textContent =
    memory.best_val_loss ? memory.best_val_loss.toFixed(4) : '—';
}

function updateLeaderboard(memory) {
  const tbody = document.getElementById('leaderboard-body');
  const history = (memory.history || []).slice(-10).reverse();

  if (history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6"><em>No generations yet. Run the pipeline!</em></td></tr>';
    return;
  }

  tbody.innerHTML = history.map(h => `
    <tr>
      <td>${h.generation || '?'}</td>
      <td class="status-${h.verdict || 'pending'}">${(h.verdict || h.status || '?').toUpperCase()}</td>
      <td>${h.val_loss ? h.val_loss.toFixed(4) : '—'}</td>
      <td>${h.loss_per_watt ? h.loss_per_watt.toFixed(6) : '—'}</td>
      <td>${h.param_count ? h.param_count.toLocaleString() : '—'}</td>
      <td>${h.stability_score !== undefined ? h.stability_score.toFixed(2) : '—'}</td>
    </tr>
  `).join('');
}

function updateChart(memory) {
  const canvas = document.getElementById('evolution-chart');
  const ctx = canvas.getContext('2d');
  const history = memory.history || [];

  // Simple line chart of loss_per_watt over generations
  const data = history.filter(h => h.loss_per_watt).map(h => ({
    x: h.generation,
    y: h.loss_per_watt
  }));

  if (data.length < 2) {
    ctx.font = '14px sans-serif';
    ctx.fillStyle = '#586069';
    ctx.fillText('Not enough data for chart yet. Run a few generations!', 20, 150);
    return;
  }

  const w = canvas.width = canvas.offsetWidth;
  const h = canvas.height = 300;
  const pad = 50;

  const xs = data.map(d => d.x);
  const ys = data.map(d => d.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys) * 0.9, yMax = Math.max(...ys) * 1.1;

  ctx.clearRect(0, 0, w, h);

  // Axes
  ctx.strokeStyle = '#e1e4e8';
  ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#586069';
  ctx.font = '12px sans-serif';
  ctx.fillText('Loss/Watt', 5, pad - 10);
  ctx.fillText('Generation', w - pad - 40, h - 10);

  // Data line
  ctx.strokeStyle = '#0366d6';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d, i) => {
    const px = pad + (d.x - xMin) / Math.max(xMax - xMin, 1) * (w - 2 * pad);
    const py = h - pad - (d.y - yMin) / Math.max(yMax - yMin, 1e-9) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Data points
  ctx.fillStyle = '#0366d6';
  data.forEach(d => {
    const px = pad + (d.x - xMin) / Math.max(xMax - xMin, 1) * (w - 2 * pad);
    const py = h - pad - (d.y - yMin) / Math.max(yMax - yMin, 1e-9) * (h - 2 * pad);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function updateQDGrid(qd) {
  const grid = document.getElementById('qd-grid');
  const niches = qd.niches || {};
  const meta = qd.metadata || {};
  const cBins = meta.complexity_bins || 4;
  const sBins = meta.sparsity_bins || 3;

  grid.style.gridTemplateColumns = `repeat(${sBins}, 80px)`;

  let filled = 0;
  let html = '';
  for (let r = 0; r < cBins; r++) {
    for (let c = 0; c < sBins; c++) {
      const key = `(${r}, ${c})`;
      const niche = niches[key];
      if (niche) {
        filled++;
        html += `<div class="qd-cell filled" title="Gen ${niche.generation || '?'}: loss=${niche.val_loss || '?'}">
          G${niche.generation || '?'}<br>${niche.val_loss ? niche.val_loss.toFixed(2) : '—'}
        </div>`;
      } else {
        html += `<div class="qd-cell empty">empty</div>`;
      }
    }
  }
  grid.innerHTML = html;

  const coverage = filled / (cBins * sBins);
  document.getElementById('qd-coverage').textContent = (coverage * 100).toFixed(0) + '%';
}

function generateText() {
  const input = document.getElementById('input-text').value;
  const output = document.getElementById('output-text');

  // Mock byte-level "generation" for demo purposes
  const chars = 'abcdefghijklmnopqrstuvwxyz .,!?';
  let generated = input;
  for (let i = 0; i < 50; i++) {
    generated += chars[Math.floor(Math.random() * chars.length)];
  }
  output.textContent = generated;
}

// Load data on page load
loadData();
</script>
