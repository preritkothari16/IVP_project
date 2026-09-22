const API = 'http://127.0.0.1:8000';

function imgUrl(root, className, fname) {
  return `${API}/api/dataset/image?dataset_root=${encodeURIComponent(root)}&class_name=${encodeURIComponent(className)}&filename=${encodeURIComponent(fname)}`;
}

// --- Navigation ---
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    link.classList.add('active');
    document.getElementById('page-' + link.dataset.page).classList.add('active');
  });
});

// --- Dataset Explorer ---
async function loadDataset() {
  const root = document.getElementById('explorer-path').value;
  const statsEl = document.getElementById('dataset-stats');
  const gridEl = document.getElementById('dataset-classes');
  statsEl.innerHTML = '';
  gridEl.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

  try {
    const res = await fetch(`${API}/api/classify/dataset-info?preprocessed_root=${encodeURIComponent(root)}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    statsEl.classList.remove('hidden');
    statsEl.innerHTML = `
      <div class="stat-card"><div class="stat-value">${data.num_classes}</div><div class="stat-label">Classes</div></div>
      <div class="stat-card"><div class="stat-value">${data.num_images}</div><div class="stat-label">Images</div></div>
      <div class="stat-card"><div class="stat-value">${data.num_features}</div><div class="stat-label">Features</div></div>
    `;

    gridEl.innerHTML = '';
    for (const [name, count] of Object.entries(data.class_counts)) {
      const card = document.createElement('div');
      card.className = 'class-card';
      card.innerHTML = `<span class="class-name">${name}</span><span class="class-count">${count}</span>`;
      card.onclick = () => showSamples(root, name);
      gridEl.appendChild(card);
    }
  } catch (err) {
    gridEl.innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
  }
}

async function showSamples(root, className) {
  const modal = document.getElementById('sample-modal');
  document.getElementById('modal-title').textContent = className;
  document.getElementById('modal-images').innerHTML = '<p>Loading...</p>';
  modal.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/api/dataset/sample-images?dataset_root=${encodeURIComponent(root)}&class_name=${encodeURIComponent(className)}&limit=12`);
    const data = await res.json();
    const container = document.getElementById('modal-images');
    container.innerHTML = '';
    for (const fname of data.images) {
      const img = document.createElement('img');
      img.src = imgUrl(root, className, fname);
      img.alt = fname;
      img.loading = 'lazy';
      container.appendChild(img);
    }
  } catch (err) {
    document.getElementById('modal-images').innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
  }
}

function closeModal() {
  document.getElementById('sample-modal').classList.add('hidden');
}

// --- Preprocessing Pipeline ---
function handlePipelineFile(input) {
  if (input.files[0]) runPipeline(input.files[0]);
}

function handlePipelineDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  if (e.dataTransfer.files[0]) runPipeline(e.dataTransfer.files[0]);
}

async function runPipeline(file) {
  const stages = document.getElementById('pipeline-stages');
  const loading = document.getElementById('pipeline-loading');
  stages.classList.add('hidden');
  loading.classList.remove('hidden');

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/api/preprocess/single`, { method: 'POST', body: form });
    const data = await res.json();
    const keys = ['original', 'resized', 'filtered', 'enhanced', 'clean_mask', 'masked_output'];
    keys.forEach((key, i) => {
      document.getElementById('stage-' + i).src = 'data:image/png;base64,' + data[key];
    });
    stages.classList.remove('hidden');
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    loading.classList.add('hidden');
  }
}

// --- Classifier ---
function handleClassifyFile(input) {
  if (input.files[0]) runClassify(input.files[0]);
}

function handleClassifyDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  if (e.dataTransfer.files[0]) runClassify(e.dataTransfer.files[0]);
}

async function runClassify(file) {
  const resultEl = document.getElementById('classify-result');
  const loading = document.getElementById('classify-loading');
  const root = document.getElementById('explorer-path').value;
  const k = document.getElementById('k-value').value;
  resultEl.classList.add('hidden');
  loading.classList.remove('hidden');

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/api/classify/single?preprocessed_root=${encodeURIComponent(root)}&k=${k}`, {
      method: 'POST', body: form
    });
    const data = await res.json();
    resultEl.innerHTML = `
      <div class="prediction">${data.prediction}</div>
      <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}% (k=${k})</div>
      <div class="neighbors">
        ${data.neighbors.map(n => `<span class="neighbor-tag">${n}</span>`).join('')}
      </div>
    `;
    resultEl.classList.remove('hidden');
  } catch (err) {
    resultEl.innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
    resultEl.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
  }
}

// --- Results Dashboard ---
async function runEvaluation() {
  const root = document.getElementById('results-path').value;
  const k = document.getElementById('results-k').value;
  const content = document.getElementById('results-content');
  const loading = document.getElementById('results-loading');
  content.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/api/classify/evaluate?preprocessed_root=${encodeURIComponent(root)}&k=${k}`, {
      method: 'POST'
    });
    const data = await res.json();

    document.getElementById('stat-accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('stat-train').textContent = data.train_count;
    document.getElementById('stat-test').textContent = data.test_count;

    renderConfusionMatrix(data.confusion_matrix);
    renderF1Chart(data.per_class_metrics);
    renderMetricsTable(data.per_class_metrics);

    content.classList.remove('hidden');
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    loading.classList.add('hidden');
  }
}

function renderConfusionMatrix(cm) {
  const container = document.getElementById('confusion-matrix');
  const classes = cm.classes;
  const matrix = cm.matrix;
  let html = '<table><tr><th></th>';
  classes.forEach(c => { html += `<th>${c.substring(0, 8)}</th>`; });
  html += '</tr>';
  const maxVal = Math.max(...matrix.flat());
  matrix.forEach((row, i) => {
    html += `<tr><th>${classes[i].substring(0, 8)}</th>`;
    row.forEach(val => {
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const bg = val === 0 ? '#fff' : `rgba(232,93,117,${0.1 + intensity * 0.8})`;
      const color = intensity > 0.5 ? '#fff' : '#333';
      html += `<td style="background:${bg};color:${color}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</table>';
  container.innerHTML = html;
}

function renderF1Chart(metrics) {
  const canvas = document.getElementById('f1-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 300 * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = '300px';
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = 300;
  const padding = { top: 20, right: 20, bottom: 120, left: 50 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;

  ctx.clearRect(0, 0, w, h);

  const sorted = [...metrics].sort((a, b) => b.f1 - a.f1);
  const barWidth = Math.max(4, chartW / sorted.length - 2);

  ctx.fillStyle = '#636e72';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 10; i++) {
    const y = padding.top + chartH - (i / 10) * chartH;
    ctx.fillText((i / 10).toFixed(1), padding.left - 8, y + 4);
    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  sorted.forEach((m, i) => {
    const x = padding.left + i * (barWidth + 2);
    const barH = m.f1 * chartH;
    const y = padding.top + chartH - barH;
    const hue = m.f1 > 0.7 ? 350 : m.f1 > 0.4 ? 30 : 180;
    ctx.fillStyle = `hsl(${hue}, 70%, 55%)`;
    ctx.fillRect(x, y, barWidth, barH);

    ctx.save();
    ctx.translate(x + barWidth / 2, h - padding.bottom + 8);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#636e72';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(m.class.substring(0, 14), 0, 0);
    ctx.restore();
  });
}

function renderMetricsTable(metrics) {
  const container = document.getElementById('metrics-table');
  let html = '<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>';
  const sorted = [...metrics].sort((a, b) => b.f1 - a.f1);
  sorted.forEach(m => {
    const color = m.f1 > 0.7 ? 'green' : m.f1 > 0.4 ? 'orange' : 'red';
    html += `<tr>
      <td><strong>${m.class}</strong></td>
      <td>${(m.precision * 100).toFixed(1)}%</td>
      <td>${(m.recall * 100).toFixed(1)}%</td>
      <td style="color:${color};font-weight:600">${(m.f1 * 100).toFixed(1)}%</td>
      <td>${m.support}</td>
    </tr>`;
  });
  html += '</table>';
  container.innerHTML = html;
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  loadDataset();
});
