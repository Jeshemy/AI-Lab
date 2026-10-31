const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function showError(message) {
  const box = $('#error-message');
  if (box) {
    box.hidden = false;
    box.textContent = message;
  }
}

function clearError() {
  const box = $('#error-message');
  if (box) {
    box.hidden = true;
    box.textContent = '';
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function download(name, text) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function sendJson(url, data) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Could not analyze the session.');
  return result;
}

function makeFactorRow(labelText, widthPercent) {
  const row = document.createElement('div');
  row.className = 'factor-row';

  const name = document.createElement('span');
  name.textContent = labelText;

  const track = document.createElement('span');
  track.className = 'factor-track';

  const fill = document.createElement('i');
  fill.style.width = Math.max(6, widthPercent) + '%';

  track.append(fill);
  row.append(name, track);
  return row;
}

function renderResult(data) {
  $('#empty-result').hidden = true;
  $('#result-content').hidden = false;

  const label = $('#result-label');
  label.textContent = data.prediction;
  label.className = 'result-label ' + data.prediction.toLowerCase();

  $('#confidence').textContent = Math.round(data.probability * 100) + '%';
  $('#meter-fill').style.width = Math.round(data.probability * 100) + '%';
  $('#explanation').textContent = data.explanation;

  const importanceList = $('#importance-list');
  importanceList.replaceChildren();
  for (const item of data.feature_importance || []) {
    importanceList.append(makeFactorRow(item.label, Math.round(item.importance * 100)));
  }

  const factorList = $('#factor-list');
  factorList.replaceChildren();
  if (data.factors && data.factors.length) {
    for (const factor of data.factors) {
      factorList.append(
        makeFactorRow(`${factor.label} · ${factor.direction} attack score`, factor.strength)
      );
    }
  } else {
    const empty = document.createElement('p');
    empty.className = 'factor-empty';
    empty.textContent = 'No individual field produced a measurable score shift.';
    factorList.append(empty);
  }
}

if (document.body.dataset.page === 'predict') {

  $$('.tab').forEach((button) =>
    button.addEventListener('click', () => {
      $$('.tab').forEach((tab) => tab.classList.toggle('active', tab === button));
      $$('.tab-pane').forEach((pane) =>
        pane.classList.toggle(
          'active',
          pane.id === (button.dataset.tab === 'manual' ? 'manual-pane' : 'upload-pane')
        )
      );
      clearError();
    })
  );

  $('#session-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();

    const form = new FormData(event.currentTarget);
    const data = Object.fromEntries(form.entries());

    try {
      renderResult(await sendJson('/api/predict', data));
    } catch (err) {
      showError(err.message);
    }
  });

  $('#session-form').addEventListener('reset', () => {
    setTimeout(() => {
      $('#result-content').hidden = true;
      $('#empty-result').hidden = false;
      clearError();
    }, 0);
  });

  $('#clear-result').addEventListener('click', () => {
    $('#result-content').hidden = true;
    $('#empty-result').hidden = false;
    clearError();
  });

  const input = $('#csv-file');
  const zone = $('#drop-zone');
  let selectedFile = null;
  let inspected = null;

  input.addEventListener('change', () => input.files[0] && inspectFile(input.files[0]));

  for (const eventName of ['dragenter', 'dragover']) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add('drag');
    });
  }

  for (const eventName of ['dragleave', 'drop']) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove('drag');
    });
  }

  zone.addEventListener('drop', (event) =>
    event.dataTransfer.files[0] && inspectFile(event.dataTransfer.files[0])
  );

  async function inspectFile(file) {
    clearError();
    selectedFile = file;
    $('#mapping-panel').hidden = true;
    $('#file-status').textContent = 'Reading columns in ' + file.name + '…';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/inspect-csv', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Could not read this CSV.');

      inspected = data;
      $('#file-status').textContent = '';
      $('#mapping-panel').hidden = false;
      $('#file-meta').textContent =
        `${file.name} · ${data.total.toLocaleString()} rows · ${data.headers.length} columns`;

      renderMapping(data);
      renderPreview(data);
    } catch (err) {
      $('#file-status').textContent = '';
      showError(err.message);
    }
  }

  const modelFields = [
    ['login_attempts', 'Login attempts'],
    ['failed_logins', 'Failed logins'],
    ['session_duration', 'Session duration'],
    ['network_packet_size', 'Packet size'],
    ['ip_reputation_score', 'IP reputation score'],
    ['protocol_type', 'Protocol'],
    ['encryption_used', 'Encryption'],
    ['browser_type', 'Browser type'],
    ['unusual_time_access', 'Unusual time access'],
  ];

  function renderMapping(data) {
    const grid = $('#mapping-grid');
    grid.replaceChildren();

    for (const [key, label] of modelFields) {
      const wrap = document.createElement('label');
      wrap.className = 'mapping-item';

      const title = document.createElement('span');
      title.textContent = label;

      const select = document.createElement('select');
      select.dataset.field = key;
      select.append(new Option('Not mapped', ''));
      select.append(new Option('— skip this feature —', ''));
      for (const header of data.headers) select.append(new Option(header, header));
      select.value = data.auto_mapping[key] || '';

      wrap.append(title, select);
      grid.append(wrap);
    }

    const count = Object.keys(data.auto_mapping).length;
    $('#mapping-note').textContent = count
      ? `${count} column${count === 1 ? '' : 's'} matched automatically. Review or change the mapping before analysis.`
      : 'No familiar session columns found. Choose which CSV columns correspond to the session features, or analyze without mapping to preserve and export the rows without assigning unsupported scores.';
  }

  function renderPreview(data) {
    const box = $('#csv-preview');
    const headers = data.headers.slice(0, 6);
    const extra = data.headers.length - headers.length;

    box.innerHTML = `
      <div class="preview-title">File preview
        <span>first ${data.preview.length} rows${extra > 0 ? ` · ${extra} more columns in file` : ''}</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${data.preview.map((row) =>
              `<tr>${headers.map((h) => `<td>${escapeHtml(row[h] || '')}</td>`).join('')}</tr>`
            ).join('')}
          </tbody>
        </table>
      </div>`;
  }

  $('#analyze-csv').addEventListener('click', async () => {
    if (!selectedFile) return;
    clearError();

    const button = $('#analyze-csv');
    button.disabled = true;
    button.textContent = 'Analyzing…';

    const formData = new FormData();
    formData.append('file', selectedFile);

    const mapping = {};
    $$('#mapping-grid select').forEach((select) => {
      if (select.value) mapping[select.dataset.field] = select.value;
    });
    formData.append('mapping', JSON.stringify(mapping));

    try {
      const response = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Could not analyze this CSV.');

      fillBatch(data);
      $('#file-status').textContent = `Analysis complete · ${data.total.toLocaleString()} rows`;
    } catch (err) {
      showError(err.message);
    } finally {
      button.disabled = false;
      button.innerHTML = 'Analyze CSV <span>→</span>';
    }
  });

  $('#change-csv').addEventListener('click', () => {
    selectedFile = null;
    inspected = null;
    input.value = '';
    $('#mapping-panel').hidden = true;
    $('#file-status').textContent = '';
    clearError();
    input.click();
  });

  const batchPageSize = 25;
  let batchPage = 1;

  function renderBatchPage() {
    const data = window.batchData;
    if (!data) return;

    const start = (batchPage - 1) * batchPageSize;
    const end = Math.min(start + batchPageSize, data.results.length);
    const shown = data.headers.slice(0, 5);
    const more = data.headers.length - shown.length;
    const pages = Math.max(1, Math.ceil(data.results.length / batchPageSize));

    $('#batch-body').innerHTML = data.results.slice(start, end).map((r) => `
      <tr>
        <td>${r.row}</td>
        ${shown.map((h) => `<td>${escapeHtml(r.source[h] || '')}</td>`).join('')}
        ${more ? `<td>${more} more</td>` : ''}
        <td>
          <span class="badge ${r.prediction === 'Needs mapping' ? 'unscored' : r.prediction.toLowerCase()}">
            ${escapeHtml(r.prediction)}
          </span>
        </td>
        <td>${r.probability == null ? '—' : Math.round(r.probability * 100) + '%'}</td>
        <td>${escapeHtml(r.explanation)}</td>
      </tr>`).join('');

    $('#batch-page-input').value = batchPage;
    $('#batch-page-input').max = pages;
    $('#batch-page-count').textContent = pages.toLocaleString();
    $('#batch-prev').disabled = batchPage <= 1;
    $('#batch-next').disabled = batchPage >= pages;
    $('#batch-range').textContent =
      ` · rows ${start + 1}–${end} of ${data.results.length.toLocaleString()}`;
  }

  function fillBatch(data) {
    const panel = $('#batch-results');
    panel.hidden = false;

    $('#batch-count').textContent = `${data.total.toLocaleString()} rows`;
    $('#batch-summary').textContent =
      `${data.attacks} Attack · ${data.normals} Normal · ${data.unmapped} unscored (no mapped features). Demonstration estimates only.`;

    const shown = data.headers.slice(0, 5);
    const more = data.headers.length - shown.length;

    $('#batch-head').innerHTML = `
      <tr>
        <th>ROW</th>
        ${shown.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}
        ${more ? '<th>OTHER COLUMNS</th>' : ''}
        <th>SCREENING</th>
        <th>CONFIDENCE</th>
        <th>KEY SIGNALS</th>
      </tr>`;

    window.batchData = data;
    batchPage = 1;
    renderBatchPage();
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  $('#batch-prev').addEventListener('click', () => {
    if (batchPage > 1) {
      batchPage--;
      renderBatchPage();
    }
  });

  $('#batch-next').addEventListener('click', () => {
    if (window.batchData && batchPage < Math.ceil(window.batchData.results.length / batchPageSize)) {
      batchPage++;
      renderBatchPage();
    }
  });

  $('#batch-page-input').addEventListener('change', (event) => {
    if (!window.batchData) return;
    const pages = Math.max(1, Math.ceil(window.batchData.results.length / batchPageSize));
    batchPage = Math.min(pages, Math.max(1, Math.floor(Number(event.target.value) || 1)));
    renderBatchPage();
  });

  $('#download-results').addEventListener('click', () => {
    const data = window.batchData;
    if (!data) return;

    const headers = [
      ...data.headers,
      'safe_flow_prediction',
      'safe_flow_confidence',
      'safe_flow_attack_probability',
      'safe_flow_explanation',
    ];

    const quote = (value) => {
      let text = String(value ?? '');
      if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
      return `"${text.replaceAll('"', '""')}"`;
    };

    const lines = [
      headers.map(quote).join(','),
      ...data.results.map((r) =>
        [
          ...data.headers.map((h) => r.source[h] ?? ''),
          r.prediction,
          r.probability ?? '',
          r.attack_probability ?? '',
          r.explanation,
        ].map(quote).join(',')
      ),
    ];

    download('safe-flow-results.csv', '\ufeff' + lines.join('\r\n'));
  });

  $('#download-sample').addEventListener('click', () =>
    download(
      'safe-flow-sample.csv',
      'login_attempts,failed_logins,session_duration,network_packet_size,ip_reputation_score,protocol_type,encryption_used,browser_type,unusual_time_access\n' +
      '3,0,1558,472,0.30,TCP,DES,Firefox,0\n' +
      '4,1,493,599,0.61,TCP,DES,Edge,0\n'
    )
  );
}