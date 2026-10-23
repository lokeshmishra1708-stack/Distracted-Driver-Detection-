(() => {
  const form = document.querySelector('#predict-form');
  const input = document.querySelector('#image-input');
  const zone = document.querySelector('#drop-zone');
  const error = document.querySelector('#upload-error');
  const button = document.querySelector('#predict-button');
  const preview = document.querySelector('#preview-wrap');
  let previewUrl;

  function showFile(file) {
    error.textContent = '';
    if (!file) return;
    if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 8 * 1024 * 1024) {
      error.textContent = file.size > 8 * 1024 * 1024 ? 'This image exceeds the 8 MB upload limit.' : 'Choose a JPG or PNG image.';
      input.value = '';
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    document.querySelector('#preview-image').src = previewUrl;
    document.querySelector('#preview-name').textContent = file.name;
    preview.hidden = false;
  }

  input?.addEventListener('change', () => showFile(input.files[0]));
  document.querySelector('#remove-image')?.addEventListener('click', () => {
    input.value = '';
    preview.hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
  ['dragenter', 'dragover'].forEach(type => zone?.addEventListener(type, event => {
    event.preventDefault(); zone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(type => zone?.addEventListener(type, event => {
    event.preventDefault(); zone.classList.remove('dragging');
  }));
  zone?.addEventListener('drop', event => {
    if (event.dataTransfer.files.length) {
      try { input.files = event.dataTransfer.files; } catch { /* File selection still works on browsers that disallow assignment. */ }
      showFile(event.dataTransfer.files[0]);
    }
  });
  form?.addEventListener('submit', async event => {
    event.preventDefault();
    if (!input.files.length) { error.textContent = 'Choose an image first.'; return; }
    button.disabled = true;
    button.querySelector('span').textContent = 'Analyzing…';
    error.textContent = '';
    try {
      const response = await fetch('/predict', { method: 'POST', body: new FormData(form) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Prediction failed.');
      renderResult(result);
      addHistory(input.files[0].name, result);
    } catch (exception) {
      error.textContent = exception.message || 'Unable to connect to the prediction service.';
    } finally {
      button.disabled = false;
      button.querySelector('span').textContent = 'Analyze behavior';
    }
  });

  function renderResult(result) {
    document.querySelector('#empty-results').hidden = true;
    const panel = document.querySelector('#result-content');
    panel.hidden = false;
    document.querySelector('#result-code').textContent = result.class_code;
    document.querySelector('#result-name').textContent = result.class_name;
    document.querySelector('#result-confidence').textContent = `${(result.probability * 100).toFixed(1)}%`;
    const list = document.querySelector('#probability-list');
    list.replaceChildren();
    result.probabilities.forEach(item => {
      const row = document.createElement('div'); row.className = 'prob-row';
      const code = document.createElement('span'); code.className = 'prob-code'; code.textContent = item.code;
      const track = document.createElement('div'); track.className = 'prob-track';
      const fill = document.createElement('div'); fill.className = 'prob-fill'; fill.style.width = `${Math.max(0, Math.min(100, item.probability * 100))}%`;
      track.append(fill);
      const number = document.createElement('span'); number.className = 'prob-number'; number.textContent = `${(item.probability * 100).toFixed(1)}%`;
      row.append(code, track, number); list.append(row);
    });
  }

  function addHistory(filename, result) {
    const list = document.querySelector('#history-list');
    list.querySelector('.history-empty')?.remove();
    const row = document.createElement('div'); row.className = 'history-item';
    const code = document.createElement('span'); code.className = 'history-code'; code.textContent = result.class_code;
    const name = document.createElement('span'); name.textContent = result.class_name;
    const file = document.createElement('span'); file.className = 'muted'; file.textContent = filename;
    row.append(code, name, file); list.prepend(row);
  }

  document.querySelector('#clear-history')?.addEventListener('click', async () => {
    try {
      await fetch('/history/clear', { method: 'POST' });
      const list = document.querySelector('#history-list'); list.replaceChildren();
      const message = document.createElement('p'); message.className = 'muted history-empty'; message.textContent = 'No predictions in this session.'; list.append(message);
    } catch { /* Keep the UI responsive if history clearing fails. */ }
  });

  fetch('/metrics').then(response => response.json()).then(payload => {
    if (!payload.available) return;
    const source = payload.metrics;
    const values = [['test_accuracy', 'TEST ACCURACY'], ['macro_precision', 'MACRO PRECISION'], ['macro_recall', 'MACRO RECALL'], ['macro_f1', 'MACRO F1'], ['test_images', 'TEST IMAGES']];
    const container = document.querySelector('#metrics-content'); container.replaceChildren();
    values.forEach(([key, label]) => {
      if (source[key] === undefined) return;
      const tile = document.createElement('div'); tile.className = 'metric-tile';
      const value = document.createElement('strong'); value.textContent = key === 'test_images' ? String(source[key]) : `${(source[key] * 100).toFixed(1)}%`;
      const caption = document.createElement('span'); caption.textContent = label;
      tile.append(value, caption); container.append(tile);
    });
    const charts = document.querySelector('#evaluation-charts');
    let loaded = 0;
    const modelName = source.model_name === 'mobilenetv2' ? 'mobilenetv2' : 'custom_cnn';
    charts.querySelectorAll('img').forEach((image, index) => {
      if (index < 2) image.src = `/figures/${modelName}_${index === 0 ? 'accuracy' : 'loss'}.png`;
    });
    charts.querySelectorAll('img').forEach(image => {
      image.addEventListener('load', () => { loaded += 1; charts.hidden = loaded === 0; }, { once: true });
      image.addEventListener('error', () => image.remove(), { once: true });
    });
  }).catch(() => {});
})();
