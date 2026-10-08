const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewBox = document.getElementById('previewBox');
const predictButton = document.getElementById('predictButton');
const result = document.getElementById('result');
const uploadTitle = document.getElementById('uploadTitle');

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  previewBox.hidden = false;
  predictButton.disabled = false;
  uploadTitle.textContent = file.name;
  result.hidden = true;
});

predictButton.addEventListener('click', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  predictButton.disabled = true;
  result.hidden = false;
  result.innerHTML = '<p>Classifying…</p>';
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
    result.innerHTML = `<h2>${escapeHtml(data.prediction)}</h2><div class="stats"><div class="stat"><strong>${data.confidence}%</strong><span>Confidence</span></div><div class="stat"><strong>${data.latency_ms} ms</strong><span>Inference latency</span></div></div>`;
  } catch (error) {
    result.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  } finally {
    predictButton.disabled = false;
  }
});

document.getElementById('resetButton').addEventListener('click', () => {
  fileInput.value = '';
  preview.removeAttribute('src');
  previewBox.hidden = true;
  predictButton.disabled = true;
  result.hidden = true;
  uploadTitle.textContent = 'Choose an image';
});

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}
