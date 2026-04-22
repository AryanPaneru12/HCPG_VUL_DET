/**
 * HCPG-GNN Auditor v4.0 — Pipeline Controller
 * Manages the analysis pipeline UI state and progress
 */

/**
 * Set a pipeline step to a given state
 * @param {number} i - Step index (0-6)
 * @param {string} state - 'pending' | 'active' | 'done' | 'error'
 */
function setStep(i, state) {
  const step = document.getElementById('step' + i);
  const status = document.getElementById('step' + i + 's');
  step.className = 'pipe-step ' + state;
  const labels = { pending: 'PENDING', active: 'RUNNING', done: 'DONE', error: 'FAILED' };
  status.className = 'pipe-status ' + (state === 'active' ? 'running' : state);
  status.textContent = labels[state] || state.toUpperCase();
}

/**
 * Set progress bar percentage
 * @param {number} pct - 0 to 100
 */
function setProgress(pct) {
  document.getElementById('progressFill').style.width = pct + '%';
}

/**
 * Reset all pipeline steps to idle
 */
function resetPipeline() {
  for (let i = 0; i < 7; i++) setStep(i, '');
  setProgress(0);
  document.getElementById('pipelineOverall').textContent = 'IDLE';
  document.getElementById('pipelineOverall').className = 'badge badge-cyan';
}

/**
 * Utility: sleep for ms milliseconds
 * @param {number} ms
 * @returns {Promise}
 */
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
