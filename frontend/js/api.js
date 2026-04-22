/**
 * HCPG-GNN Auditor v4.0 — API Communication Layer
 * Handles all backend communication
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : window.location.origin;

/**
 * Analyze a smart contract via the backend API
 * @param {string} code - Solidity source code
 * @returns {Object|null} Analysis response or null if API unavailable
 */
async function callAPI(code) {
  try {
    const response = await fetch(API_BASE + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_code: code })
    });
    if (!response.ok) throw new Error('API request failed: ' + response.status);
    return await response.json();
  } catch (e) {
    console.warn('API not available, using local detection:', e.message);
    return null;
  }
}

/**
 * Check backend API health
 * @returns {boolean}
 */
async function checkAPIHealth() {
  try {
    const resp = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(3000) });
    const data = await resp.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}

/**
 * Fetch model info from backend
 * @returns {Object|null}
 */
async function fetchModelInfo() {
  try {
    const resp = await fetch(API_BASE + '/api/model/info');
    return await resp.json();
  } catch {
    return null;
  }
}

/**
 * Upload a .sol file for analysis
 * @param {File} file
 * @returns {Object|null}
 */
async function uploadFile(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(API_BASE + '/api/analyze/file', {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) throw new Error('Upload failed');
    return await resp.json();
  } catch (e) {
    console.warn('File upload failed:', e.message);
    return null;
  }
}
