/**
 * HCPG-GNN Auditor v4.0 — Graph Visualization Engine
 * Renders Call Graph, CFG, and HCPG SVG visualizations
 */

/* ---- SVG Helpers -------------------------------------------------------- */

function makeDefs(svg, id, color) {
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', id);
  marker.setAttribute('markerWidth', '8');
  marker.setAttribute('markerHeight', '8');
  marker.setAttribute('refX', '6');
  marker.setAttribute('refY', '3');
  marker.setAttribute('orient', 'auto');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'M0,0 L0,6 L8,3 z');
  path.setAttribute('fill', color);
  marker.appendChild(path);
  defs.appendChild(marker);
  svg.appendChild(defs);
  return defs;
}

/* ---- Call Graph --------------------------------------------------------- */

function drawCallGraph(fns, hasVulns) {
  const svg = document.getElementById('callGraphSvg');
  svg.innerHTML = '';
  if (!fns.length) return;

  makeDefs(svg, 'cg-arrow', hasVulns ? '#ef4444' : '#00e5ff');

  const cx = 150, cy = 115;
  const r = Math.min(80, 180 / Math.max(fns.length, 1));
  const positions = [];

  if (fns.length === 1) {
    positions.push({ x: cx, y: cy });
  } else {
    fns.forEach((f, i) => {
      const angle = (i / fns.length) * Math.PI * 2 - Math.PI / 2;
      positions.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
    });
  }

  const edgeColor = hasVulns ? '#ef444480' : '#00e5ff40';
  if (fns.length > 1) {
    for (let i = 0; i < fns.length - 1; i++) {
      const p1 = positions[i], p2 = positions[i + 1];
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
      line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
      line.setAttribute('stroke', edgeColor);
      line.setAttribute('stroke-width', '1.5');
      line.setAttribute('marker-end', 'url(#cg-arrow)');
      svg.appendChild(line);
    }
  }

  fns.forEach((fname, i) => {
    const { x, y } = positions[i];
    const isVuln = hasVulns && (
      fname.toLowerCase().includes('withdraw') ||
      fname.toLowerCase().includes('drain') ||
      fname.toLowerCase().includes('bid') ||
      fname.toLowerCase().includes('setprice')
    );
    const nodeColor = isVuln ? '#ef4444' : '#00e5ff';

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', x); circle.setAttribute('cy', y); circle.setAttribute('r', 18);
    circle.setAttribute('fill', isVuln ? 'rgba(239,68,68,0.15)' : 'rgba(0,229,255,0.08)');
    circle.setAttribute('stroke', nodeColor); circle.setAttribute('stroke-width', '1.5');
    svg.appendChild(circle);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', x); label.setAttribute('y', y + 1);
    label.setAttribute('text-anchor', 'middle'); label.setAttribute('dominant-baseline', 'middle');
    label.setAttribute('fill', nodeColor); label.setAttribute('font-size', '7.5');
    label.setAttribute('font-family', 'Space Mono');
    label.textContent = fname.length > 10 ? fname.slice(0, 9) : fname;
    svg.appendChild(label);
  });
}

/* ---- Control Flow Graph ------------------------------------------------- */

function drawCFG(nodes, edges, hasVulns) {
  const svg = document.getElementById('cfgSvg');
  svg.innerHTML = '';
  makeDefs(svg, 'cfg-arrow', '#7c3aed');

  const cols = 4;
  const positions = [];
  const count = Math.min(nodes, 20);
  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    positions.push({ x: 28 + col * 68, y: 25 + row * 50 });
  }

  for (let i = 0; i < Math.min(count - 1, edges); i++) {
    const p1 = positions[i], p2 = positions[i + 1];
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y + 8);
    line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y - 8);
    line.setAttribute('stroke', 'rgba(124,58,237,0.5)');
    line.setAttribute('stroke-width', '1.2');
    line.setAttribute('marker-end', 'url(#cfg-arrow)');
    svg.appendChild(line);
  }

  positions.forEach((pos, i) => {
    const isEntry = i === 0;
    const isExit = i === count - 1;
    const isVulnNode = hasVulns && (i === 2 || i === 3);
    const color = isEntry ? '#22c55e' : isExit ? '#f59e0b' : isVulnNode ? '#ef4444' : '#7c3aed';

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', pos.x - 22); rect.setAttribute('y', pos.y - 9);
    rect.setAttribute('width', 44); rect.setAttribute('height', 18);
    rect.setAttribute('rx', 3); rect.setAttribute('fill', color + '18');
    rect.setAttribute('stroke', color); rect.setAttribute('stroke-width', '1');
    svg.appendChild(rect);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x); text.setAttribute('y', pos.y + 1);
    text.setAttribute('text-anchor', 'middle'); text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', color); text.setAttribute('font-size', '7');
    text.setAttribute('font-family', 'Space Mono');
    text.textContent = isEntry ? 'ENTRY' : isExit ? 'EXIT' : 'B' + i;
    svg.appendChild(text);
  });
}

/* ---- HCPG Unified Graph ------------------------------------------------- */

function drawHCPG(fns, cfgCount, hasVulns) {
  const svg = document.getElementById('hcpgSvg');
  svg.innerHTML = '';
  makeDefs(svg, 'hcpg-arrow', '#64748b');

  const fnNodes = fns.slice(0, 5);
  const stmtCount = Math.min(cfgCount, 10);
  const allNodes = [];

  fnNodes.forEach((f, i) => {
    const x = 40 + i * (220 / Math.max(fnNodes.length - 1, 1));
    const y = 40;
    allNodes.push({
      x, y, type: 'fn', label: f.slice(0, 8),
      isVuln: hasVulns && (
        f.toLowerCase().includes('withdraw') ||
        f.toLowerCase().includes('drain') ||
        f.toLowerCase().includes('bid')
      )
    });
  });

  for (let i = 0; i < stmtCount; i++) {
    const x = 20 + i * (260 / Math.max(stmtCount - 1, 1));
    const y = 130 + (i % 2) * 30;
    allNodes.push({ x, y, type: 'stmt', label: 's' + i });
  }

  // Function → Statement edges
  fnNodes.forEach((f, fi) => {
    const fnNode = allNodes[fi];
    for (let si = 0; si < Math.min(3, stmtCount); si++) {
      const stmtNode = allNodes[fnNodes.length + fi * 2 + si];
      if (!stmtNode) continue;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', fnNode.x); line.setAttribute('y1', fnNode.y + 10);
      line.setAttribute('x2', stmtNode.x); line.setAttribute('y2', stmtNode.y - 8);
      line.setAttribute('stroke', 'rgba(100,116,139,0.35)');
      line.setAttribute('stroke-width', '1');
      line.setAttribute('marker-end', 'url(#hcpg-arrow)');
      svg.appendChild(line);
    }
  });

  // Cross-function edge
  if (fnNodes.length > 1) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', allNodes[0].x); line.setAttribute('y1', allNodes[0].y);
    line.setAttribute('x2', allNodes[fnNodes.length - 1].x); line.setAttribute('y2', allNodes[fnNodes.length - 1].y);
    line.setAttribute('stroke', hasVulns ? 'rgba(239,68,68,0.6)' : 'rgba(0,229,255,0.4)');
    line.setAttribute('stroke-width', hasVulns ? '1.5' : '1');
    line.setAttribute('stroke-dasharray', '5,3');
    line.setAttribute('marker-end', 'url(#hcpg-arrow)');
    svg.appendChild(line);
  }

  // Draw nodes
  allNodes.forEach(n => {
    const isFn = n.type === 'fn';
    const color = isFn ? (n.isVuln ? '#ef4444' : '#85C1E9') : '#F7DC6F';
    const r = isFn ? 14 : 9;

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', n.x); circle.setAttribute('cy', n.y); circle.setAttribute('r', r);
    circle.setAttribute('fill', color + '20');
    circle.setAttribute('stroke', color);
    circle.setAttribute('stroke-width', '1.5');
    svg.appendChild(circle);

    if (isFn) {
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', n.x); text.setAttribute('y', n.y + 1);
      text.setAttribute('text-anchor', 'middle'); text.setAttribute('dominant-baseline', 'middle');
      text.setAttribute('fill', color); text.setAttribute('font-size', '6.5');
      text.setAttribute('font-family', 'Space Mono');
      text.textContent = n.label;
      svg.appendChild(text);
    }
  });
}

/* ---- Heatmap ------------------------------------------------------------ */

function generateHeatmap(fns, totalNodes, hasVulns) {
  const grid = document.getElementById('heatmapGrid');
  grid.innerHTML = '';
  const count = Math.min(totalNodes, 60);

  for (let i = 0; i < count; i++) {
    let heat;
    if (hasVulns) {
      const isHot = [2, 3, 8, 9, 14].includes(i);
      heat = isHot ? 0.7 + Math.random() * 0.3 : Math.random() * 0.4;
    } else {
      heat = Math.random() * 0.25 + 0.05;
    }
    const alpha = 0.1 + heat * 0.9;
    const r = Math.round(heat * 255);
    const b = Math.round((1 - heat) * 150);
    const g = Math.round((1 - heat) * 100);

    const cell = document.createElement('div');
    cell.className = 'heatmap-cell';
    cell.style.background = `rgba(${r},${g},${b},${alpha})`;
    cell.style.border = `1px solid rgba(${r},${g},${b},0.3)`;
    grid.appendChild(cell);
  }
}

/* ---- Idle State --------------------------------------------------------- */

function clearGraphs() {
  ['callGraphSvg', 'cfgSvg', 'hcpgSvg'].forEach(id => {
    const svg = document.getElementById(id);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  });
}

function clearHeatmap() {
  document.getElementById('heatmapGrid').innerHTML = '';
}

function drawIdleGraphs() {
  ['callGraphSvg', 'cfgSvg', 'hcpgSvg'].forEach(id => {
    const s = document.getElementById(id);
    s.innerHTML = '';
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', 150); t.setAttribute('y', 115);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('dominant-baseline', 'middle');
    t.setAttribute('fill', '#1e2d45'); t.setAttribute('font-size', '11');
    t.setAttribute('font-family', 'Space Mono');
    t.textContent = 'Run analysis to visualize';
    s.appendChild(t);
  });
}
