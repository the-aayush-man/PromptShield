const colors = {
  safe: "#15803d",
  teal: "#0f766e",
  injection: "#b42318",
  jailbreak: "#b7791f",
  extraction: "#6d28d9",
  roleplay: "#2563eb",
  muted: "#98a2b3",
  ink: "#17212b",
};

const labelColors = {
  "Safe Prompt": colors.safe,
  "Prompt Injection": colors.injection,
  Jailbreak: colors.jailbreak,
  "Data Extraction": colors.extraction,
  "Roleplay Manipulation": colors.roleplay,
};

const elements = {
  promptInput: document.querySelector("#promptInput"),
  analyzeBtn: document.querySelector("#analyzeBtn"),
  reportBtn: document.querySelector("#reportBtn"),
  clearHistoryBtn: document.querySelector("#clearHistoryBtn"),
  modelChip: document.querySelector("#modelChip"),
  totalPrompts: document.querySelector("#totalPrompts"),
  safePrompts: document.querySelector("#safePrompts"),
  threatsDetected: document.querySelector("#threatsDetected"),
  highRisk: document.querySelector("#highRisk"),
  statusPill: document.querySelector("#statusPill"),
  threatType: document.querySelector("#threatType"),
  riskScore: document.querySelector("#riskScore"),
  confidenceScore: document.querySelector("#confidenceScore"),
  severityLevel: document.querySelector("#severityLevel"),
  indicatorList: document.querySelector("#indicatorList"),
  mitigationList: document.querySelector("#mitigationList"),
  explanationList: document.querySelector("#explanationList"),
  historyBody: document.querySelector("#historyBody"),
  typeChart: document.querySelector("#typeChart"),
  distributionChart: document.querySelector("#distributionChart"),
  trendChart: document.querySelector("#trendChart"),
};

let currentReportUrl = "";

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function riskClass(level) {
  const normalized = String(level || "").toLowerCase();
  if (normalized === "low" || normalized === "benign") return "safe";
  if (normalized === "medium" || normalized === "moderate") return "medium";
  if (normalized === "critical") return "critical";
  return "high";
}

function setList(node, items, formatter) {
  node.innerHTML = "";
  if (!items || !items.length) {
    const li = document.createElement("li");
    li.textContent = "None";
    node.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = formatter ? formatter(item) : item;
    node.appendChild(li);
  });
}

function setStatus(text, level) {
  elements.statusPill.textContent = text;
  elements.statusPill.className = `pill ${riskClass(level)}`;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

async function analyzePrompt() {
  const prompt = elements.promptInput.value.trim();
  if (!prompt) {
    setStatus("Prompt required", "Medium");
    return;
  }

  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.classList.add("loading");
  setStatus("Analyzing", "Medium");
  try {
    const result = await requestJson("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
    renderResult(result);
    currentReportUrl = result.report_url;
    elements.reportBtn.disabled = false;
    await refreshDashboard();
  } catch (error) {
    setStatus(error.message, "High");
  } finally {
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.classList.remove("loading");
  }
}

function renderResult(result) {
  elements.threatType.textContent = result.classification;
  elements.riskScore.textContent = `${result.risk_score} / 100`;
  elements.confidenceScore.textContent = `${result.confidence_percent}%`;
  elements.severityLevel.textContent = result.severity;
  setStatus(result.risk_level, result.risk_level);

  setList(
    elements.indicatorList,
    result.triggered_indicators,
    (item) => `${item.rule_id} - ${item.label}: ${item.evidence}`,
  );
  setList(elements.mitigationList, result.mitigations);
  setList(elements.explanationList, result.explanations);
}

async function refreshDashboard() {
  const [stats, history] = await Promise.all([
    requestJson("/api/stats"),
    requestJson("/api/history?limit=25"),
  ]);
  renderStats(stats);
  renderHistory(history.items);
}

function renderStats(stats) {
  elements.totalPrompts.textContent = formatNumber(stats.total);
  elements.safePrompts.textContent = formatNumber(stats.safe);
  elements.threatsDetected.textContent = formatNumber(stats.threats);
  elements.highRisk.textContent = formatNumber(stats.high_risk);

  drawPieChart(elements.typeChart, stats.by_type);
  drawBarChart(elements.distributionChart, stats.by_type);
  drawTrendChart(elements.trendChart, stats.trend);
}

function renderHistory(items) {
  elements.historyBody.innerHTML = "";
  if (!items.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5">No prompts analyzed yet</td>';
    elements.historyBody.appendChild(row);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("tr");
    const time = new Date(item.created_at).toLocaleString();
    const prompt = escapeHtml(item.prompt);
    const classification = escapeHtml(item.classification);
    row.innerHTML = `
      <td>${time}</td>
      <td><span class="prompt-snippet">${prompt}</span></td>
      <td>${classification}</td>
      <td>${item.risk_score}</td>
      <td><a class="mini-link" href="/api/report/${item.id}.pdf">PDF</a></td>
    `;
    elements.historyBody.appendChild(row);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  return { ctx, width: rect.width, height: rect.height };
}

function emptyChart(ctx, width, height) {
  ctx.fillStyle = colors.muted;
  ctx.font = "700 13px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("No data", width / 2, height / 2);
}

function drawPieChart(canvas, data) {
  const { ctx, width, height } = setupCanvas(canvas);
  const total = data.reduce((sum, item) => sum + item.count, 0);
  if (!total) return emptyChart(ctx, width, height);

  const cx = width * 0.34;
  const cy = height * 0.5;
  const radius = Math.min(width, height) * 0.31;
  let angle = -Math.PI / 2;

  data.forEach((item) => {
    const slice = (item.count / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = labelColors[item.label] || colors.muted;
    ctx.fill();
    angle += slice;
  });

  ctx.font = "700 12px system-ui";
  ctx.textAlign = "left";
  data.slice(0, 5).forEach((item, index) => {
    const y = 42 + index * 30;
    ctx.fillStyle = labelColors[item.label] || colors.muted;
    ctx.fillRect(width * 0.66, y - 9, 12, 12);
    ctx.fillStyle = colors.ink;
    ctx.fillText(`${item.label} (${item.count})`, width * 0.66 + 20, y);
  });
}

function drawBarChart(canvas, data) {
  const { ctx, width, height } = setupCanvas(canvas);
  const max = Math.max(...data.map((item) => item.count), 0);
  if (!max) return emptyChart(ctx, width, height);

  const barHeight = 24;
  const gap = 18;
  const left = 18;
  const labelWidth = Math.min(170, width * 0.42);
  const chartWidth = width - labelWidth - 52;

  ctx.font = "700 12px system-ui";
  data.slice(0, 5).forEach((item, index) => {
    const y = 28 + index * (barHeight + gap);
    const barWidth = (item.count / max) * chartWidth;
    ctx.fillStyle = colors.ink;
    ctx.fillText(item.label, left, y + 17);
    ctx.fillStyle = labelColors[item.label] || colors.muted;
    ctx.fillRect(labelWidth, y, barWidth, barHeight);
    ctx.fillStyle = colors.muted;
    ctx.fillText(String(item.count), labelWidth + barWidth + 8, y + 17);
  });
}

function drawTrendChart(canvas, trend) {
  const { ctx, width, height } = setupCanvas(canvas);
  if (!trend.length) return emptyChart(ctx, width, height);

  const padding = 34;
  const max = Math.max(...trend.map((item) => item.count), 1);
  const points = trend.map((item, index) => {
    const x =
      padding + (index / Math.max(1, trend.length - 1)) * (width - padding * 2);
    const y = height - padding - (item.count / max) * (height - padding * 2);
    return { x, y, item };
  });

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();

  ctx.strokeStyle = colors.teal || "#0f766e";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  points.forEach((point) => {
    ctx.fillStyle = colors.teal || "#0f766e";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = colors.muted;
  ctx.font = "700 11px system-ui";
  ctx.textAlign = "center";
  const first = points[0];
  const last = points[points.length - 1];
  ctx.fillText(first.item.day, first.x, height - 9);
  if (last !== first) ctx.fillText(last.item.day, last.x, height - 9);
}

async function loadHealth() {
  try {
    const health = await requestJson("/api/health");
    const test = health.model.test_metrics || {};
    const macroF1 = test.macro_f1 ? `Macro F1 ${(test.macro_f1 * 100).toFixed(1)}%` : "Model ready";
    elements.modelChip.textContent = macroF1;
  } catch {
    elements.modelChip.textContent = "Model unavailable";
  }
}

elements.analyzeBtn.addEventListener("click", analyzePrompt);
elements.promptInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    analyzePrompt();
  }
});
elements.reportBtn.addEventListener("click", () => {
  if (currentReportUrl) window.location.href = currentReportUrl;
});
elements.clearHistoryBtn.addEventListener("click", async () => {
  if (!confirm("Clear PromptShield history?")) return;
  await fetch("/api/history", { method: "DELETE" });
  currentReportUrl = "";
  elements.reportBtn.disabled = true;
  await refreshDashboard();
});
window.addEventListener("resize", () => refreshDashboard());

loadHealth();
refreshDashboard();
