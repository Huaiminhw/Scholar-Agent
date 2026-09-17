const $ = (q) => document.querySelector(q);
let lastResult = null;
const esc = (value = "") => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const show = (el, yes = true) => el.classList.toggle("hidden", !yes);

document.querySelectorAll(".examples button").forEach(btn => btn.onclick = () => { $("#question").value = btn.textContent; $("#question").focus(); });
$("#mode").onchange = e => { $("#top-k").disabled = $("#selector").disabled = e.target.value === "understanding"; };
$("#close-dialog").onclick = () => $("#json-dialog").close();
$("#json-btn").onclick = () => { $("#json-output").textContent = JSON.stringify(lastResult, null, 2); $("#json-dialog").showModal(); };

$("#search-form").onsubmit = async (event) => {
  event.preventDefault();
  show($("#error"), false); show($("#results"), false); show($("#summary"), false); show($("#progress"));
  $("#submit").disabled = true;
  try {
    const response = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question:$("#question").value, mode:$("#mode").value, top_k:+$("#top-k").value, use_selector:$("#selector").checked})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "请求失败");
    lastResult = data; render(data);
  } catch (err) {
    $("#error").textContent = `运行失败：${err.message}`; show($("#error"));
  } finally { show($("#progress"), false); $("#submit").disabled = false; }
};

function render(data) {
  const docs = data.documents || [], u = data.query_understanding || {}, r = data.retrieval || {};
  const latency = data.total_latency_ms ? `${(data.total_latency_ms/1000).toFixed(1)}s` : `${Math.round(u.metrics?.latency_ms || 0)}ms`;
  $("#summary").innerHTML = [
    [docs.length, "返回论文"], [r.metrics?.raw_hits ?? "—", "原始召回"], [r.sources?.length ?? "—", "检索来源"], [latency, "总耗时"]
  ].map(x => `<div class="metric"><span>${x[1]}</span><b>${x[0]}</b></div>`).join(""); show($("#summary"));
  if (data.mode === "understanding") {
    $("#paper-list").innerHTML = `<div class="analysis"><h4>问题理解结果</h4><pre>${esc(JSON.stringify(u.result, null, 2))}</pre></div>`;
  } else {
    $("#paper-list").innerHTML = docs.length ? docs.map(paperCard).join("") : '<div class="analysis">没有找到符合条件的论文。</div>';
  }
  show($("#results")); $("#results").scrollIntoView({behavior:"smooth", block:"start"});
}

function paperCard(p) {
  const url = p.urls?.[0], authors = (p.authors || []).slice(0, 4).join("、") || "作者未知";
  const title = url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(p.title || "无标题")}</a>` : esc(p.title || "无标题");
  const sources = p.provenance?.sources || [];
  return `<article class="paper"><div class="rank">${String(p.rank).padStart(2,"0")}</div><div><h4>${title}</h4><div class="meta">${esc(authors)} · ${esc(p.year || "年份未知")} · ${esc(p.venue || "来源未知")}</div><p class="abstract">${esc((p.abstract || "暂无摘要").slice(0, 420))}${(p.abstract || "").length > 420 ? "…" : ""}</p><div class="tags">${sources.map(s => `<span class="tag">${esc(s)}</span>`).join("")} ${p.citation_count != null ? `<span class="tag">引用 ${p.citation_count}</span>` : ""}</div></div></article>`;
}
