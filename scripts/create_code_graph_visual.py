#!/usr/bin/env python3
"""Create a readable interactive view of the repository code graph."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TEMPLATE = r'''<div id="codegraph-v2">
<style>
#codegraph-v2{width:100%;color:var(--foreground)}
#codegraph-v2 .cg-controls{display:grid;grid-template-columns:minmax(220px,1fr) auto auto;gap:8px;align-items:end;margin:12px 0}
#codegraph-v2 .cg-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 12px}
#codegraph-v2 .cg-flow-step{padding:8px;text-align:center;color:var(--muted-foreground);border-bottom:3px solid var(--border)}
#codegraph-v2 .cg-flow-step.active{color:var(--foreground);border-color:var(--viz-series-4)}
#codegraph-v2 .cg-legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0;color:var(--muted-foreground)}
#codegraph-v2 .cg-legend span::before{content:"";display:inline-block;width:10px;height:10px;margin-right:5px;background:currentColor}
#codegraph-v2 .cg-legend .cg-function::before{border-radius:50%}
#codegraph-v2 .cg-legend .cg-import-node::before{transform:rotate(45deg)}
#codegraph-v2 .cg-legend .cg-anchor::before{background:transparent;border:3px solid currentColor;border-radius:50%;box-sizing:border-box}
#codegraph-v2 .cg-import{color:var(--viz-series-1)}#codegraph-v2 .cg-call{color:var(--viz-series-2)}#codegraph-v2 .cg-change{color:var(--viz-series-3)}#codegraph-v2 .cg-anchor{color:var(--viz-series-4)}
#codegraph-v2 .cg-plot{height:560px;position:relative}#codegraph-v2 .cg-plot svg{display:block;width:100%;height:100%}
#codegraph-v2 .cg-link{stroke-opacity:.34;fill:none}#codegraph-v2 .cg-link.import_to_import,#codegraph-v2 .cg-link.import_to_file{stroke:var(--viz-series-1)}#codegraph-v2 .cg-link.function_to_function{stroke:var(--viz-series-2)}#codegraph-v2 .cg-link.file_to_file{stroke:var(--viz-series-3)}#codegraph-v2 .cg-link.file_contains_function,#codegraph-v2 .cg-link.function_defined_in_file,#codegraph-v2 .cg-link.file_contains_import{stroke:var(--muted-foreground);stroke-dasharray:3 3}
#codegraph-v2 .cg-link.is-flow{stroke-opacity:.9;stroke-width:2.2;animation:cg-flow 1.4s ease-out 1}
#codegraph-v2 .cg-node path{stroke:var(--background);stroke-width:2}#codegraph-v2 .cg-node text{fill:var(--foreground);font-size:11px;paint-order:stroke;stroke:var(--background);stroke-width:3px;stroke-linejoin:round}#codegraph-v2 .cg-node.is-anchor path{stroke:var(--viz-series-4);stroke-width:5px}
#codegraph-v2 .cg-status{margin:8px 0;color:var(--muted-foreground)}#codegraph-v2 .cg-grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-top:12px}#codegraph-v2 .cg-table{width:100%;border-collapse:collapse}#codegraph-v2 .cg-table th,#codegraph-v2 .cg-table td{padding:6px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}#codegraph-v2 .cg-table td:last-child{text-align:right;font-variant-numeric:tabular-nums}#codegraph-v2 .cg-detail{overflow-wrap:anywhere}
@keyframes cg-flow{from{stroke-dasharray:2 16;stroke-dashoffset:44;stroke-opacity:.2}to{stroke-dashoffset:0;stroke-opacity:.9}}
@media(prefers-reduced-motion:reduce){#codegraph-v2 .cg-link.is-flow{animation:none}}
@media(max-width:700px){#codegraph-v2 .cg-controls{grid-template-columns:1fr 1fr}#codegraph-v2 .cg-controls label{grid-column:1/-1}#codegraph-v2 .cg-flow{grid-template-columns:1fr 1fr}#codegraph-v2 .cg-grid{grid-template-columns:1fr}#codegraph-v2 .cg-plot{height:440px}}
</style>
<h2>Codebase knowledge graph</h2>
<div class="cg-controls viz-controls">
  <label class="form-label">Query<input id="cg-query" class="form-control" value="Where are BM25 and dense HNSW results fused with reciprocal rank fusion?"></label>
  <button id="cg-run" class="btn btn-primary" type="button">Find anchor + rank</button>
  <button id="cg-architecture" class="btn" type="button">Show architecture</button>
</div>
<div class="cg-flow" aria-live="polite"><div class="cg-flow-step active" data-step="1">1 · Query</div><div class="cg-flow-step" data-step="2">2 · Top anchors</div><div class="cg-flow-step" data-step="3">3 · Personalized PageRank</div><div class="cg-flow-step" data-step="4">4 · Agent context</div></div>
<div class="cg-legend"><span>File entity</span><span class="cg-function cg-call">Function entity</span><span class="cg-import-node cg-import">Import entity</span><span class="cg-call">Function call</span><span class="cg-import">Import relationship</span><span class="cg-change">Git co-change</span><span class="cg-anchor">Query anchor</span></div>
<div id="cg-status" class="cg-status" aria-live="polite"></div>
<div id="cg-plot" class="cg-plot" role="img" aria-label="Labeled repository knowledge graph"></div>
<div class="cg-grid"><section><h3>PageRank ranking</h3><div class="table-responsive"><table class="table table-sm cg-table"><thead><tr><th>#</th><th>Node</th><th>Score</th></tr></thead><tbody id="cg-ranking"></tbody></table></div></section><section><h3>Selected node</h3><div id="cg-detail" class="cg-detail text-muted">Click a labeled node to inspect it.</div></section></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(()=>{const DATA=__DATA__,root=document.getElementById('codegraph-v2'),FILES=__FILES__,FEDGES=__FEDGES__;let sim,resizeMode='query';
const color=d=>d.anchor?'var(--viz-series-4)':d.kind==='file'?'var(--viz-series-3)':d.kind==='function'?'var(--viz-series-2)':'var(--viz-series-1)';
const short=id=>{const x=id.replace(/^(file|function|import):/,'').replace(/^src\.hybrid_rag\./,'');return x.length>38?'…'+x.slice(-37):x};
function render(rawNodes,rawEdges,anchorIds=[]){if(sim)sim.stop();const anchorSet=new Set(anchorIds);const box=document.getElementById('cg-plot');box.innerHTML='';const w=Math.max(320,box.clientWidth),h=box.clientHeight;const svg=d3.select(box).append('svg').attr('viewBox',`0 0 ${w} ${h}`);svg.append('rect').attr('data-chart-frame','').attr('x',1).attr('y',1).attr('width',w-2).attr('height',h-2).attr('fill','var(--card)').attr('stroke','var(--border)');const zoom=svg.append('g');svg.call(d3.zoom().scaleExtent([.45,4]).on('zoom',e=>zoom.attr('transform',e.transform)));const nodes=rawNodes.map(d=>({...d,anchor:anchorSet.has(d.id)})),edges=rawEdges.map(d=>({...d}));const links=zoom.append('g').selectAll('line').data(edges).join('line').attr('class',d=>`cg-link ${d.type}${anchorIds.length?' is-flow':''}`).attr('stroke-width',d=>Math.min(5,1+Math.log1p(d.weight||1)));const groups=zoom.append('g').selectAll('g').data(nodes).join('g').attr('class',d=>`cg-node${d.anchor?' is-anchor':''}`).style('cursor','pointer').on('click',(e,d)=>document.getElementById('cg-detail').innerHTML=`<strong>${d.id}</strong><br>${d.file||d.path||d.kind}<br>Kind: ${d.kind||'file'}`);const shape=d3.symbol().size(d=>d.anchor?300:Math.min(240,100+20*Math.sqrt(d.degree||1))).type(d=>d.kind==='file'?d3.symbolSquare:d.kind==='import'?d3.symbolDiamond:d3.symbolCircle);groups.append('path').attr('d',shape).attr('fill',color);groups.append('text').attr('x',10).attr('y',4).text(d=>d.label||short(d.id));groups.append('title').text(d=>d.id);sim=d3.forceSimulation(nodes).force('link',d3.forceLink(edges).id(d=>d.id).distance(d=>55+Math.min(80,8*(d.weight||1))).strength(.25)).force('charge',d3.forceManyBody().strength(-230)).force('center',d3.forceCenter(w/2,h/2)).force('collision',d3.forceCollide(24)).on('tick',()=>{links.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);groups.attr('transform',d=>`translate(${Math.max(12,Math.min(w-12,d.x))},${Math.max(12,Math.min(h-12,d.y))})`)});}
function setStep(step){root.querySelectorAll('.cg-flow-step').forEach(el=>el.classList.toggle('active',Number(el.dataset.step)<=step))}
function architecture(){resizeMode='architecture';setStep(1);document.getElementById('cg-status').textContent=`Architecture view · ${FILES.length} labeled files · ${FEDGES.length} aggregated relationships`;document.getElementById('cg-ranking').innerHTML='';render(FILES,FEDGES);}
const terms=s=>new Set((s.toLowerCase().replace(/[._/\\-]+/g,' ').match(/[a-z][a-z0-9]*/g)||[]).filter(x=>x.length>2));
function query(){resizeMode='query';setStep(1);const q=terms(document.getElementById('cg-query').value),hits=DATA.nodes.map(n=>{const nt=terms(JSON.stringify(n)),overlap=[...q].filter(x=>nt.has(x)).length;const kindBoost=n.kind==='function'?1.15:n.kind==='file'?1.05:1;return{node:n,similarity:(overlap/Math.max(1,q.size))*kindBoost}}).filter(x=>x.similarity>0).sort((a,b)=>b.similarity-a.similarity||a.node.id.localeCompare(b.node.id));if(!hits.length){document.getElementById('cg-status').textContent='No query anchor found';return}const anchors=hits.slice(0,Math.min(5,hits.length));setTimeout(()=>setStep(2),160);const ids=DATA.nodes.map(n=>n.id),idx=new Map(ids.map((id,i)=>[id,i])),out=ids.map(()=>[]);DATA.edges.forEach(e=>{const a=idx.get(e.source),b=idx.get(e.target);if(a!==undefined&&b!==undefined){const weight=e.type==='file_to_file'?Math.log1p(e.cochange_count||1):(e.weight||1);out[a].push({target:b,weight});if(e.bidirectional)out[b].push({target:a,weight})}});let personalization=ids.map(()=>0);const totalAnchor=anchors.reduce((s,a)=>s+a.similarity,0)||1;anchors.forEach(a=>personalization[idx.get(a.node.id)]=a.similarity/totalAnchor);let p=ids.map(()=>1/ids.length),d=.85;for(let z=0;z<40;z++){const np=personalization.map(v=>(1-d)*v);let dangling=0;out.forEach((os,i)=>{if(!os.length)dangling+=p[i];else{const total=os.reduce((s,e)=>s+e.weight,0);os.forEach(e=>np[e.target]+=d*p[i]*e.weight/total)}});personalization.forEach((v,i)=>np[i]+=d*dangling*v);p=np}setTimeout(()=>setStep(3),360);const ranked=ids.map((id,i)=>({node:DATA.nodes[i],score:p[i]})).sort((a,b)=>b.score-a.score).slice(0,24);const keep=new Set(ranked.map(r=>r.node.id));anchors.forEach(a=>keep.add(a.node.id));ranked.filter(r=>r.node.kind==='file').slice(0,8).forEach(r=>DATA.edges.filter(e=>e.type==='file_to_file'&&(e.source===r.node.id||e.target===r.node.id)).slice(0,2).forEach(e=>keep.add(e.source===r.node.id?e.target:e.source)));const neighborhoodEdges=DATA.edges.filter(e=>keep.has(e.source)&&keep.has(e.target));const neighborhoodNodes=DATA.nodes.filter(n=>keep.has(n.id));const anchor=anchors[0];document.getElementById('cg-status').textContent=`Top anchor: ${anchor.node.id} · similarity ${(anchor.similarity*100).toFixed(1)}% · ${anchors.length} personalized seeds`;document.getElementById('cg-ranking').innerHTML=ranked.slice(0,15).map((r,i)=>`<tr><td>${i+1}</td><td>${short(r.node.id)}<br><span class="text-muted">${r.node.file||r.node.kind}</span></td><td>${r.score.toFixed(6)}</td></tr>`).join('');render(neighborhoodNodes,neighborhoodEdges,anchors.map(a=>a.node.id));setTimeout(()=>setStep(4),760);}
document.getElementById('cg-run').onclick=query;document.getElementById('cg-query').onkeydown=e=>{if(e.key==='Enter')query()};document.getElementById('cg-architecture').onclick=architecture;query();let timer;new ResizeObserver(()=>{clearTimeout(timer);timer=setTimeout(()=>resizeMode==='architecture'?architecture():query(),150)}).observe(document.getElementById('cg-plot'));
})();
</script></div>'''


def aggregate_files(graph: dict) -> tuple[list[dict], list[dict]]:
    node_file: dict[str, str] = {}
    files: dict[str, dict] = {}
    for node in graph["nodes"]:
        path = node.get("file") or node.get("path")
        if not path:
            continue
        file_id = f"file:{path}"
        node_file[node["id"]] = file_id
        files.setdefault(file_id, {"id": file_id, "kind": "file", "path": path, "label": Path(path).name, "degree": 0})
    aggregated: dict[tuple[str, str, str], int] = defaultdict(int)
    for edge in graph["edges"]:
        source = node_file.get(edge["source"], edge["source"] if edge["source"].startswith("file:") else None)
        target = node_file.get(edge["target"], edge["target"] if edge["target"].startswith("file:") else None)
        if not source or not target or source == target or source not in files or target not in files:
            continue
        aggregated[(source, target, edge["type"])] += int(edge.get("cochange_count", 1))
        files[source]["degree"] += 1
        files[target]["degree"] += 1
    edges = [{"source": s, "target": t, "type": kind, "weight": weight} for (s, t, kind), weight in aggregated.items()]
    return list(files.values()), edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("outputs/code_knowledge_base.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    files, file_edges = aggregate_files(graph)
    html = TEMPLATE.replace("__DATA__", json.dumps({"nodes": graph["nodes"], "edges": graph["edges"]}, separators=(",", ":")))
    html = html.replace("__FILES__", json.dumps(files, separators=(",", ":")))
    html = html.replace("__FEDGES__", json.dumps(file_edges, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"files": len(files), "file_edges": len(file_edges), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
