const form = document.getElementById('predict-form');
const resultEl = document.getElementById('result');
const resultDetails = document.getElementById('result-details');
const outputMode = document.getElementById('output-mode');
const leaderboardBody = document.getElementById('leaderboard-body');
const bestModelEl=document.getElementById('bestModel');
const bestR2El=document.getElementById('bestR2');
const bestMAEEl=document.getElementById('bestMAE');
const lastPredEl=document.getElementById('lastPrediction');
const loader=document.getElementById('loader');
function showToast(msg){const t=document.getElementById('toast');if(!t)return;t.textContent=msg;t.style.display='block';setTimeout(()=>{t.style.display='none'},3000)}
function _r2value(it){return (typeof it.r2==='number')?it.r2:((typeof it.r2_test==='number')?it.r2_test:0)}
function renderLeaderboard(data){if(!Array.isArray(data))return;leaderboardBody.innerHTML='';data.sort((a,b)=>(_r2value(b)-_r2value(a))).forEach(it=>{const tr=document.createElement('tr');const r2v=_r2value(it);const r2=(r2v!==null&&r2v!==undefined)?r2v.toFixed(3):'';const r2t=(typeof it.r2_train==='number')?it.r2_train.toFixed(3):'';const mae=(typeof it.mae==='number')?it.mae.toFixed(1):'';const rmse=(typeof it.rmse==='number')?it.rmse.toFixed(1):'';tr.innerHTML=`<td>${it.name||''}</td><td>${r2}</td><td>${r2t}</td><td>${mae}</td><td>${rmse}</td>`;leaderboardBody.appendChild(tr);});if(data.length){const best=[...data].sort((a,b)=>(_r2value(b)-_r2value(a)))[0];if(bestModelEl)bestModelEl.textContent=best.name||'';if(bestR2El)bestR2El.textContent=_r2value(best).toFixed(3);if(bestMAEEl)bestMAEEl.textContent=(typeof best.mae==='number')?best.mae.toFixed(1):'--';}}
renderLeaderboard(window.__LEADERBOARD__||[]);
async function refreshLeaderboard(){try{const r=await fetch('/model/leaderboard?t='+Date.now());const data=await r.json();renderLeaderboard(data);}catch(_){}}
refreshLeaderboard();
async function postJSON(url,obj){const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});if(!res.ok){throw new Error(await res.text())}return res.json()}
function formToJSON(form){const data=new FormData(form);const obj={};for(const [k,v] of data.entries()){obj[k]=v}return obj}
function formatINR(n){try{return new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:2}).format(Number(n))}catch(_){return `₹ ${Number(n).toFixed(2)}`}}
async function populateCategory(selectId,column){
  try{
    const sel=document.getElementById(selectId);if(!sel)return;
    const defaults = (column==='medical_history' || column==='family_medical_history')
      ? ['None','Diabetes','Hypertension','Heart disease','High blood pressure','Asthma','Cancer','COPD','Chronic kidney disease','Thyroid disorder','Arthritis','Depression','Anxiety','Stroke','Obesity']
      : ['None'];
    let fetched=[];
    try{
      const r=await fetch('/category-values?column='+encodeURIComponent(column));
      const data=await r.json();
      fetched=Array.isArray(data.values)?data.values:[];
    }catch(_){}
    const seen=new Set();
    const union=[...defaults,...fetched].filter(v=>{
      const s=String(v||'').trim();
      if(!s.length) return false;
      const key=s.toLowerCase();
      if(seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const current=sel.value;
    sel.innerHTML='';
    union.forEach(v=>{
      const opt=document.createElement('option');
      opt.value=v;opt.textContent=v;
      sel.appendChild(opt);
    });
    if(current){sel.value=current;}
  }catch(e){}
}
window.addEventListener('load',()=>{
  populateCategory('medical-history-select','medical_history');
  populateCategory('family-history-select','family_medical_history');
  const mh=document.getElementById('medical-history-select');
  const fh=document.getElementById('family-history-select');
  ['focus','click'].forEach(ev=>{
    if(mh)mh.addEventListener(ev,()=>populateCategory('medical-history-select','medical_history'));
    if(fh)fh.addEventListener(ev,()=>populateCategory('family-history-select','family_medical_history'));
  });
});
// initial population kept via load listener above
if(form){form.addEventListener('submit',async e=>{e.preventDefault();const payload=formToJSON(form);const btn=document.getElementById('predictBtn');const age=Number(payload.age||0);if(age&&(age<18||age>100)){showToast('Invalid Age');return;}try{if(loader)loader.style.display='block';if(btn)btn.disabled=true;const res=await postJSON('/predict',payload);const predStr=formatINR(res.prediction);const mode=(outputMode&&outputMode.value)||'summary';if(mode==='detailed'&&resultDetails){let rows='';Object.keys(payload).forEach(k=>{rows+=`<tr><td>${k}</td><td>${payload[k]}</td></tr>`});resultDetails.innerHTML=`<table class="table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table>`;resultDetails.style.display='block';}else if(resultDetails){resultDetails.style.display='none';}resultEl.innerHTML=`<div class="prediction-box">${predStr}</div>`;if(lastPredEl)lastPredEl.textContent=predStr;refreshLeaderboard();loadImportance();}catch(err){showToast('Error: '+err.message);if(resultDetails){resultDetails.style.display='none';}}finally{if(btn)btn.disabled=false;if(loader)loader.style.display='none';}})}
 
async function loadImportance(){try{const imp=await fetch('/feature-importance?t='+Date.now()).then(r=>r.json());const labels=Object.keys(imp);const values=labels.map(k=>imp[k]);const ctx=document.getElementById('importanceChart');const txt=document.getElementById('importanceText');if((!labels.length)&&txt){txt.textContent='No feature importance available'}if(ctx){if(!window.Chart||!labels.length){ctx.style.display='none'}else{ctx.style.display='block';new Chart(ctx,{type:'bar',data:{labels:labels.slice(0,20),datasets:[{label:'Importance',data:values.slice(0,20),backgroundColor:labels.slice(0,20).map(()=> 'rgba(96,165,250,0.6)'),borderRadius:6}]},options:{indexAxis:'y',maintainAspectRatio:false,scales:{x:{ticks:{color:'#cbd5e1'}},y:{ticks:{color:'#cbd5e1'}}},plugins:{legend:{labels:{color:'#cbd5e1'}}}})}}if(txt&&labels.length){const pairs=labels.map((k,i)=>[k,values[i]]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,20);const rows=pairs.map(p=>`<tr><td>${p[0]}</td><td>${(typeof p[1]==='number')?p[1].toFixed(3):p[1]}</td></tr>`).join('');txt.innerHTML=`<table class=\"table\"><thead><tr><th>Feature</th><th>Impact</th></tr></thead><tbody>${rows}</tbody></table>`}}catch(e){const txt=document.getElementById('importanceText');if(txt){txt.textContent='Failed to load feature importance'}}}
loadImportance();
// SHAP removed for assignment compliance
const jsonForm=document.getElementById('json-form');const jsonInput=document.getElementById('json-input');const jsonResult=document.getElementById('json-result');const jsonShap=document.getElementById('json-shap');const jsonShapChart=document.getElementById('jsonShapChart');const colsEl=document.getElementById('columns');
async function initColumns(){
  try{
    const cols=await fetch('/model/columns').then(r=>r.json());
    const num=(cols.numeric||[]);
    const cat=(cols.categorical||[]);
    const labels=[].concat(num,cat);
    if(colsEl && (!(colsEl.textContent)|| !colsEl.textContent.trim())){colsEl.textContent=labels.join(', ')}
    if(jsonInput&&labels.length){
      const obj={};
      num.forEach(k=>{
        obj[k]=(k==='age')?30:(k==='bmi')?25:(k==='children')?0:1;
      });
      cat.forEach(k=>{
        obj[k]='Unknown';
      });
      jsonInput.value=JSON.stringify(obj,null,2)
    }
  }catch(e){
    if(colsEl){colsEl.textContent='No model columns available yet'}
  }
}
initColumns();
if(jsonForm){
  let shapChart=null;
  jsonForm.addEventListener('submit',async e=>{
    e.preventDefault();
    try{
      const payload=JSON.parse(jsonInput.value||'{}');
      const res=await postJSON('/predict-json',payload);
      jsonResult.textContent=JSON.stringify(res);
      if(lastPredEl && res && typeof res.prediction==='number'){lastPredEl.textContent=formatINR(res.prediction)}
      refreshLeaderboard();
      loadImportance();
      try{
        const ex=await postJSON('/shap/explain',payload);
        const contribs=(ex&&ex.contributions)||[];
        if(jsonShapChart && contribs.length){
          const labels=contribs.slice(0,20).map(it=>it.name);
          const values=contribs.slice(0,20).map(it=>it.value);
          if(shapChart){shapChart.destroy();}
          shapChart=new Chart(jsonShapChart,{type:'bar',data:{labels:labels,datasets:[{label:'SHAP Impact',data:values,backgroundColor:'#f59e0b'}]},options:{indexAxis:'y',scales:{x:{ticks:{color:'#cbd5e1'}},y:{ticks:{color:'#cbd5e1'}}},plugins:{legend:{labels:{color:'#cbd5e1'}}}}});
        } else {
          if(jsonShap){jsonShap.textContent='SHAP explanation not available'}
        }
      }catch(_){
        if(jsonShap){jsonShap.textContent='SHAP explanation not available'}
      }
    }catch(err){jsonResult.textContent='Error: '+err.message}
  })
}
