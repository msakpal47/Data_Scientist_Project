function formatCurrency(v){return "₹ "+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}
function buildFIChart(items){
  var labels=items.slice(0,15).map(function(x){return x.feature});
  var data=items.slice(0,15).map(function(x){return x.importance});
  var ctx=document.getElementById("fi-chart");
  if(!ctx)return;
  new Chart(ctx,{type:"bar",data:{labels:labels,datasets:[{label:"Importance",data:data,backgroundColor:"#4f46e5"}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}});
}
function buildFITable(items){
  var tbody=document.getElementById("fi-table-body");
  if(!tbody)return;
  tbody.innerHTML="";
  items.forEach(function(x){
    var tr=document.createElement("tr");
    var td1=document.createElement("td");td1.textContent=x.feature;
    var td2=document.createElement("td");td2.textContent=x.importance.toFixed(6);
    tr.appendChild(td1);tr.appendChild(td2);tbody.appendChild(tr);
  });
}
document.addEventListener("DOMContentLoaded",function(){
  var items=window.__FEATURE_IMPORTANCE__||[];
  buildFIChart(items);
  buildFITable(items);
  var fiEmpty=document.getElementById("fi-empty");
  if(fiEmpty){
    if(!items || items.length===0){fiEmpty.textContent="Train the model to populate feature importance."}else{fiEmpty.textContent=""}
  }
  function ensureSelect(name,id,placeholder){
    var el=document.getElementById(id);
    if(el)return el;
    var current=document.querySelector('[name="'+name+'"]');
    if(!current)return null;
    var sel=document.createElement("select");
    sel.name=name;sel.id=id;sel.required=current.required;
    var opt=document.createElement("option");opt.value="";opt.textContent=placeholder||"Select";sel.appendChild(opt);
    current.parentNode.replaceChild(sel,current);
    return sel;
  }
  var storeSelect=ensureSelect("store","store-select","Select Store");
  var dateSelect=ensureSelect("date","date-select","Select Date");
  var customersSelect=ensureSelect("customers","customers-select","Select Customers");
  var dowSelect=document.querySelector('[name="day_of_week"]');
  var inline=window.__OPTIONS__;
  function fill(opt){
    if(storeSelect && Array.isArray(opt.stores)){
      storeSelect.innerHTML='<option value=\"\">Select Store</option>';
      opt.stores.forEach(function(s){
        var o=document.createElement("option");o.value=String(s);o.textContent=String(s);storeSelect.appendChild(o);
      });
    }
    if(dateSelect && Array.isArray(opt.dates)){
      dateSelect.innerHTML='<option value=\"\">Select Date</option>';
      opt.dates.forEach(function(d){
        var o=document.createElement("option");o.value=String(d);o.textContent=String(d);dateSelect.appendChild(o);
      });
    }
    if(customersSelect && Array.isArray(opt.customers)){
      customersSelect.innerHTML='<option value=\"\">Select Customers</option>';
      opt.customers.forEach(function(c){
        var o=document.createElement("option");o.value=String(c);o.textContent=String(c);customersSelect.appendChild(o);
      });
    }
    if(dowSelect && dowSelect.options.length<=1){
      dowSelect.innerHTML='<option value=\"\">Select Day</option>';
      ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].forEach(function(lbl,idx){
        var o=document.createElement("option");o.value=String(idx);o.textContent=lbl;dowSelect.appendChild(o);
      });
    }
  }
  if(inline && (inline.stores?.length || inline.dates?.length)){fill(inline)}
  else{fetch("/options").then(function(r){return r.json()}).then(fill).catch(function(){})}
  var form=document.getElementById("predict-form");
  var error=document.getElementById("form-error");
  var resultVal=document.getElementById("prediction-value");
  var resultConf=document.getElementById("prediction-confidence");
  var downloadBtn=document.getElementById("download-btn");
  var trainBtn=document.getElementById("train-btn");
  if(form){
    form.addEventListener("submit",function(e){
      e.preventDefault();
      error.textContent="";
      var fd=new FormData(form);
      var payload={
        store:Number(fd.get("store")),
        day_of_week:Number(fd.get("day_of_week")),
        date:String(fd.get("date")),
        customers:Number(fd.get("customers")),
        promo:Number(fd.get("promo")),
        holiday:String(fd.get("holiday"))
      };
      var btn=document.getElementById("predict-btn");if(btn){btn.disabled=true}
      fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}).then(function(r){return r.json().then(function(j){return{ok:r.ok,data:j}})}).then(function(resp){
        if(!resp.ok){error.textContent=resp.data.error||"Prediction failed";return}
        resultVal.textContent=Number(resp.data.predicted_sales).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
        var conf=resp.data.confidence;
        var interval=resp.data.interval||{};
        var confText="";
        if(typeof conf==="number"){confText+="Confidence: "+Math.round(conf*100)+"%"}
        if(typeof interval.lower==="number" && typeof interval.upper==="number"){
          if(confText) confText+=" • ";
          confText+="Range: "+formatCurrency(interval.lower)+" – "+formatCurrency(interval.upper);
        }
        resultConf.textContent=confText;
      }).catch(function(){error.textContent="Network error"}).finally(function(){if(btn){btn.disabled=false}});
    });
  }
  if(downloadBtn){
    downloadBtn.addEventListener("click",function(){
      window.location.href="/download";
    });
  }
  if(trainBtn){
    trainBtn.addEventListener("click",function(){
      trainBtn.disabled=true;
      fetch("/train-sync",{method:"POST"})
        .then(function(r){return r.json().then(function(j){return{ok:r.ok,data:j}})})
        .then(function(resp){
          if(!resp.ok){throw new Error(resp.data && resp.data.message || "Training failed")}
          return resp;
        })
        .catch(function(){
          return fetch("/train",{method:"POST"}).then(function(){
            return new Promise(function(resolve){
              var tries=0;
              var iv=setInterval(function(){
                fetch("/status").then(function(r){return r.json()}).then(function(s){
                  if(s.model_ready){clearInterval(iv);resolve()}
                }).catch(function(){});
                tries++;if(tries>180){clearInterval(iv);resolve()}
              },1000);
            })
          })
        })
        .then(function(){window.location.reload()})
        .catch(function(){trainBtn.disabled=false});
    });
  }
});
