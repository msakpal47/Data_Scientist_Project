import os,sqlite3,json,re,io,csv,math,joblib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from preprocess import build_features
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(BASE_DIR,"clustering.db")
DATA_CARD_PATH=os.path.join(BASE_DIR,"Data Card.txt")
def norm(s):return re.sub(r"[^a-z0-9]","",str(s).lower())
CANON=["Id","Title","Price","User_id","profileName","review_helpfulness","review_score","review_time","review_summary","review_text"]
SYN={"Id":["id","productid","product_id"],"Title":["title","booktitle"],"Price":["price","cost"],"User_id":["user_id","userid","reviewerid"],"profileName":["profilename","username","user_name"],"review_helpfulness":["reviewhelpfulness","helpfulness","votes"],"review_score":["reviewscore","score","stars","rating"],"review_time":["reviewtime","time","timestamp","unix"],"review_summary":["reviewsummary","summary","headline"],"review_text":["reviewtext","text","body","content"]}
class DB:
    def __init__(self,db_path):
        self.conn=sqlite3.connect(db_path,check_same_thread=False)
        self.conn.row_factory=sqlite3.Row
        self.table,self.cols=self._resolve()
        self.map=self._build_map()
        self.derive_score=self.map.get("review_score") is None
    def _derive_score(self,text):
        t=(text or "").strip()
        if not t:return None
        n=len(t)
        if n<50:return 1.0
        if n<150:return 2.0
        if n<300:return 3.0
        if n<600:return 4.0
        return 5.0
    def _tables(self):
        cur=self.conn.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")
        return [r[0] for r in cur.fetchall()]
    def _cols(self,table):
        cur=self.conn.execute(f"pragma table_info({table})")
        return [r[1] for r in cur.fetchall()]
    def _resolve(self):
        best=None;best_score=-1;best_cols=[]
        for t in self._tables():
            cols=self._cols(t)
            score=sum(1 for c in cols for v in sum(([k]+SYN.get(k,[]) for k in SYN),[]) if norm(c)==norm(v))
            if score>best_score:
                best=t;best_score=score;best_cols=cols
        if best is None and self._tables():
            best=self._tables()[0];best_cols=self._cols(best)
        return best,best_cols
    def _build_map(self):
        m={}
        ncols={norm(c):c for c in self.cols}
        for k in CANON:
            candidates=[k]+SYN.get(k,[])
            hit=None
            for cand in candidates:
                nc=norm(cand)
                if nc in ncols:hit=ncols[nc];break
            m[k]=hit
        return m
    def _select_base(self):
        def qi(x): 
            return "\""+str(x).replace("\"","\"\"")+"\"" if x is not None else None
        cols=[self.map.get("Id") or self.cols[0],
              self.map.get("Title"),
              self.map.get("Price"),
              self.map.get("User_id"),
              self.map.get("profileName"),
              self.map.get("review_helpfulness"),
              self.map.get("review_score"),
              self.map.get("review_time"),
              self.map.get("review_summary"),
              self.map.get("review_text")]
        cols=[c for c in cols if c]
        qcols=[qi(c) for c in cols]
        return f"select {', '.join(qcols)} from {qi(self.table)}",cols
    def _apply_sql_filters(self,sql,params,q):
        def qi(x): 
            return "\""+str(x).replace("\"","\"\"")+"\"" if x is not None else None
        where=[]
        sm=q.get("score_min");sx=q.get("score_max")
        pf=q.get("price")
        if self.map.get("review_score"):
            if sm is not None:where.append(f"{qi(self.map['review_score'])}>=?");params.append(float(sm))
            if sx is not None:where.append(f"{qi(self.map['review_score'])}<=?");params.append(float(sx))
        if pf=="has" and self.map.get("Price"):where.append(f"{qi(self.map['Price'])} is not null")
        if pf=="missing" and self.map.get("Price"):where.append(f"{qi(self.map['Price'])} is null")
        if where:sql+=" where "+(" and ".join(where))
        sk=q.get("sort_key");so=q.get("sort_order","asc")
        if sk and sk in self.map and self.map[sk]:
            sql+=f" order by {qi(self.map[sk])} {'asc' if so!='desc' else 'desc'}"
        return sql,params
    def fetch(self,q,page,page_size):
        sql,cols=self._select_base()
        params=[]
        sql,params=self._apply_sql_filters(sql,params,q)
        rows=[]
        try:
            cur=self.conn.execute(sql,params)
            rows=[dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            cur=self.conn.execute(f"select * from \"{str(self.table).replace('\"','\"\"')}\"")
            rows=[dict(r) for r in cur.fetchall()]
        mapped=[]
        for r in rows:
            def get(k):
                c=self.map.get(k)
                return r[c] if c and c in r else None
            idv=get("Id")
            title=get("Title")
            price=get("Price")
            user_id=get("User_id")
            pname=get("profileName")
            helpful=get("review_helpfulness")
            score=get("review_score")
            timev=get("review_time")
            summary=get("review_summary")
            text=get("review_text")
            if score is None and self.derive_score:
                score=self._derive_score(" ".join([str(title or ""),str(summary or ""),str(text or "")]).strip())
            if summary is None and (self.map.get("review_summary") is None) and text:
                summary=str(text)[:120]
            hr=self._helpfulness_ratio(helpful)
            mv={"Id":str(idv) if idv is not None else "",
                "Title":str(title) if title is not None else "",
                "Price":float(price) if isinstance(price,(int,float)) else (None if price is None or str(price)=="" else float(str(price)) if str(price).replace('.','',1).isdigit() else None),
                "User_id":str(user_id) if user_id is not None else "",
                "profileName":str(pname) if pname is not None else "",
                "review/helpfulness":str(helpful) if helpful is not None else "",
                "helpfulness_ratio":hr,
                "review/score":float(score) if score is not None else None,
                "review/time":int(timev) if timev is not None and str(timev).isdigit() else None,
                "review/summary":str(summary) if summary is not None else "",
                "review/text":str(text) if text is not None else ""}
            mapped.append(mv)
        qtext=(q.get("q") or "").strip().lower()
        hmin=float(q.get("helpfulness") or 0)
        sm=q.get("score_min")
        sx=q.get("score_max")
        if qtext:
            mapped=[d for d in mapped if (d["Title"]+" "+d["review/summary"]+" "+d["review/text"]).lower().find(qtext)!=-1]
        if self.map.get("review_score") is None and (sm is not None or sx is not None):
            smin=float(sm) if sm is not None else float("-inf")
            smax=float(sx) if sx is not None else float("inf")
            mapped=[d for d in mapped if d["review/score"] is not None and d["review/score"]>=smin and d["review/score"]<=smax]
        if hmin>0:
            mapped=[d for d in mapped if d["helpfulness_ratio"] is not None and d["helpfulness_ratio"]>=hmin]
        total=len(mapped)
        start=max(0,(page-1)*page_size);end=start+page_size
        return {"total":total,"page":page,"page_size":page_size,"rows":mapped[start:end]}
    def stats(self,q):
        data=self.fetch(q,1,10**9)["rows"]
        total=len(data)
        uniq=len(set(d["Id"] for d in data))
        avg=0.0
        cnt=sum(1 for d in data if d["review/score"] is not None)
        if cnt>0:
            avg=sum(d["review/score"] or 0 for d in data)/cnt
        miss=sum(1 for d in data if d["Price"] is None)
        hist={str(i):sum(1 for d in data if d["review/score"]==i) for i in [1,2,3,4,5]}
        has_score=self.map.get("review_score") is not None or self.derive_score
        has_price=self.map.get("Price") is not None
        has_helpfulness=self.map.get("review_helpfulness") is not None
        has_summary=self.map.get("review_summary") is not None
        return {"total":total,"unique_products":uniq,"average_score":avg,"missing_prices":miss,"histogram":hist,"has_score":has_score,"has_price":has_price,"has_helpfulness":has_helpfulness,"has_summary":has_summary}
    def export_csv(self,q):
        data=self.fetch(q,1,10**9)["rows"]
        header=["Id","Title","Price","User_id","profileName","review/helpfulness","review/score","review/time","review/summary","review/text"]
        out=io.StringIO()
        w=csv.writer(out)
        w.writerow(header)
        for d in data:
            w.writerow([d["Id"],d["Title"],"" if d["Price"] is None else d["Price"],d["User_id"],d["profileName"],d["review/helpfulness"],"" if d["review/score"] is None else d["review/score"],"" if d["review/time"] is None else d["review/time"],d["review/summary"],d["review/text"]])
        return out.getvalue()
    def _helpfulness_ratio(self,val):
        if val is None:return None
        s=str(val)
        m=re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$",s)
        if not m:return None
        a=int(m.group(1));b=int(m.group(2))
        if b==0:return None
        return a/b
class Handler(SimpleHTTPRequestHandler):
    db=None
    def _text(self,txt,ctype="text/plain",code=200):
        data=txt.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def _json(self,obj,code=200):
        data=json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        p=urlparse(self.path)
        if p.path.startswith("/api/"):
            if not os.path.exists(DB_PATH):
                self._json({"error":"database_not_found"},404);return
            if Handler.db is None:
                Handler.db=DB(DB_PATH)
            q=parse_qs(p.query)
            def getq(k,default=None):
                v=q.get(k)
                return v[0] if v else default
            if p.path=="/api/datacard":
                if os.path.exists(DATA_CARD_PATH):
                    try:
                        with open(DATA_CARD_PATH,"r",encoding="utf-8") as f:
                            txt=f.read()
                        items=[]
                        for line in txt.splitlines():
                            line=line.strip()
                            if not line: 
                                continue
                            parts=re.split(r"\s{2,}",line)
                            name=parts[0].strip()
                            desc=parts[1].strip() if len(parts)>1 else ""
                            items.append({"name":name,"description":desc})
                        self._json({"items":items});return
                    except Exception:
                        self._json({"items":[]});return
                self._json({"items":[]});return
            if p.path=="/api/stats":
                rq={"q":getq("q"),"score_min":getq("score_min"),"score_max":getq("score_max"),"helpfulness":getq("helpfulness"),"price":getq("price")}
                self._json(Handler.db.stats(rq));return
            if p.path=="/api/reviews":
                page=int(getq("page","1"));ps=int(getq("page_size","25"))
                rq={"q":getq("q"),"score_min":getq("score_min"),"score_max":getq("score_max"),"helpfulness":getq("helpfulness"),"price":getq("price"),"sort_key":getq("sort_key"),"sort_order":getq("sort_order")}
                self._json(Handler.db.fetch(rq,page,ps));return
            if p.path=="/api/predict":
                model_path=os.path.join(BASE_DIR,"models","reviews_cluster.pkl")
                def rows_all(limit,offset):
                    sql,cols=Handler.db._select_base()
                    try:
                        cur=Handler.db.conn.execute(sql+f" limit {int(limit)} offset {int(offset)}")
                        return [dict(r) for r in cur.fetchall()]
                    except sqlite3.OperationalError:
                        cur=Handler.db.conn.execute(f"select * from \"{str(Handler.db.table).replace('\"','\"\"')}\" limit {int(limit)} offset {int(offset)}")
                        return [dict(r) for r in cur.fetchall()]
                limit=int(getq("limit","1000"));offset=int(getq("offset","0"))
                raw_rows=rows_all(limit,offset)
                rows=[]
                ids=[]
                for r in raw_rows:
                    def g(k):
                        c=Handler.db.map.get(k)
                        return r.get(c) if c else None
                    idv=g("Id")
                    title=g("Title") or ""
                    summary=g("review_summary") or ""
                    text=g("review_text") or ""
                    score=g("review_score")
                    helpful=g("review_helpfulness")
                    price=g("Price")
                    row={
                        "Id":str(idv or ""),
                        "Title":str(title or ""),
                        "review/summary":str(summary or ""),
                        "review/text":str(text or ""),
                        "review/score":score,
                        "review/helpfulness":helpful,
                        "Price":price
                    }
                    ids.append(row["Id"])
                    rows.append(row)
                if not os.path.exists(model_path):
                    self._json({"error":"model_not_trained"},400);return
                model=joblib.load(model_path)
                vec=model["vectorizer"];scaler=model["scaler"];svd=model["svd"];cluster=model["cluster"]
                X=build_features(rows,vec,scaler,svd,fit=False)
                labels=cluster.predict(X)
                self._json({"ids":ids,"labels":[int(l) for l in labels]});return
            if p.path=="/api/export":
                rq={"q":getq("q"),"score_min":getq("score_min"),"score_max":getq("score_max"),"helpfulness":getq("helpfulness"),"price":getq("price")}
                csv_data=Handler.db.export_csv(rq)
                data=csv_data.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type","text/csv")
                self.send_header("Content-Disposition","attachment; filename=filtered_reviews.csv")
                self.send_header("Content-Length",str(len(data)))
                self.end_headers()
                self.wfile.write(data);return
        return super().do_GET()
def main():
    os.chdir(BASE_DIR)
    port=int(os.environ.get("PORT","8000"))
    srv=ThreadingHTTPServer(("0.0.0.0",port),Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
if __name__=="__main__":
    main()
