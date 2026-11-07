# PulmScope
'''
Full pipeline: questionnaire (weighted score + ML model) → CT classifier → segmentation → decision logic → FastAPI + web page

How to run:
1. In the "Settings" cell, only check `MODELS_DIR` and the names of the three model files.
2. Run the cells from top to bottom.
3. After the "Run" cell, a **link** appears, and the web page itself is shown right below it.

- No token, sign-up, ngrok or Cloudflare is needed. The page is opened through Colab's own proxy.

Warning: This tool is for education only and is not a medical diagnosis.
'''

# 1) Settings
# Models folder on Drive
MODELS_DIR = '/content/PulmoScope-main/PulmoScope-main/Models'
CLS_MODEL_FILE = 'dl_classification_model.keras'      # classifier
SEG_MODEL_FILE = 'dl_segmentation_model.keras'        # segmentation
ML_MODEL_FILE  = 'ml_model.pkl'                       # ML model

# Thresholds
CLS_THRESHOLD = 0.5
CLS_THRESHOLD_HIGH_RISK = 0.4
SEG_THRESHOLD = 0.5
SEG_MIN_AREA = 8

# Upload limits
MAX_UPLOAD_MB = 10
MIN_SIDE_PX = 64
MAX_COLORFULNESS = 25


## 2) Load the models (once)
import os, threading
import joblib
import tensorflow as tf

# classifier & segmentation
CLS_MODEL = tf.keras.models.load_model(os.path.join(MODELS_DIR, CLS_MODEL_FILE), compile=False)
SEG_MODEL = tf.keras.models.load_model(os.path.join(MODELS_DIR, SEG_MODEL_FILE), compile=False)
print("Classifier:", CLS_MODEL.input_shape, "| Segmentation:", SEG_MODEL.input_shape)

# Questionnaire ML model
ML = None
ML_LOAD_ERROR = None
try:
    ML = joblib.load(os.path.join(MODELS_DIR, ML_MODEL_FILE))
    print("Questionnaire ML model loaded. Features:", ML["feature_names"])
except Exception as e:
    ML_LOAD_ERROR = f"{type(e).__name__}: {e}"
    print("ML model not loaded (the app works without it):", ML_LOAD_ERROR)

MODEL_LOCK = threading.Lock()


# 3) Stage 1: questionnaire
'''
Two outputs side by side:
- Weighted score (0 to 11.1) → risk word: Low / Medium / High (the same weights as in the ML notebook)
- ML model estimate (the saved GradientBoosting model) → a probability, for display only

The risk word comes from the weighted score and affects the classifier threshold.
'''

from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    gender: Literal["male", "female"]
    age: int = Field(ge=1, le=120)
    wheezing: bool
    alcohol: bool
    peer_pressure: bool

# Weights derived from the feature importances in the ML notebook (sum = 11.1)
WEIGHTS = {"gender": 1.0, "wheezing": 1.2, "alcohol": 1.9, "peer_pressure": 2.6, "age": 4.4}
MAX_SCORE = round(sum(WEIGHTS.values()), 1)   # 11.1

def feature_values(d: PatientInput) -> dict:
    """Map every answer to the 0..1 range."""
    return {
        "gender": float(d.gender == "male"),
        "wheezing": float(d.wheezing),
        "alcohol": float(d.alcohol),
        "peer_pressure": float(d.peer_pressure),
        "age": min(d.age, 100) / 100,
    }

def calculate_weighted_risk(d: PatientInput):
    v = feature_values(d)
    score = sum(v[k] * WEIGHTS[k] for k in WEIGHTS)
    risk = "Low" if score <= 3.7 else "Medium" if score <= 7.4 else "High"
    return round(float(score), 2), risk

# ML model: same encoding as the dataset (gender: male=1/female=0; yes=2/no=1)
_KEY_BY_NAME = {"GENDER": "gender", "AGE": "age", "WHEEZING": "wheezing",
                "ALCOHOL_CONSUMING": "alcohol", "PEER_PRESSURE": "peer_pressure"}
def _norm(name: str) -> str:
    return name.strip().upper().replace(" ", "_")

def ml_probability(d: PatientInput):
    """P(YES) from the GradientBoosting model; None if the model is not loaded."""
    if ML is None:
        return None
    raw = {"GENDER": 1 if d.gender == "male" else 0, "AGE": d.age,
           "WHEEZING": 2 if d.wheezing else 1,
           "ALCOHOL_CONSUMING": 2 if d.alcohol else 1,
           "PEER_PRESSURE": 2 if d.peer_pressure else 1}
    names = ML["feature_names"]
    row = pd.DataFrame([{n: raw[_norm(n)] for n in names}])[names]
    x = ML["scaler"].transform(row)
    model = ML["model"]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    idx = next((i for i, c in enumerate(classes) if str(c).strip().upper() in ("YES", "1", "TRUE")), len(classes) - 1)
    return round(float(proba[idx]), 3)

def ml_importances() -> dict:
    if ML is None or not hasattr(ML["model"], "feature_importances_"):
        return {}
    return {_KEY_BY_NAME[_norm(n)]: round(float(v), 3)
            for n, v in zip(ML["feature_names"], ML["model"].feature_importances_)
            if _norm(n) in _KEY_BY_NAME}

def questionnaire_report(d: PatientInput) -> dict:
    score, risk = calculate_weighted_risk(d)
    v, imp = feature_values(d), ml_importances()
    ml_p, ml_err = None, ML_LOAD_ERROR
    try:
        ml_p = ml_probability(d)
    except Exception as e:
        ml_err = f"{type(e).__name__}: {e}"
    return {
        "simulated_risk_score": score, "max_score": MAX_SCORE, "risk_level": risk,
        "ml_probability": ml_p, "ml_error": None if ml_p is not None else ml_err,
        "factors": [{"key": k, "value": round(v[k], 3), "weight": WEIGHTS[k],
                     "points": round(v[k] * WEIGHTS[k], 2), "ml_importance": imp.get(k)} for k in WEIGHTS],
    }


# 4) Input image processing
import io
import numpy as np
import cv2
from PIL import Image

def open_image(raw: bytes) -> Image.Image:
    if len(raw) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise ValueError("The file is not a valid JPG/PNG image.")
    return img

def to_rgb(img: Image.Image) -> Image.Image:
    """Convert RGBA, palette and 16-bit images to 8-bit RGB."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(img).astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")

def validate_ct_like(rgb: Image.Image):
    w, h = rgb.size
    if min(w, h) < MIN_SIDE_PX:
        raise ValueError(f"Image is too small (minimum side {MIN_SIDE_PX}px).")
    a = np.asarray(rgb.resize((128, 128))).astype(np.float32)
    colorfulness = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
                    + np.abs(a[..., 0] - a[..., 2])).mean() / 3
    if colorfulness > MAX_COLORFULNESS:
        raise ValueError("The image looks like a color photo. Please upload a chest CT slice (grayscale JPG/PNG).")

# Classifier: 224 resize with nearest + ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def prep_classifier(rgb: Image.Image) -> np.ndarray:
    x = np.asarray(rgb.resize((224, 224), Image.NEAREST)).astype(np.float32) / 255.0
    return (x - _MEAN) / _STD

# Segmentation: grayscale 512x512, values 0..255, repeated across three channels
def prep_segmentation(rgb: Image.Image):
    gray = np.asarray(rgb.convert("L")).astype(np.float32)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = np.clip(gray, 0, 255)
    return np.stack([gray] * 3, axis=-1), gray


# 5) Classifier + segmentation + decision logic + red mask
import base64

DISCLAIMER = ("This tool is for education/simulation only and is NOT a medical diagnosis. "
              "A single CT slice cannot rule a condition in or out. Consult a physician.")

def find_regions(prob_map: np.ndarray):
    binary = (prob_map > SEG_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    regions = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < SEG_MIN_AREA:
            continue
        comp = labels == k
        clean[comp] = 1
        regions.append({
            "x": int(stats[k, cv2.CC_STAT_LEFT]), "y": int(stats[k, cv2.CC_STAT_TOP]),
            "width": int(stats[k, cv2.CC_STAT_WIDTH]), "height": int(stats[k, cv2.CC_STAT_HEIGHT]),
            "area_px": area, "max_probability": round(float(prob_map[comp].max()), 3),
        })
    return clean, regions

def make_overlay(gray512: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = np.stack([gray512] * 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    base[m] = 0.5 * base[m] + 0.5 * np.array([255, 0, 0], dtype=np.float32)
    return Image.fromarray(base.astype(np.uint8))

def decide(cls_positive: bool, seg_positive: bool):
    if cls_positive and seg_positive:
        return "suspicious", "Suspicious", "Both the classifier and the segmentation model flagged this slice."
    if cls_positive or seg_positive:
        who = "classifier" if cls_positive else "segmentation model"
        return "uncertain", "Uncertain - needs review", f"Only the {who} flagged this slice; the two models disagree."
    return "no_findings", "No findings in this slice", "Neither model flagged this slice (other slices may differ)."

def analyze(patient: PatientInput, pil_img: Image.Image) -> dict:
    # level 1: questionnaire
    q = questionnaire_report(patient)

    # Prepare the image
    rgb = to_rgb(pil_img)
    validate_ct_like(rgb)
    x_cls = prep_classifier(rgb)
    x_seg, gray512 = prep_segmentation(rgb)

    # level 2 and 4: both models always run
    with MODEL_LOCK:
        p_normal = float(CLS_MODEL.predict(x_cls[None], verbose=0)[0, 0])
        prob_map = SEG_MODEL.predict(x_seg[None], verbose=0)[0, ..., 0]

    # Training used classes=['Abnormal','Normal'] -> the sigmoid output = P(Normal)
    p_abnormal = 1.0 - p_normal
    mask, regions = find_regions(prob_map)

    # level 3: decision
    cls_thr = CLS_THRESHOLD_HIGH_RISK if q["risk_level"] == "High" else CLS_THRESHOLD
    cls_positive = p_abnormal >= cls_thr
    seg_positive = len(regions) > 0
    code, decision, reason = decide(cls_positive, seg_positive)

    return {
        "decision": decision, "decision_code": code, "reason": reason,
        "questionnaire": q,
        "classifier": {"abnormal_probability": round(p_abnormal, 3),
                       "threshold_used": cls_thr, "flagged": bool(cls_positive)},
        "segmentation": {"flagged": bool(seg_positive), "regions": regions,
                         "note": "Region coordinates are in a 512x512 resized frame."},
        "disclaimer": DISCLAIMER,
        "overlay_image": make_overlay(gray512, mask),
    }

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


## 6) Web page
'''
A fully self-contained HTML page (no CDN or external fonts)
that FastAPI serves at the root `/`.
'''
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PulmScope | Educational Lung Nodule Screening</title>
<style>
:root{
  --bg:#f3f6fb; --card:#ffffff; --text:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --brand:#2563eb; --brand2:#1d4ed8; --soft:#eef4ff;
  --ok:#16a34a; --warn:#d97706; --bad:#dc2626; --radius:18px;
  --shadow:0 6px 24px rgba(15,23,42,.07);
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0b1220; --card:#121a2b; --text:#e5e7eb; --muted:#94a3b8; --line:#243049; --soft:#16233d; --shadow:0 6px 24px rgba(0,0,0,.35); }
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;line-height:1.7}
.wrap{max-width:1040px;margin:0 auto;padding:18px 16px 60px}
header{background:linear-gradient(135deg,#1e40af,#2563eb 55%,#0ea5e9);color:#fff;border-radius:var(--radius);padding:26px 26px 22px;box-shadow:var(--shadow)}
header h1{margin:0;font-size:28px;letter-spacing:.3px}
header p{margin:6px 0 0;opacity:.92;font-size:15px}
.badge{display:inline-block;margin-top:14px;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);padding:4px 12px;border-radius:999px;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow)}
.card h2{margin:0 0 14px;font-size:18px;display:flex;align-items:center;gap:10px}
.step{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--brand);color:#fff;font-size:14px}
.q{padding:12px 0;border-top:1px dashed var(--line)}
.q:first-of-type{border-top:0}
.q label.t{display:block;font-weight:600;margin-bottom:8px}
.q small{display:block;color:var(--muted);font-weight:400}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--text);padding:8px 22px;font:inherit;cursor:pointer;transition:.15s}
.seg button+button{border-inline-start:1px solid var(--line)}
.seg button.on{background:var(--brand);color:#fff}
.age-row{display:flex;gap:12px;align-items:center}
.age-row input[type=range]{flex:1;accent-color:var(--brand)}
.age-row input[type=number]{width:76px;padding:7px 8px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--text);font:inherit;text-align:center}
#drop{border:2px dashed var(--line);border-radius:14px;min-height:250px;display:grid;place-items:center;text-align:center;padding:14px;cursor:pointer;background:var(--soft);transition:.15s}
#drop.over{border-color:var(--brand);background:rgba(37,99,235,.12)}
#drop .hint{color:var(--muted);font-size:14px}
#drop .big{font-size:40px;line-height:1}
#preview{max-width:100%;max-height:280px;border-radius:10px;display:none;background:#000}
#fname{margin-top:8px;color:var(--muted);font-size:13px;direction:ltr;text-align:center;word-break:break-all}
.tip{margin-top:10px;color:var(--muted);font-size:13px}
button.go{width:100%;margin-top:14px;padding:14px;border:0;border-radius:14px;background:var(--brand);color:#fff;font-weight:600;font-size:17px;font-family:inherit;cursor:pointer;display:flex;gap:10px;align-items:center;justify-content:center;transition:.15s}
button.go:hover:not(:disabled){background:var(--brand2)}
button.go:disabled{opacity:.45;cursor:not-allowed}
.spin{width:18px;height:18px;border:3px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:sp .8s linear infinite;display:none}
.loading .spin{display:inline-block}
@keyframes sp{to{transform:rotate(360deg)}}
.err{display:none;margin-top:14px;padding:12px 14px;border-radius:12px;background:rgba(220,38,38,.1);border:1px solid rgba(220,38,38,.35);color:var(--bad)}
#result{display:none;margin-top:18px}
.banner{border-radius:var(--radius);padding:22px 24px;color:#fff;box-shadow:var(--shadow)}
.banner h3{margin:0;font-size:24px}
.banner p{margin:6px 0 0;opacity:.95}
.suspicious{background:linear-gradient(135deg,#b91c1c,#ef4444)}
.uncertain{background:linear-gradient(135deg,#b45309,#f59e0b)}
.no_findings{background:linear-gradient(135deg,#15803d,#22c55e)}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}
@media(max-width:820px){.metrics{grid-template-columns:1fr}}
.metric h4{margin:0 0 10px;font-size:15px;color:var(--muted);font-weight:600}
.big-n{font-size:30px;font-weight:700;line-height:1.2}
.pill{display:inline-block;padding:2px 12px;border-radius:999px;color:#fff;font-size:14px;font-weight:600}
.pill.Low{background:var(--ok)} .pill.Medium{background:var(--warn)} .pill.High{background:var(--bad)}
.bar{position:relative;height:12px;border-radius:99px;background:var(--line);margin:14px 0 6px;overflow:visible}
.bar .fill{position:absolute;inset-block:0;inset-inline-start:0;border-radius:99px;background:var(--brand)}
.bar .tick{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--text);opacity:.7}
.bar.zones{background:linear-gradient(to right,var(--ok) 0 33.33%,var(--warn) 33.33% 66.66%,var(--bad) 66.66% 100%);opacity:.9}
.bar.zones .mark{position:absolute;top:-6px;width:4px;height:24px;background:var(--text);border-radius:3px;transform:translateX(-50%)}
.sub{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:6px}
th,td{padding:8px 6px;border-bottom:1px solid var(--line);text-align:start}
th{color:var(--muted);font-weight:600}
td.num{direction:ltr;text-align:center}
.mini{height:8px;background:var(--line);border-radius:9px;min-width:70px;position:relative}
.mini i{position:absolute;inset-block:0;inset-inline-start:0;background:var(--brand);border-radius:9px}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media(max-width:820px){.imgs{grid-template-columns:1fr}}
.imgs figure{margin:0}
.imgs img{width:100%;border-radius:12px;background:#000;display:block;object-fit:contain;max-height:520px}
.imgs figcaption{margin-top:6px;color:var(--muted);font-size:13px;text-align:center}
details{margin-top:12px}
summary{cursor:pointer;color:var(--brand);font-weight:600}
footer{margin-top:22px;padding:14px 16px;border-radius:14px;background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.35);font-size:13.5px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1> PulmScope</h1>
    <p>Lung nodule screening simulation: questionnaire + two deep-learning models (classifier and segmentation)</p>
    <span class="badge">Educational only – not a medical diagnosis</span>
  </header>

  <div class="grid">
    <section class="card">
      <h2><span class="step">1</span> Questionnaire</h2>

      <div class="q"><label class="t">What is your gender?</label>
        <div class="seg" data-key="gender"><button type="button" data-val="male" class="on">Male</button><button type="button" data-val="female">Female</button></div></div>

      <div class="q"><label class="t">How old are you?</label>
        <div class="age-row"><input type="range" id="ageRange" min="1" max="120" value="50"><input type="number" id="ageNum" min="1" max="120" value="50"></div></div>

      <div class="q"><label class="t">Do you have wheezing?<small>A whistling sound when breathing</small></label>
        <div class="seg" data-key="wheezing"><button type="button" data-val="true">Yes</button><button type="button" data-val="false" class="on">No</button></div></div>

      <div class="q"><label class="t">Do you consume alcohol?<small>Alcohol consuming</small></label>
        <div class="seg" data-key="alcohol"><button type="button" data-val="true">Yes</button><button type="button" data-val="false" class="on">No</button></div></div>

      <div class="q"><label class="t">Are you under peer pressure?<small>Peer pressure</small></label>
        <div class="seg" data-key="peer_pressure"><button type="button" data-val="true">Yes</button><button type="button" data-val="false" class="on">No</button></div></div>
    </section>

    <section class="card">
      <h2><span class="step">2</span> Upload a CT image</h2>
      <div id="drop">
        <div id="dropInner">
          <div class="big">🩻</div>
          <div><b>Drag the image here</b> or click to choose</div>
          <div class="hint">JPG or PNG only, up to 10 MB</div>
        </div>
        <img id="preview" alt="Preview">
      </div>
      <input type="file" id="file" accept="image/png,image/jpeg" hidden>
      <div id="fname"></div>
      <div class="tip">Upload a grayscale chest CT slice. Color photos, X-rays or phone pictures will not give valid results.</div>
      <button class="go" id="go" disabled><span class="spin"></span><span id="goText">Analyze</span></button>
      <div class="err" id="err"></div>
    </section>
  </div>

  <section id="result">
    <div class="banner" id="banner"><h3 id="bTitle"></h3><p id="bText"></p></div>

    <div class="metrics">
      <div class="card metric"><h4>1) Questionnaire (simulated risk)</h4>
        <div id="mQ"></div></div>
      <div class="card metric"><h4>2) Image classifier</h4>
        <div id="mC"></div></div>
      <div class="card metric"><h4>3) Segmentation</h4>
        <div id="mS"></div></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h4 style="margin:0 0 6px;color:var(--muted)">Weights and each answer's contribution to the questionnaire score</h4>
      <div id="factors"></div>
    </div>

    <div class="imgs">
      <figure><img id="imgOrig" alt="Original image"><figcaption>Original image</figcaption></figure>
      <figure><img id="imgOver" alt="Suspicious region"><figcaption>Suspicious region (red)</figcaption></figure>
    </div>
  </section>

  <footer id="disc">⚠️ This tool is for education and simulation only and is NOT a medical diagnosis. A single CT slice cannot rule a condition in or out; please consult a physician.</footer>
</div>

<script>
(function(){
  "use strict";
  var $ = function(s){ return document.querySelector(s); };
  var state = { gender:"male", age:50, wheezing:false, alcohol:false, peer_pressure:false, file:null, url:null };
  var MAX_BYTES = 10 * 1024 * 1024;

  var LABELS = { gender:"Gender", age:"Age", wheezing:"Wheezing", alcohol:"Alcohol consumption", peer_pressure:"Peer pressure" };
  var RISK_TXT = { Low:"Low", Medium:"Medium", High:"High" };
  var DECISIONS = {
    suspicious:  { title:"Suspicious", text:"Both models (classifier and segmentation) flagged this slice. Please consult a physician." },
    uncertain:   { title:"Uncertain – needs review", text:"Only one of the two models flagged this slice, so the models disagree." },
    no_findings: { title:"No suspicious findings in this slice", text:"Neither model flagged this slice. This does not mean the whole lung is healthy." }
  };

  function fa(n, d){ return Number(n).toLocaleString("en-US", { minimumFractionDigits:d||0, maximumFractionDigits:d||0 }); }
  function pct(x){ return fa(x * 100, 1) + "%"; }
  function esc(s){ return String(s).replace(/[&<>"']/g, function(c){ return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]; }); }

  /* ---------- Questionnaire ---------- */
  document.querySelectorAll(".seg").forEach(function(seg){
    seg.addEventListener("click", function(e){
      var b = e.target.closest("button"); if(!b) return;
      seg.querySelectorAll("button").forEach(function(x){ x.classList.remove("on"); });
      b.classList.add("on");
      var v = b.dataset.val;
      state[seg.dataset.key] = (v === "true") ? true : (v === "false") ? false : v;
    });
  });
  var ageR = $("#ageRange"), ageN = $("#ageNum");
  function setAge(v){ v = Math.max(1, Math.min(120, parseInt(v, 10) || 1)); state.age = v; ageR.value = v; ageN.value = v; }
  ageR.addEventListener("input", function(){ setAge(ageR.value); });
  ageN.addEventListener("change", function(){ setAge(ageN.value); });

  /* ---------- Upload ---------- */
  var drop = $("#drop"), input = $("#file"), go = $("#go"), err = $("#err");
  function showErr(msg){ err.textContent = msg; err.style.display = "block"; }
  function clearErr(){ err.style.display = "none"; }
  function setFile(f){
    clearErr();
    if(!f) return;
    if(["image/jpeg","image/png"].indexOf(f.type) < 0){ showErr("Only JPG or PNG files are accepted."); return; }
    if(f.size > MAX_BYTES){ showErr("The file is larger than 10 MB."); return; }
    if(state.url) URL.revokeObjectURL(state.url);
    state.file = f; state.url = URL.createObjectURL(f);
    var p = $("#preview"); p.src = state.url; p.style.display = "block"; $("#dropInner").style.display = "none";
    $("#fname").textContent = f.name + "  (" + Math.round(f.size / 1024) + " KB)";
    go.disabled = false;
  }
  drop.addEventListener("click", function(){ input.click(); });
  input.addEventListener("change", function(){ setFile(input.files && input.files[0]); });
  ["dragenter","dragover"].forEach(function(ev){ drop.addEventListener(ev, function(e){ e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave","drop"].forEach(function(ev){ drop.addEventListener(ev, function(e){ e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function(e){ setFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]); });

  /* ---------- Send to API ---------- */
  function errorText(data, status){
    if(data && typeof data.detail === "string") return data.detail;
    if(data && Array.isArray(data.detail)) return data.detail.map(function(d){ return (d.loc ? d.loc.join(".") + ": " : "") + d.msg; }).join(" | ");
    return "Server error (" + status + ")";
  }
  go.addEventListener("click", function(){
    if(!state.file) return;
    clearErr();
    go.disabled = true; go.classList.add("loading"); $("#goText").textContent = "Analyzing…";
    var fd = new FormData();
    fd.append("gender", state.gender); fd.append("age", state.age);
    fd.append("wheezing", state.wheezing); fd.append("alcohol", state.alcohol); fd.append("peer_pressure", state.peer_pressure);
    fd.append("image", state.file);
    fetch("predict", { method:"POST", body:fd })
      .then(function(res){ return res.json().catch(function(){ return null; }).then(function(data){ return { ok:res.ok, status:res.status, data:data }; }); })
      .then(function(r){
        if(!r.ok) throw new Error(errorText(r.data, r.status));
        render(r.data);
      })
      .catch(function(e){ showErr("⚠️ " + (e && e.message ? e.message : "Could not connect to the server.")); })
      .then(function(){ go.disabled = false; go.classList.remove("loading"); $("#goText").textContent = "Analyze"; });
  });

  /* ---------- Show result ---------- */
  function answerText(key, f){
    if(key === "gender") return state.gender === "male" ? "Male" : "Female";
    if(key === "age") return fa(state.age) + " years";
    return state[key] ? "Yes" : "No";
  }
  function render(r){
    var d = DECISIONS[r.decision_code] || { title:r.decision, text:r.reason };
    var banner = $("#banner"); banner.className = "banner " + r.decision_code;
    $("#bTitle").textContent = d.title; $("#bText").textContent = d.text;

    var q = r.questionnaire, c = r.classifier, s = r.segmentation;
    var qScore = Math.max(0, Math.min(100, q.simulated_risk_score / q.max_score * 100));
    var mlLine = (q.ml_probability === null || q.ml_probability === undefined)
      ? '<div class="sub">Machine-learning model: ' + (q.ml_error ? esc(q.ml_error) : "not loaded") + '</div>'
      : '<div class="sub">Machine-learning estimate (GradientBoosting): <b>' + pct(q.ml_probability) + '</b><br>This number depends on the training-data distribution and is not a real probability of disease.</div>';
    $("#mQ").innerHTML =
      '<div class="big-n"><span class="pill ' + esc(q.risk_level) + '">' + (RISK_TXT[q.risk_level] || esc(q.risk_level)) + '</span></div>' +
      '<div class="bar zones"><span class="mark" style="inset-inline-start:' + qScore + '%"></span></div>' +
      '<div class="sub">Score ' + fa(q.simulated_risk_score, 2) + ' of ' + fa(q.max_score, 1) + '</div>' + mlLine;

    $("#mC").innerHTML =
      '<div class="big-n">' + pct(c.abnormal_probability) + '</div>' +
      '<div class="bar"><span class="fill" style="width:' + (c.abnormal_probability * 100) + '%"></span><span class="tick" style="inset-inline-start:' + (c.threshold_used * 100) + '%"></span></div>' +
      '<div class="sub">Probability that the slice is abnormal • threshold ' + pct(c.threshold_used) + '<br><b>' + (c.flagged ? "Flagged ⚠️" : "Not flagged ✓") + '</b></div>';

    $("#mS").innerHTML =
      '<div class="big-n">' + fa(s.regions.length) + (s.regions.length === 1 ? ' region' : ' regions') + '</div>' +
      '<div class="sub">' + (s.flagged ? "The suspicious region is marked in red on the image." : "No suspicious region found.") + '</div>' +
      (s.regions.length ? '<details><summary>Region details</summary><table><tr><th>#</th><th>Position (x,y)</th><th>Size</th><th>Probability</th></tr>' +
        s.regions.map(function(g, i){ return '<tr><td>' + fa(i + 1) + '</td><td class="num">' + g.x + ',' + g.y + '</td><td class="num">' + g.width + '×' + g.height + '</td><td class="num">' + g.max_probability + '</td></tr>'; }).join("") +
        '</table><div class="sub">Coordinates are in a 512×512 frame.</div></details>' : "");

    var hasML = q.factors.some(function(f){ return f.ml_importance !== null && f.ml_importance !== undefined; });
    $("#factors").innerHTML = '<table><tr><th>Feature</th><th>Your answer</th><th>Weight</th><th>Points</th>' + (hasML ? '<th>Importance in ML model</th>' : '') + '</tr>' +
      q.factors.map(function(f){
        return '<tr><td>' + (LABELS[f.key] || esc(f.key)) + '</td><td>' + answerText(f.key, f) + '</td><td class="num">' + fa(f.weight, 1) + '</td><td class="num">' + fa(f.points, 2) + '</td>' +
          (hasML ? '<td><div class="mini"><i style="width:' + ((f.ml_importance || 0) * 100) + '%"></i></div><span class="sub">' + pct(f.ml_importance || 0) + '</span></td>' : '') + '</tr>';
      }).join("") + '</table><div class="sub">Points = (value normalized to 0–1) × weight. Up to 3.7 is Low, up to 7.4 is Medium, above that is High.</div>';

    $("#imgOrig").src = state.url;
    $("#imgOver").src = "data:image/png;base64," + r.overlay_png_base64;
    var res = $("#result"); res.style.display = "block";
    res.scrollIntoView({ behavior:"smooth", block:"start" });
  }
})();
</script>
</body>
</html>
"""


# 7) FastAPI: connecting all the stages
'''
- `GET /` → the web page
- `POST /predict` → questionnaire + image → JSON result (red mask as a base64 PNG)
- `GET /health` → health check
'''
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(title="PulmScope API", version="2.0")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE_HTML

@app.get("/health")
def health():
    return {"status": "ok", "classifier_loaded": CLS_MODEL is not None,
            "segmentation_loaded": SEG_MODEL is not None, "ml_loaded": ML is not None}

@app.post("/predict")
def predict(
    gender: Literal["male", "female"] = Form(...),
    age: int = Form(..., ge=1, le=120),
    wheezing: bool = Form(...),
    alcohol: bool = Form(...),
    peer_pressure: bool = Form(...),
    image: UploadFile = File(...),
):

    raw = image.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    patient = PatientInput(gender=gender, age=age, wheezing=wheezing,
                           alcohol=alcohol, peer_pressure=peer_pressure)
    try:
        result = analyze(patient, open_image(raw))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["overlay_png_base64"] = pil_to_base64_png(result.pop("overlay_image"))
    return result


# 8) Run: server + open the page
'''
After running this cell, **click the blue link** (new tab) or use the page shown below the cell.
'''
import time, threading, requests, uvicorn

if globals().get("server") is not None:
    server.should_exit = True
    for _ in range(20):
        if not server_thread.is_alive():
            break
        time.sleep(0.5)

server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()

BASE_URL = "http://localhost:8000"
for _ in range(40):
    try:
        if requests.get(BASE_URL + "/health", timeout=2).status_code == 200:
            break
    except Exception:
        time.sleep(0.5)
else:
    raise RuntimeError("The server did not start; check the output of the previous cells.")
print("Server is ready:", requests.get(BASE_URL + "/health").json())

from google.colab import output
try:
    output.serve_kernel_port_as_window(8000, path="/", anchor_text="🫁 Open PulmScope in a new tab (click here)")
    output.serve_kernel_port_as_iframe(8000, path="/", height=1100)
except Exception as e:
    from google.colab.output import eval_js
    print("Automatic display failed:", e)
    print("Open this address manually:", eval_js("google.colab.kernel.proxyPort(8000)"))
