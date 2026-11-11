import io, os, re, datetime, joblib, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageEnhance
import pytesseract
from train import clean

# Setup Tesseract path automatically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tess_candidates = [
    os.path.join(BASE_DIR, '..', 'tesseract', 'tesseract.exe'),
    os.path.join(BASE_DIR, 'tesseract', 'tesseract.exe'),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Lenovo\Downloads\spam_prototype\tesseract\tesseract.exe"
]
for p in tess_candidates:
    p = os.path.abspath(p)
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        tessdata = os.path.join(os.path.dirname(p), 'tessdata')
        if os.path.exists(tessdata):
            os.environ['TESSDATA_PREFIX'] = tessdata
        break

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
LABELS = ['normal', 'spam']
ADVICE = {
    'normal': ('เสี่ยงต่ำ', 'ดูปกติ แต่ยังไม่ควรให้ข้อมูลส่วนตัวหากไม่แน่ใจผู้ส่ง'),
    'spam': ('เสี่ยงสูง (SPAM)', 'อย่าติดต่อ อย่ากดลิงก์ อย่าโอนเงิน/ให้ OTP และบล็อก/รายงานผู้ส่ง'),
}

TRUSTED_DOMAIN_SUFFIXES = ('.ac.th', '.go.th', '.or.th', '.edu')
TRUSTED_EXACT_HOSTS = {
    'classroom.google.com', 'meet.google.com', 'zoom.us', 'teams.microsoft.com'
}
HIGH_RISK_RE = re.compile(
    r'(?:bit\.ly|tinyurl\.com|cutt\.ly|t\.co/|shorturl\.at|is\.gd|rb\.gy|example\.invalid|\[url\]'
    r'|\botp\b|รหัส\s*pin|รหัสผ่าน|ปลดล็อก|บัญชีถูกระงับ|บัญชี.*ผิดปกติ|ยืนยันตัวตน.*(?:ด่วน|ทันที|ลิงก์)|กรอกเลขบัญชี'
    r'|กู้เงิน|ดอกเบี้ย\s*0|ค่าดำเนินการ|ค่าประกัน|โอนก่อน|รีบโอน|โอนมัดจำ|โอนเต็ม|รับเงินคืน'
    r'|รายได้วันละ|ได้เงินวันละ|ทุนพิเศษ|แจกฟรี|(?:ได้รับรางวัล|กดรับรางวัล|รับรางวัลใหญ่|รับรางวัลฟรี|แจกรางวัล)|พนัน|สล็อต|บาคาร่า'
    r'|คลิกลิงก์|กดลิงก์|แอดไลน์|line\s*[:：]\s*@|(?:^|[\s:])@(?!all\b|everyone\b|here\b)[a-z0-9_.-]{2,})'
)
SHORT_CASUAL_RISK_RE = re.compile(
    r'(?:\d{3,}|บาท|฿|\bline\b|\bdm\b|http|www\.|โอน|จ่าย|ชำระ|กู้|ทุน|พัสดุ|บัญชี|รหัส'
    r'|ด่วน|รีบ|ทันที|ฟรี|สมัคร|งาน|ขาย|ซื้อ|จอง|แจก|คลิก|กด|กรอก|ตรวจสอบ|ระงับ|สิทธิ|รางวัล)'
)

def extract_domains(cleaned):
    candidates = re.findall(r'(?:https?://)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)', cleaned)
    return [d for d in candidates if not re.fullmatch(r'[0-9.]+', d)]

def is_trusted_domain(dom):
    if dom.endswith(TRUSTED_DOMAIN_SUFFIXES):
        return True
    return any(dom == h or dom.endswith('.' + h) for h in TRUSTED_EXACT_HOSTS)

def apply_hybrid_safeguard(cleaned, raw_spam_prob):
    """Apply conservative rule-based safeguards against false positives on official domains and short casual chat."""
    if HIGH_RISK_RE.search(cleaned):
        return raw_spam_prob, None

    domains = extract_domains(cleaned)
    if domains and all(is_trusted_domain(d) for d in domains):
        if raw_spam_prob >= 0.40:
            return min(round(raw_spam_prob * 0.38, 4), 0.24), "Trusted Official Domain (.ac.th/.go.th)"
        return raw_spam_prob, None

    if len(cleaned) <= 25 and not domains and not SHORT_CASUAL_RISK_RE.search(cleaned):
        if raw_spam_prob >= 0.35:
            return min(round(raw_spam_prob * 0.35, 4), 0.18), "Short Everyday Chat Safeguard"

    return raw_spam_prob, None

def preprocess_image_for_ocr(img):
    """Preprocess image to boost Thai OCR accuracy"""
    if img.mode != 'L':
        img = img.convert('L')
    w, h = img.size
    if w < 1000:
        factor = max(2, int(1200 / w))
        img = img.resize((w * factor, h * factor), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)
    return img

def predict(text, source="Text Input"):
    cleaned = clean(text)
    p = model.predict_proba([cleaned])[0]
    raw_score = float(p[1])
    score, rule_note = apply_hybrid_safeguard(cleaned, raw_score)
    i = 1 if score >= 0.5 else 0
    adj_probs = [1.0 - score, score]
    res = {'text': text, 'label': LABELS[i], 'level': ADVICE[LABELS[i]][0],
           'advice': ADVICE[LABELS[i]][1], 'risk_score': round(score, 3),
           'probs': {l: round(float(x), 3) for l, x in zip(LABELS, adj_probs)}}
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    raw_preview = text.replace('\n', ' ')
    if len(raw_preview) > 60: raw_preview = raw_preview[:57] + '...'
    clean_preview = cleaned.replace('\n', ' ')
    if len(clean_preview) > 60: clean_preview = clean_preview[:57] + '...'
    
    status_icon = "SPAM [เสี่ยงสูง]" if LABELS[i] == "spam" else "NORMAL [เสี่ยงต่ำ]"
    
    print("-" * 68, flush=True)
    print(f"[*] [AI PREDICTION] {timestamp} | Source: {source}", flush=True)
    print(f"    Raw Input    : \"{raw_preview}\"", flush=True)
    print(f"    Cleaned Text : \"{clean_preview}\"", flush=True)
    if rule_note:
        print(f"    Raw ML Proba : Normal = {round(p[0]*100, 1)}% | Spam = {round(raw_score*100, 1)}%", flush=True)
        print(f"    Hybrid Rule  : {rule_note} -> Adjusted Spam = {round(score*100, 1)}%", flush=True)
    else:
        print(f"    Probability  : Normal = {round(adj_probs[0]*100, 1)}% | Spam = {round(adj_probs[1]*100, 1)}%", flush=True)
    print(f"    Final Result : {status_icon} (Risk Score: {round(score*100, 1)}%)", flush=True)
    print(f"    Advice       : {ADVICE[LABELS[i]][1]}", flush=True)
    print("-" * 68, flush=True)
    return res

@app.get('/')
def index(): return send_from_directory(app.static_folder, 'index.html')

@app.post('/api/predict')
def api_predict():
    text = (request.get_json(silent=True) or {}).get('text', '').strip()
    if not text: return jsonify(error='ไม่มีข้อความ'), 400
    return jsonify(predict(text, source="Text Input"))

@app.post('/api/predict-image')
def api_image():
    f = request.files.get('image')
    if not f: return jsonify(error='ไม่มีรูปภาพ'), 400
    try:
        raw_img = Image.open(io.BytesIO(f.read()))
        img = preprocess_image_for_ocr(raw_img)
        text = pytesseract.image_to_string(img, lang='tha+eng').strip()
    except pytesseract.TesseractNotFoundError:
        return jsonify(error='เครื่องยังไม่ได้ติดตั้งโปรแกรม Tesseract-OCR หรือไม่พบ path'), 501
    except Exception as e:
        return jsonify(error=f'เกิดข้อผิดพลาดในการอ่านรูปภาพ: {str(e)}'), 500
    if not text:
        return jsonify(error='ไม่พบข้อความในรูปภาพ หรือรูปภาพไม่ชัดเจน กรุณาลองใหม่อีกครั้ง'), 422
    
    return jsonify(predict(text, source=f"Image OCR ({len(text)} chars)"))

if __name__ == '__main__':
    print("=" * 60)
    print("  Thai Spam Risk Assessment Prototype Running")
    print("  Tesseract OCR Path:", pytesseract.pytesseract.tesseract_cmd)
    print("  Open your browser at: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000)
