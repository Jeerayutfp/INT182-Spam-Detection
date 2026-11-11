# INT182 Thai Spam Risk Assessment Prototype

ระบบตรวจสอบและประเมินความเสี่ยงข้อความและรูปภาพภาษาไทย (Normal vs Spam) พัฒนาด้วย **Flask + TF-IDF (`char_wb` 2-4 gram) + Logistic Regression** ร่วมกับ **Hybrid Rule-Based Safeguard** และรองรับการอ่านข้อความจากรูปภาพด้วย **Tesseract OCR (`tha+eng`)**

## โครงสร้างโปรเจกต์

- `spam_prototype/app.py` — Backend Flask API (`/api/predict`, `/api/predict-image`), Image Preprocessing, Tesseract OCR และ Hybrid Safeguard + Log การทำนายแบบ Real-time
- `spam_prototype/train.py` — สคริปต์เทรนและประเมินโมเดลด้วย 3-Fold `StratifiedGroupKFold` (สร้าง `model.joblib`)
- `spam_prototype/static/index.html` — หน้าเว็บสำหรับวางข้อความหรืออัปโหลดรูปภาพเพื่อประเมินความเสี่ยง
- `spam_prototype/data.xlsx` — ชุดข้อมูลหลัก (1,208 แถว, binary: `normal` / `spam`)
- `spam_prototype/extra_normal.csv` — ตัวอย่างข้อความทักทาย แชททั่วไป และประกาศทางการเสริม
- `spam_prototype/collected_data_clean.csv` — ข้อมูลจริงจากแชทกลุ่ม/ประกาศคณะที่ Clean และ Anonymize ตามหลัก PDPA แล้ว
- `spam_prototype/model.joblib` — โมเดลที่เทรนไว้พร้อมใช้งาน
- `colab_latest.ipynb` — โน้ตบุ๊กเปรียบเทียบโมเดลจาก Google Colab (TF-IDF + LR vs PhayaThaiBERT)
- `start_app.bat` — ไฟล์เปิดใช้งานบน Windows พร้อมแสดง Log ภาษาไทย (`UTF-8`)

## การติดตั้งและใช้งาน (Local Setup)

### 1. สร้าง Virtual Environment และติดตั้ง Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. ติดตั้ง Tesseract OCR (สำหรับฟีเจอร์อัปโหลดรูปภาพ)
ติดตั้ง Tesseract OCR พร้อมภาษาไทย (`tha.traineddata`) และภาษาอังกฤษ (`eng.traineddata`):
- ติดตั้งไว้ที่ `C:\Program Files\Tesseract-OCR\tesseract.exe` หรือ
- วางไฟล์แบบ Portable ไว้ในโฟลเดอร์ `./tesseract/tesseract.exe` (พร้อม `tessdata/tha.traineddata`)

### 3. (ไม่บังคับ) เทรนโมเดลใหม่จากข้อมูลล่าสุด
```bash
cd spam_prototype
python train.py
```

### 4. รันเว็บแอปพลิเคชัน
บน Windows สามารถรัน `start_app.bat` หรือรันผ่าน Command Line:
```bash
cd spam_prototype
python app.py
```
จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:5000`
