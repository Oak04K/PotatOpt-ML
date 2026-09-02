# 🥔 PotatOpt: Industrial AI Engine for Condition-Based & Predictive Maintenance

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/potatopt.svg)](https://pypi.org/project/potatopt/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-LowSpecML%20%2F%20Zero--Leakage-orange.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-CPU--Friendly%20(No%20GPU%20Required)-yellow.svg)]()
[![Compliance](https://img.shields.io/badge/Security-ISO%209001%20SHA--256%20Anti--Tamper-red.svg)]()

> **"PotatOps"** — Production-Grade, Zero-Code AutoML & Statistical Engine สำหรับ **วิศวกรรมอุตสาหการ (Industrial Engineering)** ที่นำข้อมูลเซนเซอร์จากเครื่องจักรต่างๆ ในสายการผลิตมาวิเคราะห์และวางแผน **Preventive / Predictive Maintenance (PdM)** — ตรวจจับความผิดปกติและแนวโน้มการเสื่อมสภาพของเครื่องจักรก่อนเกิดความเสียหาย โดยประมวลผลได้บนเครื่องคอมพิวเตอร์สเปกจำกัด (Potato Hardware / Edge Devices) ไม่ต้องพึ่งพา GPU

---

## ในโปรเจกต์นี้มีอะไรบ้าง (ภาพรวม 4 ชั้น)

โครงสร้างเป็น **4 ชั้นซ้อนกัน** แต่ละชั้นใช้ชั้นล่างและ**ไม่คำนวณซ้ำ**

| ชั้น | คือ | ทำอะไร |
|---|---|---|
| **`potatopt/`** | ไลบรารี | 30 ฟังก์ชัน + 1 engine: AutoML, SPC + กฎควบคุม, Drift รายเครื่อง, Cp/Cpk, Gauge R&R, MTBF/MTTR/OEE/Pareto และการแปลงทุกอย่างเป็นเงิน — ทุกฟังก์ชันคืน dict ที่เป็น JSON และ **ไม่ raise** |
| **`chart_engine.py`** | กราฟ | วาดจาก dict ที่ไลบรารีคืนมา **ไม่คำนวณเอง** กราฟจึงไม่มีทางขัดกับตัวเลขที่พิมพ์อยู่ข้างๆ |
| **`potatopt/mcp_server.py`** | ตัวต่อกับ AI | 7 เครื่องมือผ่าน MCP บน stdio — AI ขับได้โดยไม่ต้องเขียน Python และข้อมูลโรงงานไม่ออกจากเครื่อง |

```bash
python examples/tour.py       # เดินดูความสามารถทุกกลุ่มบนข้อมูลจำลอง ~1 นาที
python examples/quickstart.py # งาน PdM เต็มรูปแบบบนข้อมูลจริง AI4I 2020
```




**แก่นที่ทำให้ต่างจาก AutoML ทั่วไป 2 ข้อ:** ทุกคำตอบแปลงเป็น**เงิน** ไม่ใช่ค่า accuracy / และ**ห้ามเงียบ** — ถ้าตรวจไม่ได้ต้องบอกว่าตรวจไม่ได้ ห้ามตอบว่า "ปกติดี"

---

## สารบัญ (Table of Contents)
- [จุดเด่นและปรัชญาการออกแบบ (Key Highlights)](#จุดเด่นและปรัชญาการออกแบบ-key-highlights)
- [การติดตั้ง (Installation)](#การติดตั้ง-installation)
- [เริ่มต้นใช้งานด่วน (Quick Start)](#เริ่มต้นใช้งานด่วน-quick-start)
- [แผนผังการไหลของโค้ด (Execution Flow Map)](#แผนผังการไหลของโค้ด-execution-flow-map)
- [คู่มือการใช้งาน API (API Reference)](#คู่มือการใช้งาน-api-api-reference)
  - [1. Stateless Industrial Utilities](#1-stateless-industrial-utilities)
  - [2. PotatOptEngine — Constructor & Public Methods](#2-potatoptengine--constructor--public-methods)
- [กลไกภายในและระบบดักข้อมูลผิดปกติ (Internal Mechanics & Data Guardrails)](#กลไกภายในและระบบดักข้อมูลผิดปกติ-internal-mechanics--data-guardrails)
  - [3.1 ค่าคงที่ระดับโมดูล (Module Constants)](#31-ค่าคงที่ระดับโมดูล-module-constants)
  - [3.2 _validate_fit_inputs(X_train, y_train) — ด่านตรวจก่อนเทรน](#32-_validate_fit_inputsx_train-y_train--ด่านตรวจก่อนเทรน)
  - [3.3 _preprocess_fit_transform(X, y) — Pipeline 11 ขั้นตอน](#33-_preprocess_fit_transformx-y--pipeline-11-ขั้นตอน)
  - [3.4 _preprocess_transform(X) — การบังคับใช้ schema ตอน Inference](#34-_preprocess_transformx--การบังคับใช้-schema-ตอน-inference)
  - [3.5 _check_input_anomalies(X_proc_raw) — ตัวเฝ้าระวังตอน Inference](#35-_check_input_anomaliesx_proc_raw--ตัวเฝ้าระวังตอน-inference)
  - [3.6 ตารางสรุปพฤติกรรมเมื่อเจอข้อมูลผิดปกติ (Guardrail Summary)](#36-ตารางสรุปพฤติกรรมเมื่อเจอข้อมูลผิดปกติ-guardrail-summary)
- [ระบบคุณภาพข้อมูลและการเฝ้าระวังหลัง Deploy (Data Quality & Monitoring)](#ระบบคุณภาพข้อมูลและการเฝ้าระวังหลัง-deploy-data-quality--monitoring)
  - [4.1 Data Quality Score](#41-data-quality-score)
  - [4.2 Silent Nulls](#42-silent-nulls)
  - [4.3 PSI Drift Detection](#43-psi-drift-detection)
  - [4.4 Audit Trail & Logging](#44-audit-trail--logging)
- [การผสานกรอบคิดวิศวกรรมอุตสาหการ (IE Integration)](#การผสานกรอบคิดวิศวกรรมอุตสาหการ-ie-integration)
- [ระบบความปลอดภัย ISO 9001 (Model Integrity & Security)](#ระบบความปลอดภัย-iso-9001-model-integrity--security)
- [โครงสร้างโปรเจกต์ (Project Structure)](#โครงสร้างโปรเจกต์-project-structure)
- [License](#license)
- [ต้นทุน Token (Token Cost)](#ต้นทุน-token-token-cost)
- [Type Hints](#type-hints)
- [Control Chart สำหรับงานซ่อมบำรุง (EWMA / CUSUM)](#control-chart-สำหรับงานซ่อมบำรุง-ewma--cusum)
- [Per-Asset Drift — ทำไมการรวมหลายเครื่องจักรถึงตอบผิดทั้งสองทาง](#per-asset-drift--ทำไมการรวมหลายเครื่องจักรถึงตอบผิดทั้งสองทาง)
- [กรอบต้นทุนงานซ่อมบำรุง (Maintenance Cost Framing)](#กรอบต้นทุนงานซ่อมบำรุง-maintenance-cost-framing)
- [ติดตั้งเป็นแอป และการเปิด HTTPS](#ติดตั้งเป็นแอป-และการเปิด-https)

---

## จุดเด่นและปรัชญาการออกแบบ (Key Highlights)

1. **LowSpecML (Potato Hardware First):**
   - บีบอัดหน่วยความจำ RAM แบบ Lossless ด้วยระบบ Memory Downcaster (int8/int16/int32/float32)
   - ใช้ Ordinal Encoding แทน One-Hot Encoding ป้องกันปัญหา Sparse Matrix กิน RAM มหาศาล
   - รันโมเดล Tree-Based บน CPU ได้อย่างรวดเร็วโดยไม่ต้องใช้ GPU — ค่าเริ่มต้นคือ LightGBM, XGBoost, Random Forest (`["lgbm", "xgboost", "rf"]`) และเพิ่ม estimator อื่นที่ FLAML รองรับ เช่น CatBoost ได้ผ่านพารามิเตอร์ `estimators`
   - **ตัวเลขที่วัดได้จริง (ไม่ใช่คำโฆษณา):** `tests/test_potatopt.py` มีชุด Memory Benchmark ที่จะ Fail ถ้าประสิทธิภาพถดถอย — วัดบนข้อมูลอ้างอิง 20,000 แถว (5 คอลัมน์เซนเซอร์ + 2 คอลัมน์เชิงกลุ่ม) ได้ผล: การ Downcast คอลัมน์ตัวเลข ประหยัด **50.0%** (float64 → float32 พอดีเป๊ะ) และ Preprocessing เต็มไปป์ไลน์ ประหยัด **84.5%** (2.71MB → 0.42MB เพราะคอลัมน์ข้อความกลายเป็น `category` แทนที่จะถูก One-Hot ขยาย) พร้อมเพดาน RSS ของ `fit()` ที่ 400MB
2. **Sentinel-Safe Auto-Guardrails:**
   - ตรวจจับชุดข้อมูลที่มีของเสียน้อยมาก (Imbalance Scarcity < 5 ตัวอย่าง) แล้วสลับเป็น **Unsupervised Anomaly Detection (Isolation Forest)** อัตโนมัติ ป้องกันปัญหา Cross-Validation Crash
   - ป้องกัน Alphabetical Label Traps และคืนค่า Sentinel Label อย่างปลอดภัย
3. **Zero Data Leakage:**
   - เรียนรู้ค่าสถิติ (Imputation, MinMax/Standard bounds, Ordinal mappings) เฉพาะในขั้นตอน `fit()` และนำมาประยุกต์ใช้แบบ Idempotent ในขั้นตอน `predict()`
   - ล็อกประเภทข้อมูล (`column_dtypes`) และลำดับคอลัมน์เพื่อความแม่นยำ 100% ในระดับ Production
4. **Industrial Financial Layer (Cost-Based Threshold Optimization):**
   - ฟังก์ชันปรับจูน Threshold (`optimize_threshold`) ตามสมการต้นทุนจริง — ปรับใช้ได้ทั้งบริบทตรวจของเสีย (Scrap/False-Alarm/Inspection) และบริบทซ่อมบำรุง (ค่าเสียหายจากเครื่องพังกะทันหัน vs ค่าเปลี่ยนอะไหล่ล่วงหน้า vs ค่า Inspection):
     $$\text{Total Cost} = (\text{FN} \times \text{Cost}_{\text{Missed Failure}}) + (\text{FP} \times \text{Cost}_{\text{False Alarm}}) + (\text{TP} \times \text{Cost}_{\text{Inspection}})$$
   - แปลงผลการทำนายเป็นมูลค่าความคุ้มค่าและเงินที่ประหยัดได้ (Cost Savings / ROI)
5. **ISO 9001 SHA-256 Anti-Tamper Security:**
   - ระบบเซฟโมเดลพร้อมคู่มือ Metadata และลายเซ็นดิจิทัล SHA-256 แจ้งเตือนความปลอดภัยทันทีเมื่อไฟล์โมเดลถูกแก้ไข ดัดแปลง หรือเสียหาย
6. **Data Quality Score Gate:**
   - ตรวจประเมินสุขภาพข้อมูล 5 มิติ (Completeness, Consistency, Validity, Uniqueness, Timeliness) พร้อมจัดเกรด (`production_ready`, `usable_with_caveats`, `remediation_required`) และ Missing-Value Playbook ก่อนเทรน
7. **Condition Monitoring ด้วย PSI Drift Detection (Survives Deployment):**
   - ตรวจวัด Covariate Shift ระดับการกระจายตัวด้วย Population Stability Index (PSI) และการขยายตัวของความแปรปรวน (Variance Inflation / Std Ratio) — จับสัญญาณ **การเสื่อมสภาพของเครื่องจักร (Tool Wear / Degradation)** ได้แม้ค่าเฉลี่ยเซนเซอร์ยังไม่ขยับ ทำงานบนโมเดลที่ Deploy ไปแล้วโดยไม่ต้องพึ่งพาชุดข้อมูล Train เดิม ผ่าน Statistical Profile ที่ Freeze ไว้
8. **Full ISO 9001 Provenance & File-Based Audit Trail:**
   - ล็อกความสามารถในการสอบย้อนกลับ (Traceability) เต็มรูปแบบด้วย UTC Timestamp, ลายเซ็น SHA-256 Data Hash ของชุดข้อมูล Train, รายการเวอร์ชันไลบรารีทุกตัว และระบบบันทึก Audit Log ลงไฟล์ (`enable_audit_log`) เพื่อพิสูจน์ย้อนหลังว่า Batch ใดกระตุ้น Warning ในสายการผลิต

---

## การติดตั้ง (Installation)

### 1. โคลน Repository
```bash
git clone https://github.com/Oak04K/PotatOpt-ML.git
cd PotatOpt-ML
```

### 2. ติดตั้ง Dependencies ผ่าน `requirements.txt`
```bash
pip install -r requirements.txt
```
> **หมายเหตุ:** ใน `requirements.txt` จะมีบล็อก Development-only dependencies (`pytest`, `ruff`) รวมอยู่ด้วย สำหรับการติดตั้งบน Production Edge Device ที่มีเนื้อที่จำกัด สามารถเลือกข้ามแพ็กเกจกลุ่มนี้ได้

---

## เริ่มต้นใช้งานด่วน (Quick Start)

เพียงไม่กี่ขั้นตอนในการทำ End-to-End Machine Learning สำหรับงานวิศวกรรมอุตสาหการ พร้อมระบบตรวจสอบคุณภาพข้อมูลและเฝ้าระวัง Data Drift:

**เริ่มแบบสั้นที่สุด — บรรทัดเดียวจบ**

```python
import potatopt as po

result = po.auto_analyze("sensors.csv", target="failure",
                         cost_scrap=500, cost_fa=150, cost_insp=20)

print(result["metrics"]["f1"], result["cost"]["cost_savings"])
```

`auto_analyze()` ทำครบทั้งกระบวนการให้ในครั้งเดียว: ตรวจคุณภาพข้อมูล -> แบ่ง Train/Validation/Test -> ค้นหาโมเดล -> จูน Threshold บน Validation -> วัดผลบน Test -> แปลงเป็นเงิน -> จัดอันดับฟีเจอร์ แล้วคืนค่าเป็น dict ก้อนเดียวที่แปลงเป็น JSON ได้ทันที และ **ไม่เคย Raise** — ถ้าพังจะคืน `ok: False` พร้อมข้อความที่อ่านรู้เรื่อง

**สิ่งที่มันซ่อน กับสิ่งที่มันไม่ซ่อน:** ซ่อนเฉพาะขั้นตอนที่มีคำตอบถูกอยู่แล้ว (การแบ่งชุดข้อมูล, Encoding, Scaling, Downcasting) แต่ **ไม่ซ่อนค่าต้นทุน** (`cost_scrap`, `cost_fa`, `cost_insp`) เพราะเป็นตัวเลขที่วิศวกรหน้างานเท่านั้นที่รู้ ถ้าซ่อนเมื่อไหร่ผลลัพธ์ก็เลิกเป็นงานวิศวกรรมทันที

ถ้าต้องการควบคุมเองทีละขั้น API เดิมยังอยู่ครบทุกตัว ใช้ได้ตามปกติ:

```python
import pandas as pd
import potatopt as po
from potatopt import PotatOptEngine, split_data_three_way, inspect_data, audit_data_quality

# เปิดระบบบันทึก Audit Log สำหรับสอบย้อนกลับตามมาตรฐาน ISO 9001
po.enable_audit_log("potatopt_audit.log")

# 1. โหลดข้อมูลและตรวจสอบสุขภาพข้อมูล (Data Health Check)
df = pd.read_csv("sensor_manufacturing_data.csv")
report = inspect_data(df, target_col="Defect_Status")
print("Data Profile:", report)

# 2. ตรวจสอบ Data Quality Score (DQS) และเกรดความพร้อมก่อนนำไปเทรน
audit = audit_data_quality(df, target_col="Defect_Status")
print(f"Data Quality Score: {audit['dqs']}/100, Grade: {audit['grade']}")

# 3. แบ่งชุดข้อมูล Train / Validation / Test (Stratified 60/20/20)
#    Validation Set มีไว้จูน Threshold โดยเฉพาะ จะได้ไม่ต้องแอบดูข้อมูล Test ก่อนรายงานผล
X_train, X_val, X_test, y_train, y_val, y_test = split_data_three_way(
    df, target_col="Defect_Status", val_size=0.2, test_size=0.2
)

# 4. ประกาศและเทรน Engine (AutoML + LowSpecML)
engine = PotatOptEngine(
    task="auto",                    # เลือกระหว่าง 'auto', 'classification', 'regression'
    time_budget=30,                 # ล็อกเวลาค้นหาโมเดล 30 วินาที
    scale_method="standard",        # 'standard' หรือ 'minmax'
    cost_sensitive_weighting=True,  # จัดการ Imbalanced Data แบบ Zero-RAM
    n_jobs=-1                       # -1 = ใช้ทุกคอร์, -2 = เว้นไว้ 1 คอร์, หรือระบุจำนวน Worker ตรงๆ
)
engine.fit(X_train, y_train)

# 5. ตรวจสอบรายงานผลการเทรนและคะแนน Cross-Validation เพื่อป้องกัน Overfitting
training_report = engine.get_training_report()
print(f"Metric Optimized: {training_report['metric_optimized']}, Validation Score: {training_report['validation_score']}")

# 6. ปรับ Threshold ตามต้นทุนคุณภาพจริง (Cost of Quality) — จูนบน Validation Set เท่านั้น
engine.optimize_threshold(
    X_val, y_val, 
    cost_scrap=500.0,   # ค่าปรับของเสียหลุดไปหาลูกค้า (False Negative)
    cost_fa=150.0,      # ค่าเสียเวลาตรวจสอบเมื่อแจ้งเตือนผิดพลาด (False Positive)
    cost_insp=20.0      # ค่าตรวจเช็กชิ้นงานปกติ (True Positive)
)

# 7. ประเมินผลและคำนวณมูลค่าความคุ้มค่า
metrics = engine.evaluate(X_test, y_test)
coq = engine.calculate_cost_of_quality(X_test, y_test, cost_scrap=500, cost_fa=150, cost_insp=20)

print(f"Model Champion: {metrics['best_model_name']}")
print(f"F1-Score: {metrics.get('f1', 'N/A')}")
print(f"Cost Savings: {coq['cost_savings']} ({coq['savings_percentage']})")

# 8. บันทึกโมเดลพร้อมตรวจสอบลายเซ็น ISO 9001 SHA-256
engine.save("production_model.pkl")

# 9. เฝ้าระวัง Data Drift บนข้อมูล Batch ใหม่หน้างาน (Monitoring Path)
new_batch = pd.read_csv("new_sensor_batch.csv")
drift_result = engine.detect_drift(new_batch)
print(f"Drift Detected: {drift_result['drift_detected']}")
print(f"Recommendation: {drift_result['recommendation']}")
```

---

## แผนผังการไหลของโค้ด (Execution Flow Map)

การทำงานของ PotatOpt มีจุดเริ่มต้นและเส้นทางการประมวลผลหลัก 3 รูปแบบ คือ **เส้นทางการเทรนโมเดล (Training Path)** ผ่านเมธอด `fit()`, **เส้นทางการใช้งานจริง (Inference Path)** ผ่านเมธอด `predict()` และ **เส้นทางการเฝ้าระวังหลัง Deploy (Monitoring Path)** ผ่านเมธอด `detect_drift()` รวมถึงเมธอดอื่น ๆ ที่เรียกใช้ต่อเป็นทอด ๆ โดยทุกเส้นทางได้รับการออกแบบให้มีการส่งต่อข้อมูลที่เป็นระเบียบ ปราศจาก Data Leakage และมีระบบความปลอดภัยคอยควบคุมอย่างเข้มงวด

```
เส้นทางที่ 1: การเทรน (Training Path)

engine.fit(X_train, y_train) [Public Entry Point]
  │
  ├── 1. Sparse Matrix Unpack: hasattr(X, "toarray") -> X.toarray()
  │
  ├── 2. self._validate_fit_inputs(X_train, y_train) [GUARD: raises ValueError]
  │
  ├── 3. Length Check: len(X_train) != len(y_train) [raises ValueError]
  │
  ├── 4. self._reset_state() (ล้างค่าที่เคยเรียนรู้ทั้งหมด)
  │
  ├── 5. Task Auto-Detection (ประเมินประเภทงานหากตั้ง task="auto")
  │
  ├── 6. Drop Rows with NaN Target
  │
  ├── 7. Classification: LabelEncoder.fit_transform on y + กำหนด pos_label_idx
  │
  ├── 8. self._preprocess_fit_transform(X_train, y=y_proc)
  │        └── เรียกใช้ self._reduce_mem_usage(df) ภายใน Stage 7
  │
  ├── 9. Empty-Feature Guard: if X_proc.shape[1] == 0 [GUARD: raises ValueError]
  │
  ├── 10. Imbalance / Scarcity Guardrails
  │         │
  │         ├── [ของเสีย < 5 ชิ้น หรือ Scarcity] ──► สลับไป IsolationForest Path
  │         │                                         ├─ เทรน Unsupervised Anomaly Model
  │         │                                         ├─ กำหนด self.is_anomaly_model = True
  │         │                                         └─ Return Early สิ้นสุดการเทรน
  │         │
  │         └── [ข้อมูลปกติ] ──► ดำเนินการต่อ
  │
  ├── 11. Optional Cost-Sensitive Sample Weighting (คำนวณ sample_weight)
  │
  └── 12. FLAML AutoML.fit() -> กำหนด self.model และ self.is_fitted = True
```

```
เส้นทางที่ 2: การใช้งานจริง (Inference Path)

[User Calls: Inference & Evaluation]
  │
  ├── engine.evaluate(X, y) ──────────────┐
  │                                       │ (เรียกต่อ)
  ├── engine.calculate_cost_of_quality() ─┼───────────────┐
  │                                       │               │
  │                                       ▼               │
  ├── engine.predict(X) ──────────────────────────────────┼───┐
  │                                                       │   │
  ├── engine.predict_proba(X) ────────────────────────────┼───┤
  │                                                       │   │
  ├── engine.optimize_threshold(X, y) ────────────────────┼───┤
  │                                                       │   │
  └── engine.get_shap_values(X) ──────────────────────────┼───┤
                                                          │   │
                                                          ▼   ▼
                                           self._preprocess_transform(X)
                                             │
                                             └──► self._check_input_anomalies(X_proc)
                                                    [OBSERVER: ตรวจจับ Schema Mismatch & Out-of-bounds
                                                     ไม่ raise ขัดจังหวะ แต่จะเตือนผ่าน WARNING]
```

```
เส้นทางที่ 3: การเฝ้าระวังหลัง Deploy (Monitoring Path)

[User Calls: Post-Deployment Monitoring]
  │
  ├── engine.detect_drift(X_batch)
  │     │
  │     ├── 1. self._preprocess_transform(X_batch, apply_bounds_clip=False)
  │     │        (บายพาส Bounds Clipping เพื่อวัดการกระจายตัวที่แท้จริง ไม่ให้ถูกกลบ)
  │     │
  │     ├── 2. _psi_core() เปรียบเทียบกับ self.train_profile (bin edges & frequencies)
  │     │
  │     ├── 3. การจัดระดับความรุนแรง (Severity Classification)
  │     │        ├─ PSI > 0.25 ──► "major" (drift_detected = True)
  │     │        ├─ 0.10 < PSI <= 0.25 ──► "moderate"
  │     │        └─ PSI <= 0.10 ──► "stable"
  │     │
  │     └── 4. ส่งผล Recommendation และบันทึกผ่าน logger (WARNING หากมี Major Drift, INFO เมื่อปกติ)
  │
  └── engine.get_inference_health()
        └── อ่านค่า Counters: transform_calls, rows_transformed, warning_events, warning_rate
            ที่ถูกสะสมและนับเพิ่มโดย _preprocess_transform ในทุก ๆ รอบ Inference
```

### ตารางสรุปการเรียกใช้ (Call Dependency Table)

| เมธอด (Method) | ประเภท | เรียกใช้ภายใน (Calls) | ต้อง fit() ก่อนหรือไม่ |
|---|---|---|---|
| `fit()` | Public | `_validate_fit_inputs`, `_reset_state`, `_preprocess_fit_transform` | ไม่ (คือจุดเริ่มต้น) |
| `predict()` | Public | `_preprocess_transform` | ต้อง |
| `predict_proba()` | Public | `_preprocess_transform` | ต้อง |
| `optimize_threshold()` | Public | `_preprocess_transform` | ต้อง |
| `check_calibration()` | Public *(ใหม่ v1.4.0)* | `predict_proba()`, `check_calibration()` (module-level) | ต้อง |
| `evaluate()` | Public | `predict()` | ต้อง |
| `calculate_cost_of_quality()` | Public | `predict()` | ต้อง |
| `get_feature_importance()` | Public | ไม่เรียกเมธอดอื่น | ต้อง |
| `get_shap_values()` | Public | `_preprocess_transform` | ต้อง |
| `save()` | Public | ไม่เรียกเมธอดอื่น | ต้อง |
| `load()` | Classmethod | ไม่เรียกเมธอดอื่น | ไม่ |
| `audit_data_quality()` | Public (Utility) | `detect_silent_nulls`, `detect_outliers` | ไม่ |
| `detect_silent_nulls()` | Public (Utility) | ไม่เรียกเมธอดอื่น | ไม่ |
| `detect_outliers()` | Public (Utility) | ไม่เรียกเมธอดอื่น | ไม่ |
| `calculate_psi()` | Public (Utility) | `_build_psi_bins`, `_psi_core` | ไม่ |
| `wilson_confidence_interval()` | Public (Utility) | ไม่เรียกเมธอดอื่น | ไม่ |
| `enable_audit_log()` | Public (Utility) | ไม่เรียกเมธอดอื่น | ไม่ |
| `get_library_versions()` | Public (Utility) | ไม่เรียกเมธอดอื่น | ไม่ |
| `get_training_report()` | Public | ไม่เรียกเมธอดอื่น | ต้อง |
| `detect_drift()` | Public | `_preprocess_transform`, `_psi_core` | ต้อง |
| `get_inference_health()` | Public | ไม่เรียกเมธอดอื่น | ต้อง |
| `_validate_fit_inputs()` | Private (Guard) | ไม่เรียกเมธอดอื่น | ไม่ |
| `_reset_state()` | Private | ไม่เรียกเมธอดอื่น | ไม่ |
| `_preprocess_fit_transform()` | Private | `_reduce_mem_usage` | ไม่ |
| `_preprocess_transform()` | Private | `_check_input_anomalies` | ต้อง |
| `_check_input_anomalies()` | Private (Observer) | ไม่เรียกเมธอดอื่น | ต้อง |
| `_reduce_mem_usage()` | Private | ไม่เรียกเมธอดอื่น | ไม่ |
| `_convert_silent_nulls()` | Private | ไม่เรียกเมธอดอื่น | ไม่ |
| `_hash_training_data()` | Private | ไม่เรียกเมธอดอื่น | ไม่ |
| `_compute_auc_metrics()` | Private | `predict_proba()` | ต้อง |
| `_binary_count_breakdown()` | Private | ไม่เรียกเมธอดอื่น | ไม่ |

---

## คู่มือการใช้งาน API (API Reference)

### 1. Stateless Industrial Utilities

#### `inspect_data(df, target_col) -> dict`
สแกนข้อมูล ตรวจนับ Missing Values แนะนำประเภทงาน (`classification` หรือ `regression`), ตัวชี้วัด (Metric) ที่เหมาะสม และประเมินสรุปคะแนน Data Quality Score อัตโนมัติ

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ที่ประกอบด้วยคีย์: `total_rows`, `total_columns`, `missing_values`, `recommended_task`, `recommended_metric`, `message`, `data_quality` (เพิ่มในเวอร์ชัน 1.1.0 บนทั้งสอง Success Paths: `{"dqs": float, "grade": str, "top_issues": list of <= 3 strings}`) หากเกิดข้อผิดพลาดจะคืนค่า `dict` ที่มีคีย์ `"error"` (ไม่มีคีย์ `data_quality` บน 3 Error Paths)
- คืนค่า error ทันทีหาก `df` เป็น `None` หรือว่างเปล่า (`empty`), หรือ `target_col` ไม่อยู่ในคอลัมน์ของข้อมูล, หรือคอลัมน์ target มีค่าเป็น NaN ทั้งหมด
- หาก target มีค่าไม่ซ้ำ (unique) $\le 1$ ค่า จะกำหนด `recommended_task = "invalid"` และ `recommended_metric = "none"`
- จัดการกรณีชื่อคอลัมน์ target ซ้ำกันใน DataFrame โดยจะเลือกใช้คอลัมน์แรก (`.iloc[:, 0]`)
- ตรวจจับตัวเลขในรูปแบบข้อความ (numeric-strings): หากทุกค่าใน target สามารถแปลงด้วย `pd.to_numeric` ได้ จะถือว่าเป็นข้อมูลตัวเลข
- การตัดสินใจเลือก Task:
  - หาก unique มี 2 ค่าพอดี -> `recommended_task = "classification"`, `recommended_metric = "f1"`
  - หากไม่ใช่ตัวเลข หรือ unique $\le 10$ ค่า -> `recommended_task = "classification"`, `recommended_metric = "macro_f1"`
  - กรณีอื่น ๆ นอกเหนือจากนี้ -> `recommended_task = "regression"`, `recommended_metric = "r2"`
- ติดสถานะ Imbalance (ข้อมูลไม่สมดุล) ในงาน Binary Classification เมื่อสัดส่วนคลาสส่วนน้อย (minority ratio) $< 0.20$

#### `split_data(df, target_col, task="classification", test_size=0.2, random_state=42) -> (X_train, X_test, y_train, y_test)`
ตัดแบ่งข้อมูล Train / Test ในอัตราส่วน 80/20 พร้อมระบบรักษาลำดับเวลาหรือการกระจายตัวของคลาส

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ตัดแบ่งข้อมูลตามค่า `test_size` (ค่าเริ่มต้น 80/20) โดยรับได้ทั้ง `float` ในช่วง 0 < test_size < 1 (สัดส่วน) และ `int` ตั้งแต่ 1 ขึ้นไป (จำนวนแถวตรงๆ) — Raise `ValueError` หากเป็นชนิดอื่นหรืออยู่นอกช่วง (`bool` ถูกปฏิเสธเสมอ)
- Raise `ValueError` ทันทีหาก `df` ว่างเปล่า, ไม่มี `target_col`, หรือข้อมูลว่างเปล่าหลังตัดแถวที่ target เป็น NaN ทิ้ง
- ตัดแถวที่ target เป็น NaN ทิ้งก่อนทำการตัดแบ่งข้อมูลเสมอ
- กรณี `task="forecasting"` -> จะตั้ง `shuffle=False` และไม่ใช้ `random_state` เลย เพื่อรักษาลำดับตามเวลา (Time Order)
- กรณีอื่น ๆ -> ใช้ `random_state` ตามที่ส่งเข้ามา (ค่าเริ่มต้น `42` ผ่านคงที่ `DEFAULT_RANDOM_STATE`) *(เดิมเป็น `42` ตายตัวในโค้ด ก่อน v1.4.0)* — Raise `ValueError` หากไม่ใช่จำนวนเต็ม (`bool` ถูกปฏิเสธเสมอ เพราะเป็น subclass ของ `int`)
- กรณี `task="classification"` -> ทำการแบ่งแบบแบ่งชั้น (Stratified Split) โดยระบบจะปิด Stratification อัตโนมัติหาก `y.nunique() < 2` หรือคลาสที่เล็กที่สุดมีจำนวนตัวอย่างน้อยกว่า 2 ตัวอย่าง นอกจากนี้ยังมีบล็อก `try/except` สำรองเพื่อ fallback ไปเป็นการแบ่งแบบไม่ stratified หากเกิดข้อผิดพลาด

#### `split_data_three_way(df, target_col, task="classification", val_size=0.2, test_size=0.2, random_state=42) -> (X_train, X_val, X_test, y_train, y_val, y_test)`
ตัดแบ่งข้อมูลเป็น 3 ส่วน Train / Validation / Test เพื่อให้จูน Threshold บน Validation แล้วรายงานผลบน Test ที่โมเดลยังไม่เคยเห็น

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- เรียก `split_data` สองรอบด้วย `random_state` เดียวกันทั้งสองรอบ: รอบแรกตัด Test ออกก่อน รอบที่สองตัด Validation ออกจากส่วนที่เหลือ จึงได้พฤติกรรม Stratified / `shuffle=False` เหมือน `split_data` ทุกประการ
- `val_size` คิดเป็นสัดส่วนของข้อมูล **ทั้งชุด** โดยรอบที่สองจะแปลงเป็น "จำนวนแถว" จริงก่อนตัด เพื่อไม่ให้ทศนิยมฐานสองปัดแถวเกิน (เช่น 0.1/0.6 บนข้อมูล 200 แถว เคยได้ 21 แถวแทนที่จะเป็น 20)
- Raise `ValueError` หาก `val_size` ไม่อยู่ในช่วง 0 < val_size < 1 หรือ `val_size + test_size >= 1.0`
- กรณี `task="forecasting"` -> ลำดับเวลาถูกรักษาไว้เป็น Train -> Validation -> Test (Test คือช่วงเวลาล่าสุด)

#### `run_seed_sweep(data, target, seeds=(0,1,2,3,4), **kwargs) -> dict` *(ใหม่ v1.4.0)*
รัน `auto_analyze` ซ้ำทีละ Seed แล้วสรุปค่า Mean / Std / Min / Max / **Spread** ของทุก Metric แทนที่จะรายงานผลจาก Seed เดียว

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ทำไมต้องมี: ผลลัพธ์จากการรันครั้งเดียวคือ 1 ตัวอย่างสุ่ม เพราะทั้ง `random_state` ของการแบ่งข้อมูลและการค้นหาโมเดลของ FLAML เป็น Stochastic — บนข้อมูลขนาดเล็กที่ Imbalance สูงอย่างที่ไลบรารีนี้เจาะกลุ่ม ค่าที่ได้กระจายกว้างพอที่จะทำให้ตัวเลขที่ตีพิมพ์จาก Seed เดียวไม่สามารถทำซ้ำได้
- คีย์ผลลัพธ์: `runs` (ราย Seed พร้อมสถานะสำเร็จ/ล้มเหลว), `summary` (ต่อ Metric: `mean`, `std`, `min`, `max`, `spread`, `n`), `stability_note` — ประโยคที่บอกตรง ๆ ว่าผลต่างที่เล็กกว่า `spread` ไม่ถือเป็นข้อค้นพบ
- ตัดค่าจาก `metrics`, `cost` และ `calibration.expected_calibration_error` เฉพาะที่เป็นตัวเลขและปรากฏครบทุก Seed เท่านั้นมาสรุปใน `summary` — Metric ที่ขาดในบาง Seed จะไม่ถูกนำมาคำนวณ Spread เพราะเทียบกันไม่ได้
- `seeds` ต้องไม่ว่างและห้ามมีค่าซ้ำ — Raise error แบบ `{"error": ...}` (ไม่ raise จริง) หากส่ง `random_state` ปนเข้ามาใน `**kwargs`, `seeds` ว่าง, มีค่าซ้ำ, หรือไม่ใช่จำนวนเต็ม
- ค่าใช้จ่าย: แต่ละ Seed คือการรัน `auto_analyze` เต็มรูปแบบหนึ่งรอบ เวลารวมจึงประมาณ `len(seeds) x time_budget`

#### `calculate_spc_limits(df, sensor_column, n_sigmas=3) -> dict`
คำนวณเส้นควบคุมคุณภาพทางสถิติ (Statistical Process Control Limits) สำหรับการติดตามค่าความผันแปรของกระบวนการ

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ประกอบด้วยคีย์: `sensor`, `mean`, `ucl`, `lcl` หากเกิดข้อผิดพลาดจะคืนค่า `dict` ที่มีคีย์ `"error"`
- สูตรการคำนวณ: $\text{UCL} = \text{mean} + (n\_\text{sigmas} \times \text{std})$ และ $\text{LCL} = \text{mean} - (n\_\text{sigmas} \times \text{std})$
- แปลงข้อมูลในคอลัมน์เป็นตัวเลขด้วย `pd.to_numeric(errors='coerce')` แล้วตัดค่า NaN ทิ้งก่อนคำนวณ
- หากค่าส่วนเบี่ยงเบนมาตรฐาน (std) เป็น NaN (เช่น มีข้อมูลเพียงแถวเดียว) จะถูกแทนค่าเป็น `0.0`
- หากค่า `n_sigmas` เป็นค่าว่าง (falsy) หรือ $\le 0$ ระบบจะ fallback กลับไปใช้ค่าเริ่มต้นคือ `3.0`

#### `check_data_drift(train_df, batch_df, threshold_pct=0.20, min_rows=0, psi_bins=10, include_categorical=False) -> dict`
ตรวจวัดการเคลื่อนตัวของการกระจายตัวของข้อมูล (Data Drift) ระหว่างชุดข้อมูลที่ใช้เทรนกับข้อมูล Batch ใหม่หน้างาน ทั้งในมิติการเลื่อนของค่าเฉลี่ย (Mean Shift) และการเปลี่ยนแปลงของการกระจายตัว (PSI / Variance Shift)

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ประกอบด้วย: `drift_detected` (bool), `drifted_features` (dict), `max_psi` (ค่า PSI สูงสุดจากทุกคอลัมน์ หรือ `None`) โดยจะเพิ่มคีย์ `"error"` หาก input ฝั่งใดฝั่งหนึ่งเป็น `None` หรือว่างเปล่า
- สแกนเฉพาะคอลัมน์ตัวเลข (numeric columns) ของ `train_df` ที่มีชื่อตรงกันใน `batch_df` เท่านั้น
- กรองค่า `inf`, `-inf` และ `NaN` ออกจากข้อมูลทั้งสองฝั่งก่อนทำการเปรียบเทียบ
- คำนวณ Normalized Shift = $\frac{|\text{batch\_mean} - \text{train\_mean}|}{\text{scale}}$
- ลำดับความสำคัญในการเลือกตัวหาร (`scale`): ใช้ค่า train std หากไม่ใช่ NaN และ $> 10^{-6}$; หากไม่เข้าเงื่อนไขจะใช้ $|\text{train\_mean}|$ หาก $> 10^{-6}$; หากยังไม่เข้าเงื่อนไขจะใช้ `1.0`
- คำนวณค่า Population Stability Index (PSI) ผ่าน `calculate_psi` และอัตราส่วนส่วนเบี่ยงเบนมาตรฐาน `std_ratio` (`batch_std / train_std`)
- **หมายเหตุ (พฤติกรรมใหม่ใน v1.1.0):** ฟีเจอร์จะถูกแจ้งเตือน (Flag) เมื่อ Normalized Mean Shift $> \text{threshold\_pct}$ **หรือ** ค่า PSI $> \text{PSI\_MAJOR\_SHIFT}$ (`0.25`) (จากเดิมที่ตรวจจับเฉพาะ Mean Shift ทำให้ไม่เห็นการขยายตัวของความแปรปรวน เช่น ปัญหาการสึกหรอของเครื่องมือ Tool Wear ที่ค่าเฉลี่ยคงที่แต่ความแปรปรวนกว้างขึ้น)
- แต่ละฟีเจอร์ที่ถูก Flag จะรายงานค่า: `kind` (`"numeric"` หรือ `"categorical"`), `train_mean`, `batch_mean`, `drift_magnitude`, `drift_%`, `psi`, `std_ratio`, `trigger` (`"mean_shift"` หรือ `"psi_shift"`)

**คีย์ `skipped_features` (เพิ่มใน Phase 2.8):** รายงานคอลัมน์ที่ *ไม่ได้ตัดสิน* พร้อมเหตุผล ซึ่งสำคัญกว่าที่คิด เพราะเดิมทีคอลัมน์ที่หายไปจาก batch จะถูกข้ามอย่างเงียบๆ วัดจริง: เทรนด้วย `[temp, vibration]` แต่ batch มีแค่ `[temp]` ระบบเดิมคืน `{'drift_detected': False, 'drifted_features': {}, 'max_psi': 0.0}` — เป็นการ**ยืนยันว่าเซนเซอร์ที่หายไปนั้นปกติดี** ตอนนี้จะได้ `skipped_features['vibration'] = 'missing from the batch'` แทน

**`min_rows` นับ "จำนวนค่าที่อ่านได้จริง" ไม่ใช่จำนวนแถว** คอลัมน์ที่เป็น NaN 98% ยังมีครบ 200 แถว ด่านที่ดูจาก `len(df)` จึงปล่อยผ่าน วัดจริงด้วย 200 แถวที่มีค่าจริงแค่ 4 ค่า โดยสุ่ม batch จาก*การแจกแจงเดียวกับ*ชุดเทรน: **แจ้งเตือนหลอก 100.0%** ตั้ง `min_rows=DRIFT_MIN_ROWS` (30) เพื่อปิดช่องนี้

**`include_categorical`** เปิดการเปรียบเทียบคอลัมน์เชิงกลุ่ม (Operator, Recipe, Lot, Shift) ผ่าน `calculate_categorical_psi` ค่าเริ่มต้นเป็น `False` เพื่อให้โค้ดเดิมทั้งหมดได้ผลลัพธ์เท่าเดิมทุกประการ

#### `calculate_categorical_psi(train_values, batch_values, max_categories=50) -> float or None`
PSI สำหรับคอลัมน์เชิงกลุ่ม `calculate_psi` แบ่ง bin ด้วย Quantile ซึ่งใช้กับ label ไม่ได้ ฟังก์ชันนี้เทียบความถี่ของแต่ละหมวดโดยตรง ทำให้การเปลี่ยนสัดส่วน เช่น เปลี่ยนกะ เปลี่ยนสูตรการผลิต หรือเปลี่ยนคนคุมเครื่อง มองเห็นได้แทนที่จะหลุดออกจากรายงานไปเงียบๆ

- หมวดที่โผล่ใน batch แต่ไม่เคยมีในชุดเทรน (พนักงานใหม่ อะไหล่ล็อตใหม่) ถือเป็น**สัญญาณจริง** จึงถูกเก็บไว้และใส่ค่าพื้น $\varepsilon$ ไม่ใช่ทิ้ง
- คืน `None` เมื่อคอลัมน์ตัดสินไม่ได้: มีหมวดเดียว หรือมีมากกว่า `max_categories` (แปลว่าเป็นรหัสประจำตัว ไม่ใช่หมวดหมู่)
- เกณฑ์เดียวกับ `calculate_psi`: < 0.10 นิ่ง, 0.10–0.25 เปลี่ยนปานกลาง, > 0.25 เปลี่ยนมาก

#### `check_asset_drift(train_df, batch_df, asset_col, threshold_pct=0.20, min_rows=30, n_sigma_floor=3.0, include_categorical=True) -> dict`
ตรวจ Drift **แยกรายเครื่องจักร** แทนที่จะรวมทุกเครื่องเป็นก้อนเดียว ดูรายละเอียดและตัวเลขที่วัดได้ในหัวข้อ "Per-Asset Drift" ท้ายเอกสาร

#### `audit_data_quality(df, target_col=None) -> dict`
ตรวจประเมินคะแนนคุณภาพข้อมูล (Data Quality Score: DQS) บน 5 มิติถ่วงน้ำหนัก พร้อมส่งคืนข้อสรุปปัญหา (Issues) และแผนปฏิบัติการแก้ไข (Remediation Plan)

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ประกอบด้วย: `dqs`, `grade`, `verdict`, `total_rows`, `total_columns`, `dimensions`, `silent_nulls`, `outliers`, `duplicate_rows`, `issues`, `remediation` หรือคืนค่า `{"error": "Data quality audit failed."}` เมื่อเกิดข้อผิดพลาด
- คำนวณคะแนน 5 มิติตาม `DQS_WEIGHTS`: Completeness (30%), Consistency (25%), Validity (20%), Uniqueness (15%), Timeliness (10%)
  - Completeness = $100 \times (1 - \frac{\text{explicit nulls} + \text{silent nulls}}{\text{total cells}})$
  - Consistency = $100 \times (1 - \frac{\text{mixed-type columns}}{\text{total columns}})$ โดยคอลัมน์ Mixed-type คือ object dtype ที่มีค่า non-null แปลงเป็นตัวเลขได้ระหว่าง 0% ถึง 100% (ไม่รวม 0% และ 100%)
  - Validity = $100 \times (1 - \frac{\text{flagged outlier cells}}{\text{numeric cells}})$
  - Uniqueness = $100 \times (1 - \frac{\text{duplicate rows}}{\text{total rows}})$
  - Timeliness = $100 \times (1 - \frac{\text{implausible timestamps}}{\text{parsed timestamps}})$ โดย implausible คือเวลาในอนาคตหรือก่อน 1970-01-01 (หากชุดข้อมูลไม่มีคอลัมน์ datetime มิตินี้จะถูกตัดออก และค่าน้ำหนักที่เหลือจะถูก Renormalise ให้รวมกันได้ 1.0)
- เกณฑ์การตัดเกรด (`grade`): $\ge 85.0$ (`"production_ready"`), $\ge 65.0$ (`"usable_with_caveats"`), ต่ำกว่า 65.0 (`"remediation_required"`)
- เรียงลำดับ `issues` ตามความรุนแรง: critical > high > medium > low
- ส่งคืน `remediation` เป็น Action strings สูงสุด 10 รายการที่ตัดข้อความซ้ำออกแล้ว
- Playbook จัดการ Missing Values รายคอลัมน์ (Severity -> Action):
  - $< 1\%$ (low) -> ตัดแถวที่ได้รับผลกระทบ หรือ Impute ด้วย median/mode
  - $1-10\%$ (medium) -> Impute และสร้าง indicator column `<col>_was_null`
  - $10-30\%$ (high) -> Impute อย่างระมัดระวังและสืบหาสาเหตุต้นน้ำ
  - $> 30\%$ (critical) -> ห้าม Impute โดยไม่ตรวจสอบ; ส่งให้ผู้เชี่ยวชาญทบทวนหรือตัดคอลัมน์ทิ้ง
- บันทึก `verdict` ผ่าน `logger.warning` เมื่อเกรดเป็น `"remediation_required"` และ `logger.info` ในกรณีอื่น ๆ

#### `detect_silent_nulls(df) -> dict`
ตรวจหาค่าที่แทนความหมายว่าข้อมูลสูญหาย (Missing) แต่ไม่ได้ถูกเก็บเป็นค่า NaN จริง

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ในรูปแบบ `{column: {"count", "ratio", "tokens", "kind"}}` หากไม่พบหรือเกิดข้อผิดพลาดจะคืนค่า `{}`
- คอลัมน์ object/string: ตรวจจับข้อความที่ตรงกับ `SILENT_NULL_TOKENS` แบบ case-insensitive หลังตัดช่องว่าง (`.strip()`) รายงาน `kind = "placeholder_string"` (ค่า NaN จริงจะไม่ถูกนับเป็น placeholder)
- คอลัมน์ตัวเลข: ตรวจจับค่าที่ตรงกับ `NUMERIC_SENTINELS` (`-999, -9999, -99999, 999999, -1e30, 1e30`) รายงาน `kind = "numeric_sentinel"`
- ตัวเลข Sentinel จะถูกรายงานเท่านั้น **ไม่มีการแปลงค่าอัตโนมัติ** เนื่องจากค่าเซนเซอร์จริงหน้างานอาจมีค่า -999 ได้อย่างถูกต้อง

#### `detect_outliers(df, method="modified_zscore", threshold=None) -> dict`
ตรวจจับค่าผิดปกติทางสถิติ (Outliers) ในคอลัมน์ตัวเลขโดยไม่แก้ไขหรือแตะต้องข้อมูลดิบ

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ในรูปแบบ `{column: {"count", "ratio", "method", "threshold", "flagged_min", "flagged_max"}}` หากไม่พบหรือเกิดข้อผิดพลาดจะคืนค่า `{}`
- ค่าเริ่มต้น Threshold: `3.5` สำหรับ `"modified_zscore"`, `1.5` สำหรับ `"iqr"`
- วิธี `"modified_zscore"` ใช้สถิติ Iglewicz-Hoaglin: $0.6745 \times \frac{|x - \text{median}|}{\text{MAD}}$
- กรณี MAD เป็น 0 (มีข้อมูลซ้ำกันเกินครึ่ง) จะ fallback ไปใช้สูตร $\frac{|x - \text{median}|}{1.253314 \times \text{mean\_absolute\_deviation}}$ และหากคอลัมน์เป็นค่าคงที่ทั้งหมดจะไม่รายงาน
- ค่า `flagged_min` / `flagged_max`: สำหรับ `"iqr"` จะเป็นค่าขอบเขต Fence (Lower/Upper); สำหรับ `"modified_zscore"` จะเป็นค่าต่ำสุดและสูงสุดของแถวข้อมูลที่ถูก Flag
- ไม่มีการแก้ไขข้อมูลดิบ เนื่องจากค่าผิดปกติทางกายภาพกับค่าสุดโต่งที่เกิดขึ้นจริงไม่สามารถแยกได้หากไม่มีความรู้เฉพาะทาง (Domain Input)

#### `calculate_psi(train_values, batch_values, n_bins=10) -> float or None`
คำนวณค่า Population Stability Index (PSI) เพื่อวัดการเปลี่ยนแปลงของการกระจายตัวของข้อมูลระหว่าง Train และ Live Batch

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- แบ่งข้อมูลชุดเทรนออกเป็น Quantile Bins ตามจำนวน `n_bins` (ค่าเริ่มต้น `10`) และสร้างเวกเตอร์ความถี่ของการเทรน
- ทำการ Clip ค่าในชุด Batch ให้อยู่ในช่วงขอบเขตของ Train เพื่อให้ค่าที่หลุดขอบเขตตกลงใน Bin หัวท้าย
- กำหนดค่า Floor สำหรับ Bin ที่ว่างเปล่าด้วย `1e-4` เพื่อให้ค่าลอการิทึมเป็นค่าจำกัด (Finite)
- คำนวณสูตร PSI: $\sum (b_{\text{pct}} - t_{\text{pct}}) \times \ln\left(\frac{b_{\text{pct}}}{t_{\text{pct}}}\right)$
- เกณฑ์การแปลผล (Bands): $< 0.10$ มีเสถียรภาพ (Stable), $0.10-0.25$ มีการเปลี่ยนแปลงปานกลาง (Moderate Shift), $> 0.25$ มีการเปลี่ยนแปลงอย่างมีนัยสำคัญ (Major Shift)
- คืนค่า `None` หากชุดข้อมูลเทรนมีจำนวนแถวน้อยกว่า `n_bins` หรือมีค่าไม่ซ้ำ (distinct values) น้อยกว่า 3 ค่า
- ตอบสนองต่อการเปลี่ยนแปลงในทุกตำแหน่งของการกระจายตัว ช่วยตรวจจับปัญหาความแปรปรวนขยายตัว (Variance Inflation) จากการสึกหรอของเครื่องมือ (Tool Wear) ที่การเปรียบเทียบค่าเฉลี่ยตรวจไม่พบ

#### `wilson_confidence_interval(successes, trials, confidence=0.95) -> dict`
คำนวณช่วงความเชื่อมั่นของ Wilson Score Interval สำหรับสัดส่วนแบบทวินาม (Binomial Proportion)

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ประกอบด้วย: `point`, `lower`, `upper`, `n`, `confidence`
- เหมาะสำหรับงานอุตสาหกรรมที่มีจำนวนของเสียน้อย (Small Defect Counts) เช่น พบของเสีย 3 ชิ้นจาก 30 ตัวอย่าง ให้ค่า point `0.100`, lower `0.0346`, upper `0.2562` (ขณะที่วิธี Normal Approximation จะให้ค่า lower ติดลบ) และที่ 50/50 ค่า upper bound จะอยู่ที่ `1.0` พอดี ไม่เกิดช่วง degenerate `[1, 1]`
- กรณี Input ไม่ถูกต้อง (`trials <= 0`, `successes > trials`, `confidence` ไม่อยู่ในช่วง $(0, 1)$) จะคืนค่า `point=None`, `lower=None`, `upper=None` โดยไม่ raise exception

#### `check_calibration(y_true, y_prob, n_bins=10) -> dict` *(ใหม่ v1.4.0)*
ตรวจว่า Probability ที่โมเดลทำนายออกมา "หมายความตามที่พูดจริงหรือไม่" — คนละเรื่องกับ Discrimination (การจัดอันดับถูก/ผิด)
โมเดลอาจมี AUC สูงมากแต่ Probability บิดเบี้ยวจนอ่านเป็นความน่าจะเป็นจริงไม่ได้เลยก็ได้

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- แบ่ง `y_prob` เป็น `n_bins` ช่วงเท่า ๆ กัน แล้วเทียบ Mean Predicted Probability กับ Observed Rate จริงในแต่ละช่วง
- คืนค่า `brier_score` (ยิ่งต่ำยิ่งดี), `brier_skill_score` (เทียบกับการทายด้วย Base Rate เฉย ๆ — `0` คือไม่ดีไปกว่านั้น), `expected_calibration_error` (ECE), `max_calibration_error` (MCE), `is_well_calibrated` (ECE ≤ `CALIBRATION_ECE_LIMIT = 0.05`), ตาราง `bins` รายช่วง และ `interpretation` เป็นประโยคอธิบาย
- ทำไมต้องมี: `optimize_threshold()` เลือกจุดตัดที่ถูกที่สุด**บนคะแนนของโมเดล** ไม่ว่าคะแนนนั้นจะ Calibrate แล้วหรือไม่ก็ตาม ถ้า Calibration แย่ จะพูดว่า "จุดตัดนี้คือโอกาสเสีย 30%" ไม่ได้ และย้าย Threshold ไปใช้กับไลน์ที่มี Failure Rate ต่างกันไม่ได้ด้วย
- ต้องการ Class แบบทวินามเท่านั้น (สอง class พอดี) — รับทั้ง Label ตัวเลข 0/1 และ Label ข้อความ (เช่น `"OK"`/`"NG"`) โดยจะเลือก Label ที่เรียงลำดับหลังสุดเป็น Positive class
- คืนค่า `{"error": ...}` และไม่ raise เมื่อ input ผิดรูปแบบ: ความยาวไม่ตรงกัน, มีคลาสเดียว, `n_bins < 2`, หรือ `y_prob` อยู่นอกช่วง `[0, 1]`

#### `enable_audit_log(filepath="potatopt_audit.log", level=logging.INFO) -> str or None`
เปิดการบันทึก Guardrail และ Drift Events ลงไฟล์ Log สำหรับการสอบย้อนกลับตามมาตรฐาน ISO 9001

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ผูก `FileHandler` เข้ากับ logger `"potatopt"` ด้วย Format `"%(asctime)s | %(levelname)s | %(message)s"`
- คืนค่า Absolute Path ของไฟล์ Log หรือคืนค่า `None` หากเกิดข้อผิดพลาดด้านระบบไฟล์
- มีคุณสมบัติ Idempotent: การเรียกซ้ำด้วยพาธเดิมจะไม่เพิ่ม Handler ซ้ำซ้อน
- รองรับการสอบย้อนกลับ (Traceability) ตามข้อกำหนด ISO 9001 เพื่อให้ Operator สามารถพิสูจน์ย้อนหลังได้ว่า Batch การผลิตใดทำให้เกิดสัญญาณเตือน Warning เมื่อใด

#### `get_library_versions() -> dict`
รวบรวมเวอร์ชันของไลบรารีใน Runtime Stack สำหรับการทำซ้ำ (Reproducibility) ตามมาตรฐาน ISO 9001

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ตรวจสอบเวอร์ชันของ: `numpy`, `pandas`, `scipy`, `scikit-learn`, `flaml`, `lightgbm`, `xgboost`, `shap`, `joblib`
- แพ็กเกจที่ไม่ได้ติดตั้งจะให้ค่าเป็น `None`

#### `to_jsonable(value) -> dict | list | str | int | float | bool | None`
แปลงผลลัพธ์ที่มี NumPy / pandas ปนอยู่ให้กลายเป็น Python ธรรมดาที่ `json.dumps` รับได้ สำหรับใช้เป็นด่านเชื่อมต่อกับ MCP Server, REST API หรือ Dashboard

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- แปลงชนิดข้อมูล NumPy (`np.integer`, `np.floating`, `np.bool_`, `np.ndarray`) และ pandas (`DataFrame` -> list of records, `Series`/`Index` -> list) ลงมาเป็นชนิดพื้นฐานของ Python
- `NaN` และ `Infinity` ถูกแปลงเป็น `None` เพราะ `json.dumps` จะปล่อยโทเคน `NaN` / `Infinity` ออกมาดิบๆ ซึ่งไม่ใช่ JSON ที่ถูกต้องและจะพังกับ Parser ที่เข้มงวดอย่าง `JSON.parse` ของ JavaScript
- เดินลงไปในโครงสร้างซ้อนกัน (`dict`, `list`, `tuple`, `set`) ทุกชั้น และแปลง Key ของ `dict` เป็น `str` เสมอ
- ชนิดข้อมูลที่ไม่รู้จักจะถูกแปลงด้วย `str()` แทนการ Raise — ด่านแปลงข้อมูลต้องไม่ทำให้ Request ทั้งก้อนล้ม
- ใช้คู่กับ `.predict()` (คืน `ndarray`) และ `.get_feature_importance()` (คืน `DataFrame`) ได้ทันที โดยทั้งสองเมธอดไม่ต้องเปลี่ยนพฤติกรรมเดิม

---

### 2. PotatOptEngine — Constructor & Public Methods

#### คอนสตรัคเตอร์ (Constructor)

| พารามิเตอร์ | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `task` | `"auto"` | เลือก `"auto"`, `"classification"`, `"regression"`, `"forecasting"` — โหมด auto จะตัดสินใจให้ตอน fit() |
| `time_budget` | `30` | งบเวลาค้นหาโมเดลของ FLAML (วินาที) — ถูกบังคับขั้นต่ำเป็น 1 ด้วย `max(1, time_budget)` |
| `scale_method` | `'standard'` | `'standard'` = StandardScaler, `'minmax'` = MinMaxScaler, ค่าอื่นจะไม่ scale เลย |
| `apply_smote` | `None` | Alias เดิมของ `cost_sensitive_weighting` — ถ้าใส่ค่าไม่ใช่ None จะ override `cost_sensitive_weighting` |
| `cost_sensitive_weighting` | `False` | Opt-in: ถ่วงน้ำหนักคลาสแทน SMOTE เพื่อประหยัด RAM |
| `collinear_threshold` | `0.9` | เกณฑ์ correlation ที่จะตัดฟีเจอร์ซ้ำซ้อนทิ้ง — ตั้ง `1.0` ขึ้นไปเพื่อปิดฟีเจอร์นี้ |
| `estimators` | `None` | ถ้าเป็น None จะใช้ `["lgbm", "xgboost", "rf"]` |
| `handle_silent_nulls` | `True` | แปลง `SILENT_NULL_TOKENS` ในคอลัมน์ object เป็น NaN ทั้งตอน fit และ inference ก่อน imputation เพื่อให้ถูก impute แทนการถูก encode เป็น category ปกติ |
| `audit_data` | `True` | รัน `audit_data_quality` บนข้อมูลดิบระหว่าง `fit()` และเก็บผลลัพธ์แบบย่อใน `self.train_data_quality` |
| `n_jobs` | `-1` | จำนวน Worker ที่ส่งต่อให้ `IsolationForest` และ FLAML ตามธรรมเนียม joblib (`-1` = ทุกคอร์, `-2` = เว้นไว้ 1 คอร์, จำนวนเต็มบวก = ระบุตรงๆ) — Raise `ValueError` หากไม่ใช่จำนวนเต็มหรือเป็น `0` เปิดให้ปรับได้เพราะการใช้ครบทุกคอร์บนเครื่อง 2 คอร์มักช้ากว่าการใช้คอร์เดียว และเครื่องสเปกต่ำยังต้องเหลือกำลังไว้ใช้งานอย่างอื่นระหว่างเทรน |
| `random_state` | `42` | *(ใหม่ v1.4.0)* Seed เดียวที่คุมทั้ง `IsolationForest` (ทั้งสองจุด fallback) และ `seed` ของ FLAML — เดิมเป็น `42` ตายตัวในโค้ดสามจุดแยกกัน ทำให้ผลลัพธ์ทุกอันเป็น Single-seed โดยไม่มีใครสังเกตเห็น ค่าเริ่มต้นยังคงเป็น `42` เพื่อไม่ให้โค้ดเดิมพฤติกรรมเปลี่ยน ถูกบันทึกไว้ใน `get_training_report()` และ Metadata ของ `.save()` — Raise `ValueError` หากไม่ใช่จำนวนเต็ม |

#### เมธอดสาธารณะ (Public Methods)

#### `.fit(X_train, y_train)` -> `self`
ประมวลผลข้อมูลผ่าน Preprocessing Pipeline 11 ขั้นตอน และค้นหาโมเดลที่ดีที่สุดโดยอัตโนมัติ

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- รองรับ input หลากหลายรูปแบบ: DataFrame, numpy array หรือ scipy sparse matrix (แปลงอัตโนมัติด้วย `.toarray()`)
- เรียกใช้ `_reset_state()` เสมอ เพื่อให้สามารถนำ engine object เดิมมาสั่ง refit ใหม่ได้อย่างปลอดภัยโดยไม่มี state ตกค้าง
- การตัดสินใจเมื่อ `task="auto"`: แปลงค่า `y` ด้วย `pd.to_numeric` หากมากกว่า 90% ของข้อมูลเป็นตัวเลข และมีค่าไม่ซ้ำมากกว่า 10 ค่า จะกำหนดเป็น `"regression"` นอกเหนือจากนั้นจะกำหนดเป็น `"classification"`
- แถวที่มี target เป็น NaN จะถูกตัดทิ้ง หากไม่มีข้อมูลเหลืออยู่เลยจะ Raise `ValueError`
- กรณี Classification หากมีจำนวนคลาสที่แตกต่างกันน้อยกว่า 2 คลาส จะ Raise `ValueError`
- กำหนด `pos_label_idx` ให้ตรงกับค่า encoded ของคลาสส่วนน้อย (Minority Class) เฉพาะในงาน Binary Classification เพื่อป้องกันปัญหา Alphabetical LabelEncoder Trap
- Cost-sensitive weighting จะเปิดทำงานจริงเมื่อเปิดใช้งาน (`True`) **และ** สัดส่วนคลาสส่วนน้อย $< 0.20$ เท่านั้น โดยคำนวณน้ำหนักผ่าน `compute_sample_weight(class_weight='balanced')`
- การตั้งค่า FLAML ที่ใช้: `metric="auto"`, `seed=self.random_state` (ค่าเริ่มต้น `42`), และ `estimator_list` ตามที่ระบุใน constructor
- Raise `RuntimeError` หาก FLAML ค้นหาโมเดลเสร็จสิ้นแล้วไม่พบ `best_estimator`

#### `.predict(X)` -> `numpy.ndarray`
ทำนายผลสำหรับข้อมูลชุดใหม่ โดยส่งคืนผลลัพธ์เป็น Label ดั้งเดิม

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- รองรับ input หลากหลาย: DataFrame, Series, dict, list, numpy array, scipy sparse
- กรณี Anomaly Model (IsolationForest): หากผลลัพธ์เป็น `1` -> คืนค่า Label ของคลาสส่วนใหญ่ (Majority Class); หากเป็น `-1` -> คืนค่า Label ของคลาสส่วนน้อย (Minority Class) หรือคืนค่า String `"ANOMALY"` ในกรณีที่เป็น Multiclass-Scarcity Sentinel
- กรณีโมเดลปกติ: หากเป็น Binary Classification และมีการตั้งค่า `optimal_threshold != 0.5` ระบบจะคำนวณ Label ใหม่จากเงื่อนไข `predict_proba[:, pos_label_idx] >= optimal_threshold` แทนการใช้ argmax ตามปกติของโมเดล
- คืนค่าเป็น Label ดั้งเดิมเสมอ (ผ่าน `LabelEncoder.inverse_transform`) ไม่ใช่ตัวเลข integer ที่ถูก encode ไว้

#### `.predict_proba(X)` -> `numpy.ndarray` (2 คอลัมน์)
คำนวณค่าความน่าจะเป็นของแต่ละคลาส หรือค่า Calibrated Anomaly Score

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- Raise `ValueError` หาก `task` ไม่ใช่ Classification
- สำหรับ Anomaly Model จะคืนค่า Calibrated Score (ไม่ใช่ค่าความน่าจะเป็นจริง):
  $$P(\text{anomaly}) = \frac{1}{1 + \exp(\text{clip}(\text{score} \times 5.0, -15.0, 15.0))}$$
  โดย `score` มาจาก `IsolationForest.decision_function` ซึ่งสูตรนี้เป็น Deterministic ต่อแถว ทำให้การส่งข้อมูลแถวเดียวหรือแบตช์ 10,000 แถวได้ค่าตรงกันอย่างแม่นยำ
- คืนค่า `None` หากโมเดลพื้นฐานไม่มีฟังก์ชัน `predict_proba`

#### `.optimize_threshold(X_val, y_val, cost_scrap=500, cost_fa=150, cost_insp=20)` -> `float`
ค้นหาจุดตัด Threshold ที่ให้ต้นทุนรวมของเสียและต้นทุนการตรวจสอบต่ำที่สุดตามหลักการ Cost of Quality

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `0.5` ทันที (ไม่มีการปรับจูน) หาก: `task` ไม่ใช่ Classification, หรือเป็น Anomaly Model, หรือ LabelEncoder มีจำนวนคลาสไม่ใช่ 2 คลาส
- กวาดหา Threshold ที่ดีที่สุดในช่วง `np.arange(0.05, 0.95, 0.05)`
- สมการต้นทุนรวมที่ใช้หาค่าต่ำสุด:
  $$\text{Cost} = (\text{FN} \times \text{cost\_scrap}) + (\text{FP} \times \text{cost\_fa}) + (\text{TP} \times \text{cost\_insp})$$
- บังคับใช้ `confusion_matrix(..., labels=[0, 1])` เพื่อป้องกันปัญหา Matrix ยุบมิติและ Crash ในกรณีที่ชุดทดสอบไม่มีของเสียเลย (Zero Defects)
- บันทึกค่า Threshold ที่ดีที่สุดลงใน `self.optimal_threshold` เพื่อให้ `.predict()` นำไปใช้ตัดสินใจต่อโดยอัตโนมัติ
- **ต้องส่ง Validation Set เท่านั้น ไม่ใช่ Test Set** — Threshold ถูกเลือกให้ "ถูกที่สุดบนข้อมูลชุดนี้" ถ้าเอาชุดเดียวกันไปรายงานผลต่อ ตัวเลขต้นทุนที่ประหยัดได้จะเอนเข้าข้างตัวเองอย่างเป็นระบบ (ใช้ `split_data_three_way()` แบ่ง Validation ออกมาต่างหาก)
- บันทึกลายนิ้วมือของชุดข้อมูลที่ใช้จูนไว้ใน `self.threshold_tuning_fingerprint` (แฮช SHA-1 ของคอลัมน์ target ประกอบกับขนาดตาราง) โดยบันทึกเฉพาะเส้นทางที่จูนสำเร็จจริง — กรณีคืนค่า `0.5` ก่อนกำหนดจะไม่บันทึก
- `.evaluate()` จะนำลายนิ้วมือนี้ไปเทียบกับชุดที่กำลังรายงานผล ถ้าตรงกันจะขึ้น `logger.warning` และตั้ง `threshold_leakage_warning = True`
- **ข้อควรระวังเรื่องการตีความ (v1.4.0):** Threshold ที่ได้คือจุดตัดที่ถูกที่สุด**บนคะแนนดิบของโมเดล** ไม่ว่าคะแนนนั้นจะเป็น Probability ที่ Calibrate แล้วหรือไม่ก็ตาม ไม่ใช่คำยืนยันว่าเครื่องมีโอกาสพังตามตัวเลขนั้นจริง ควรเรียก `.check_calibration()` บน Validation Set ชุดเดียวกันก่อนจะนำ Threshold ไปพูดเป็น "โอกาส %", อ้างต้นทุนคาดหวังต่อครั้ง หรือย้ายไปใช้กับไลน์ที่ Failure Rate ต่างกัน

#### `.check_calibration(X, y, n_bins=10) -> dict` *(ใหม่ v1.4.0)*
ตรวจว่า Probability ของโมเดลตัวนี้ "หมายความตามที่พูดจริงหรือไม่" — เป็น Wrapper บาง ๆ รอบฟังก์ชัน `check_calibration()` ระดับโมดูล

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ดึงคอลัมน์ Probability ของ Positive Class จาก `.predict_proba()` แล้ว Map `y` ให้ตรงกับคลาสเดียวกันก่อนส่งต่อ
- ควรเรียกบน **Validation Set** ชุดเดียวกับที่ใช้ `.optimize_threshold()` เพราะเป็นชุดข้อมูลที่ตัวเลขต้นทุนอ้างอิงอยู่
- คืนค่า `{"error": ...}` และไม่ raise หาก: ยังไม่ `fit()`, `task` ไม่ใช่ Classification, หรือ LabelEncoder มีมากกว่า 2 คลาส
- กรณี Anomaly Fallback Model จะคืนค่า `is_well_calibrated = False` เสมอ พร้อม `probability_source = "isolation_forest_sigmoid"` เพราะ `predict_proba` ของโมเดลนี้คือ Sigmoid ตายตัวบน `decision_function` ของ `IsolationForest` ออกแบบมาให้ใช้จัดอันดับ (Ranking) เท่านั้น ไม่ใช่ประมาณการ Failure Rate — โมเดลปกติจะคืนค่า `probability_source = "model_predict_proba"`

#### `.evaluate(X_test, y_test)` -> `dict`
ประเมินประสิทธิภาพของโมเดลตามประเภทของ Task พร้อมเมทริกซ์และรายงานผลอย่างละเอียด

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- **Binary Classification** คืนค่าคีย์:
  - `task_type`: `"binary_classification"`
  - `threshold_leakage_warning`: `True` เมื่อ Threshold ถูกจูนบนชุดข้อมูลเดียวกับที่กำลังรายงานผลอยู่ (ตัวเลขจะดีเกินจริง) — ปกติต้องเป็น `False`
  - `accuracy`: ความแม่นยำรวม
  - `precision`: ความแม่นยำในการตรวจจับของเสีย
  - `recall`: อัตราการตรวจจับของเสีย (Detection Rate)
  - `f1`: Harmonic Mean ของ Precision และ Recall
  - `confusion_matrix`: Confusion Matrix ในรูปแบบลิสต์ 2D
  - `best_model_name`: ชื่ออัลกอริทึมแชมเปียน
  - `threshold_used`: ค่า Threshold ที่ใช้ตัดสินใจ
  - `roc_auc`: ค่า ROC-AUC หรือ `None` หากชุดทดสอบมีเพียงคลาสเดียว
  - `pr_auc`: ค่า Average Precision (PR-AUC) ซึ่งเป็นตัวชี้วัดตัดสินสำคัญสำหรับ Rare Defects (ที่อัตราของเสีย 1% ทั้ง Accuracy และ ROC-AUC ยังคงมองโลกในแง่ดี ขณะที่ PR-AUC จะดิ่งลงสู่ Base Rate)
  - `defect_base_rate`: สัดส่วนของคลาสบวกในชุดทดสอบ (Floor ขั้นต่ำของ Random Model บน PR-AUC)
  - `mcc`: Matthews Correlation Coefficient (ช่วง -1 ถึง 1)
  - `recall_ci_95`: ช่วงความเชื่อมั่น 95% ของ Recall จาก Wilson Score Interval (`dict`: `point`, `lower`, `upper`, `n`, `confidence`)
  - `precision_ci_95`: ช่วงความเชื่อมั่น 95% ของ Precision จาก Wilson Score Interval (`dict`: `point`, `lower`, `upper`, `n`, `confidence`)
  - `n_test_rows`: จำนวนแถวในชุดทดสอบ
- **Multi-class Classification** คืนค่าคีย์:
  - `task_type`: `"multi_class_classification"`
  - `accuracy`: ความแม่นยำรวม
  - `macro_f1`: ค่า Macro-averaged F1 Score
  - `classification_report`: รายงานผลจำแนกรายคลาส
  - `confusion_matrix`: Confusion Matrix
  - `best_model_name`: ชื่ออัลกอริทึมแชมเปียน
  - `mcc`: Matthews Correlation Coefficient หรือ `None`
  - `n_test_rows`: จำนวนแถวในชุดทดสอบ
- **Regression / Forecasting** คืนค่าคีย์:
  - `task_type`: `"regression_forecasting"`
  - `r2`: ค่า R-squared
  - `rmse`: Root Mean Squared Error
  - `best_model_name`: ชื่ออัลกอริทึมแชมเปียน
  - `mae`: Mean Absolute Error
  - `mape`: Mean Absolute Percentage Error (คิดเป็นเปอร์เซ็นต์ โดยตัดแถวที่ค่าจริงเป็น 0 ออกเนื่องจากหาค่าไม่ได้; คืนค่า `None` หากค่าจริงเป็น 0 ทั้งหมด)
  - `n_test_rows`: จำนวนแถวในชุดทดสอบที่ถูกต้อง
  - เส้นทาง Regression จะตัดแถวที่ไม่สามารถแปลงเป็นตัวเลขได้ใน `y_test` หรือ `y_pred` ทิ้ง และหากไม่มีข้อมูลที่ถูกต้องเหลืออยู่เลย จะคืนค่า `r2=0.0`, `rmse=0.0`, `mae=0.0`, `mape=None`, `n_test_rows=0` แทนการ crash
- **Anomaly Sentinel Fallback** คืนค่า dict โครงสร้างเดียวกับ Multi-class Classification พร้อมเพิ่มฟิลด์ `"note"` เพื่ออธิบายสถานะ fallback และ `"n_test_rows"`
- ตรวจ Threshold Leakage ทุกครั้งที่เรียก: เทียบลายนิ้วมือชุดข้อมูลปัจจุบันกับ `self.threshold_tuning_fingerprint` ถ้าตรงกันจะขึ้น `logger.warning` และตั้งคีย์ `threshold_leakage_warning` เป็น `True` — การตรวจนี้ไม่เคย Raise และไม่เคยขวางการประเมินผล หากแฮชข้อมูลไม่ได้จะข้ามการตรวจไปเงียบๆ

#### `.calculate_cost_of_quality(X_test, y_test, cost_scrap=500.0, cost_fa=150.0, cost_insp=20.0, base_det_rate=0.0, base_fp_rate=0.0)` -> `dict`
คำนวณเปรียบเทียบต้นทุนคุณภาพทางการเงินระหว่างระบบเดิมกับโมเดล Machine Learning

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- รองรับเฉพาะงาน Standard Binary Classification เท่านั้น โดยจะคืนค่า `{"error": ...}` หากเป็น Regression, Multi-class หรือ Anomaly Model
- `base_det_rate` คืออัตราการตรวจจับของเสียของระบบเดิม (Manual/SPC) และ `base_fp_rate` คืออัตราการเตือนผิดพลาดของระบบเดิม เพื่อใช้สร้าง Baseline ในการเปรียบเทียบ
- $\text{baseline\_cost} = (\text{base\_FN} \times \text{cost\_scrap}) + (\text{base\_FP} \times \text{cost\_fa}) + (\text{base\_TP} \times \text{cost\_insp})$
- $\text{model\_cost} = (\text{FN} \times \text{cost\_scrap}) + (\text{FP} \times \text{cost\_fa}) + (\text{TP} \times \text{cost\_insp})$
- คืนค่า: `baseline_cost`, `model_cost`, `cost_savings`, `savings_percentage` (ข้อความ string เช่น `"42.50%"`)

#### `.get_feature_importance()` -> `DataFrame` หรือ `None`
ดึงค่าน้ำหนักความสำคัญของแต่ละฟีเจอร์ เรียงลำดับจากมากไปน้อย

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `None` หากเป็น Anomaly Model หรือยังไม่ได้ผ่านการ `.fit()`
- รองรับทั้ง Tree-based models ผ่าน `feature_importances_` และ Linear/SGD models ผ่าน `coef_` (สำหรับ `coef_` มิติ 2D จะใช้ค่าเฉลี่ย `mean(abs(coef))` ข้ามทุกคลาส)
- คืนค่า `None` หากความยาวของเวกเตอร์ Importance ไม่ตรงกับ `self.feature_names` (เป็นการตรวจสอบความปลอดภัย ไม่ crash)
- จัดเรียงลำดับผลลัพธ์ใน DataFrame จากค่าความสำคัญมากไปน้อย (Descending)

#### `.get_shap_values(X_test)` -> `tuple (explainer, shap_values)`
คำนวณค่า SHAP (SHapley Additive exPlanations) เพื่ออธิบายการตัดสินใจของโมเดล

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `(None, None)` หากเป็น Anomaly Model หรือยังไม่ได้ผ่านการ `.fit()`
- มีระบบสำรอง 3 ชั้น (Tri-layer Fallback) ตามลำดับ: `shap.TreeExplainer` -> `shap.LinearExplainer` -> Generic `shap.Explainer`
- คอลัมน์ Categorical จะถูกแปลงเป็น Numeric Codes ล่วงหน้า (โดยแทนค่า NaN ด้วย `-1`) ก่อนส่งเข้าคำนวณ SHAP

#### `.explain_predictions(X, top_k=None, max_rows=1000)` -> `dict`
มุมมองแบบ JSON ของ `get_shap_values()` — ตัด SHAP Explainer Object ทิ้ง แล้วยุบเมทริกซ์รายแถวให้เหลือค่าเฉลี่ยสัมบูรณ์ต่อฟีเจอร์ เรียงจากมากไปน้อย

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่าเป็น dict ที่ `json.dumps` รับได้เสมอ และ **ไม่เคย Raise** — หากอธิบายไม่ได้จะคืน `available: False` พร้อม `reason` ที่อ่านรู้เรื่อง (ยังไม่ Fit, อยู่ในโหมด Anomaly Fallback, หรือ SHAP ล้มเหลวกับ Estimator ตัวนั้น)
- `max_rows` (ค่าเริ่มต้น `1000`) จำกัดจำนวนแถวที่นำไปคำนวณ โดยตัดจากส่วนหัวของ `X` เพราะ SHAP เป็นขั้นตอนที่กิน RAM มากที่สุดในไปป์ไลน์
- `top_k` เก็บเฉพาะฟีเจอร์ที่มีอิทธิพลสูงสุด N อันดับ (`None` = เก็บทั้งหมด)
- รองรับรูปแบบผลลัพธ์ของ SHAP ทุกทรง: `Explanation` object, list ของ array แยกตามคลาส, และ array ทรง `(rows, features)` หรือ `(rows, features, classes)`
- คีย์ที่คืน: `available`, `reason`, `n_rows_explained`, `top_k`, `feature_attributions` (ลิสต์ของ `{"feature": str, "mean_abs_shap": float}` เรียงจากมากไปน้อย)

#### ความเข้ากันได้กับ scikit-learn (Estimator Compatibility)
`PotatOptEngine` สืบทอด `sklearn.base.BaseEstimator` ทำให้นำไปใช้กับเครื่องมือมาตรฐานของ scikit-learn ได้ทันที

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- ได้ `get_params()` / `set_params()` / `clone()` / `__repr__` มาจาก `BaseEstimator` โดยตรง — คอนสตรัคเตอร์เดิมผ่าน `clone()` ได้โดยไม่ต้องแก้อะไร
- `__sklearn_tags__()` ประกาศชนิดของ Estimator ตาม `task` ที่ตั้งไว้ (`classification` -> classifier, `regression`/`forecasting` -> regressor) **เรื่องนี้มีผลจริงกับตัวเลข:** ถ้าไม่ประกาศ scikit-learn จะถอยไปใช้ `KFold` ธรรมดา ซึ่งบนข้อมูลที่ของเสียน้อยอาจได้ Fold ที่แทบไม่มีคลาสบวกเลย — จากการวัดจริง F1 แบบ 3-fold เปลี่ยนจาก `[0.000, 0.706, 0.737]` เป็น `[0.571, 0.743, 0.800]` หลังจากประกาศ Tag แล้วได้ `StratifiedKFold`
- `task="auto"` จะไม่ถูกติด Tag โดยเจตนา เพราะ Task ถูกตัดสินตอน `fit()` แต่ scikit-learn ถาม Tag ก่อนหน้านั้น — ถ้าต้องการ Stratified CV ให้ระบุ `task="classification"` ตรงๆ
- `classes_` เป็น property ที่คืนคลาสจาก LabelEncoder และ Raise `AttributeError` ก่อน `fit()` ตามธรรมเนียมของ scikit-learn (ทำให้ `hasattr(engine, "classes_")` คืน `False` อย่างถูกต้อง) — Scorer ฝั่ง Classification ของ scikit-learn ต้องใช้ Attribute นี้
- `__sklearn_is_fitted__()` บอก `check_is_fitted` ว่าโมเดลเทรนแล้ว เพราะสถานะถูกเก็บใน `is_fitted` ซึ่งไม่ได้ลงท้ายด้วย `_` ตามที่ scikit-learn มองหา — ถ้าไม่มีตัวนี้ `Pipeline.predict()` จะโยน `NotFittedError` ทั้งที่โมเดลเทรนเสร็จแล้ว
- ต้องใช้ scikit-learn เวอร์ชัน 1.6 ขึ้นไปจึงจะได้ระบบ Tag นี้ เวอร์ชันเก่ากว่ายังใช้งานได้ปกติแต่จะไม่ได้ `StratifiedKFold` อัตโนมัติ

ตัวอย่าง:
```python
from sklearn.model_selection import GridSearchCV
from potatopt import PotatOptEngine

search = GridSearchCV(
    PotatOptEngine(task="classification", time_budget=10, n_jobs=1),
    {"scale_method": ["standard", "minmax"], "collinear_threshold": [0.8, 0.9]},
    cv=3, scoring="f1",
)
search.fit(X_train, y_train)
print(search.best_params_)
```

#### `.get_training_report()` -> `dict`
รายงานสรุปผลการคัดเลือกโมเดลแชมเปียนและการค้นหาของ AutoML พร้อมค่า Cross-Validation Loss เพื่อเป็นหลักฐานป้องกันปัญหา Overfitting

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `{"error": "Engine is not fitted."}` หากเรียกใช้งานก่อนผ่านการ `.fit()`
- คืนค่า `dict` ประกอบด้วย: `potatopt_version`, `task`, `is_anomaly_model`, `best_estimator`, `best_config`, `validation_loss`, `validation_score`, `metric_optimized`, `search_time_budget_sec`, `estimators_searched`, `train_rows`, `n_features_used`, `cost_sensitive_weighting`, `optimal_threshold`
- ค่า `metric_optimized` จำลองการเลือก Metric อัตโนมัติของ FLAML: Binary -> `"roc_auc"`, Multiclass -> `"log_loss"`, Regression/Forecasting -> `"r2"`
- `validation_score = 1.0 - validation_loss` เฉพาะกรณี Metric เป็น `"roc_auc"` และ `"r2"` (เนื่องจาก FLAML บันทึกเป็น $1 - \text{score}$) ส่วนกรณี `"log_loss"` จะเป็น `None` เนื่องจากเป็น Raw Loss และกรณี Anomaly Fallback จะเป็น `None` เนื่องจากไม่ได้ผ่านการประเมินของ FLAML

#### `.detect_drift(X_batch)` -> `dict`
เปรียบเทียบการกระจายตัวของข้อมูล Batch การผลิตหน้างานกับ Statistical Profile ที่บันทึกไว้ตอนเทรนโมเดล (รองรับขั้นตอน DMAIC Control)

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `dict` ประกอบด้วย: `drift_detected` (bool), `max_psi`, `n_features_checked`, `psi_thresholds`, `units_note`, `features`, `recommendation`, `batch_rows` หรือคืนค่า `{"error": ...}` เมื่อเกิดข้อผิดพลาด
- ทำงานได้บนโมเดลที่ Deploy ไปแล้วโดยไม่ต้องเข้าถึงชุดข้อมูล Train เดิม เนื่องจากระบบบันทึก Statistical Fingerprint (mean, std, min, max, bin edges, bin frequencies) ลงใน `self.train_profile` และถูก Pickle ไปพร้อมโมเดล
- **ข้อกำหนดการออกแบบสำคัญ:** ในการทำนายปกติจะมีการ Clip ค่าให้อยู่ในช่วง Training Bounds แต่ใน `detect_drift` จะเรียกใช้ Preprocessing โดยกำหนด `apply_bounds_clip=False` โดยเจตนา เพื่อให้สามารถตรวจวัดการกระจายตัวที่แท้จริงได้ (จากการทดสอบ: Batch ที่มีความแปรปรวน 4 เท่า จะรายงาน `std_ratio` 2.33 เมื่อถูก Clip และรายงาน 3.94 เมื่อบายพาสการ Clip)
- รายงานค่าต่อฟีเจอร์: `psi`, `severity`, `train_mean_raw`, `batch_mean_raw`, `train_std_raw`, `batch_std_raw`, `mean_shift_sigma`, `std_ratio`, `train_mean_scaled`, `batch_mean_scaled`
- ระดับความรุนแรง (`severity`): `"major"` (PSI > 0.25), `"moderate"` (PSI > 0.10), `"stable"` (PSI <= 0.10), `"unknown"` (ไม่สามารถแบ่ง Bin ได้); โดย `drift_detected = True` เมื่อมีฟีเจอร์ใดฟีเจอร์หนึ่งเป็น `"major"`
- ค่า `*_raw` อยู่ในหน่วยวิศวกรรมดั้งเดิมของ Operator และค่า `*_scaled` อยู่ใน Scaler Space
- บันทึก `recommendation` ลงใน Logger: ระดับ `WARNING` เมื่อตรวจพบ Drift และระดับ `INFO` เมื่อปกติ

#### `.get_inference_health()` -> `dict`
รายงานสุขภาพและสถิติการใช้งานระบบประมวลผล Inference ตั้งแต่โมเดลถูก Fit

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- คืนค่า `{"error": "Engine is not fitted."}` หากยังไม่ได้ผ่านการ `.fit()`
- คืนค่า `dict` ประกอบด้วย: `transform_calls`, `rows_transformed`, `warning_events`, `warning_rate`, `last_predict_warnings`, `silent_nulls_converted_during_fit`, `train_data_quality`
- ค่า `warning_rate` คำนวณจาก `warning_events / transform_calls` (อัตราการเตือนที่สูงขึ้นเป็นสัญญาณเตือนแรกว่าสายการผลิตเริ่มเบี่ยงเบนออกจากชุดข้อมูล Train)
- ตัวนับทั้งหมดจะถูกรีเซ็ตใหม่ทุกครั้งที่มีการเรียก `.fit()`

#### `.save(filepath="potatopt_model.pkl")` -> `str`
บันทึกออบเจกต์ Engine ทั้งหมดลงไฟล์ พร้อมสร้างไฟล์คู่ขนาน Metadata และคำนวณลายเซ็นดิจิทัล SHA-256

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- สร้างโฟลเดอร์ปลายทางให้อัตโนมัติ (`os.makedirs(..., exist_ok=True)`) เพื่อป้องกัน `FileNotFoundError` เมื่อเซฟลงไดเรกทอรีใหม่
- ทำการ Serialize ออบเจกต์ Engine ทั้งหมดด้วย `joblib`
- คำนวณค่า Hash SHA-256 โดยอ่านไฟล์แบบสตรีมทีละบล็อกขนาด 4096 ไบต์ (4096-byte blocks)
- สร้างไฟล์ Sidecar `<basename>_metadata.json` บันทึก: `model_file`, `model_hash_sha256`, `scikit_learn_version`, `task`, `is_anomaly_model`, `imputed_columns`, `dropped_collinear_columns`, `zero_variance_columns`, `high_cardinality_columns`, `optimal_threshold`, `target_classes`, `best_estimator`, `features_used`, `potatopt_version`, `saved_at_utc`, `trained_at_utc`, `train_data_sha256`, `n_train_rows`, `automl_metric`, `validation_loss`, `library_versions`, `drift_profile_features`, `train_data_quality`

#### `PotatOptEngine.load(filepath, enforce_security=True)` -> `classmethod`
โหลดโมเดลกลับมาใช้งาน พร้อมระบบตรวจสอบความถูกต้องและความปลอดภัยของไฟล์

**เกณฑ์เฉพาะที่ใช้ (Specific Rules):**
- เมื่อตั้ง `enforce_security=True`: จะ Raise `FileNotFoundError` หากไฟล์ Metadata สูญหาย, Raise `ValueError` หากใน Metadata ไม่มี `model_hash_sha256`, และ Raise `RuntimeError` หากค่า Hash ที่คำนวณใหม่ไม่ตรงกับที่บันทึกไว้
- เมื่อตั้ง `enforce_security=False`: จะข้ามขั้นตอนการตรวจสอบความปลอดภัยทั้งหมด

---

## กลไกภายในและระบบดักข้อมูลผิดปกติ (Internal Mechanics & Data Guardrails)

เมธอดในส่วนนี้เป็นไพรเวตเมธอด (Private Methods) ที่ผู้ใช้งานไม่ต้องเรียกใช้โดยตรง แต่เป็นหัวใจหลักในการป้องกันความผิดพลาด จัดการข้อมูล และรักษาเสถียรภาพของระบบในสภาวะการผลิตจริง

### 3.1 ค่าคงที่ระดับโมดูล (Module Constants)

| ค่าคงที่ (Constant) | ค่า | ความหมาย |
|---|---|---|
| `MIN_TRAIN_ROWS` | `10` | จำนวนแถวขั้นต่ำที่ยอมให้เทรน |
| `MISSING_SCHEMA_WARN_RATIO` | `0.5` | สัดส่วนคอลัมน์ที่หายไปก่อนจะเตือน schema mismatch ตอน Inference |
| `OUT_OF_BOUNDS_WARN_RATIO` | `0.10` | สัดส่วนค่าตัวเลขที่หลุดขอบเขต training bounds ก่อนจะเตือนตอน Inference |
| `PSI_MODERATE_SHIFT` | `0.10` | เกณฑ์ PSI ระดับการเปลี่ยนแปลงปานกลาง (Moderate Shift) |
| `PSI_MAJOR_SHIFT` | `0.25` | เกณฑ์ PSI ระดับการเปลี่ยนแปลงอย่างมีนัยสำคัญ (Major Shift / Drift Trigger) |
| `PSI_DEFAULT_BINS` | `10` | จำนวน Quantile Bins เริ่มต้นสำหรับการคำนวณ PSI |
| `DQS_WEIGHTS` | `{"completeness": 0.30, "consistency": 0.25, "validity": 0.20, "uniqueness": 0.15, "timeliness": 0.10}` | ค่าน้ำหนักของ 5 มิติในการคำนวณ Data Quality Score |
| `DQS_PRODUCTION_READY` | `85.0` | เกณฑ์คะแนน DQS ขั้นต่ำสำหรับสถานะ Production Ready |
| `DQS_USABLE` | `65.0` | เกณฑ์คะแนน DQS ขั้นต่ำสำหรับสถานะ Usable with Caveats |
| `SILENT_NULL_TOKENS` | `frozenset({"", "-", "--", "?", "n/a", "na", "n.a.", "null", "none", "nil", "nan", "missing", "unknown", "undefined", "#n/a", "#value!", "#div/0!"})` | เซ็ตของข้อความ Placeholder ที่มีความหมายว่าข้อมูลสูญหาย (17 tokens ตรวจจับแบบ case-insensitive หลังตัดช่องว่าง) |
| `NUMERIC_SENTINELS` | `(-999, -9999, -99999, 999999, -1e30, 1e30)` | ทูเพิลของค่าตัวเลข Sentinel ที่เซนเซอร์อุตสาหกรรม/PLC มักบันทึกเมื่อเกิด Fault (รายงานเท่านั้น ไม่แปลงค่า) |
| `MODIFIED_ZSCORE_THRESHOLD` | `3.5` | เกณฑ์ Iglewicz-Hoaglin Modified Z-score สำหรับตรวจจับ Outliers |
| `CAPABILITY_SIGMA_RATIO_LIMIT` | `1.20` | ด่านที่ 1 ของ `stable` ใน `calculate_capability()` — อัตราส่วน σ_overall/σ_within ที่เกินกว่านี้แปลว่าความแปรปรวนมาจาก*ระหว่าง*กลุ่มย่อย ไม่ใช่ภายในกลุ่ม |
| `CAPABILITY_OUTLIER_RATE_LIMIT` | `0.01` | ด่านที่ 2 — สัดส่วนจุดที่หลุด 3σ ซึ่งความบังเอิญอธิบายไม่ได้ (ตามธรรมชาติอยู่ที่ 0.27%) |
| `CAPABILITY_OUTLIER_ALPHA` | `0.05` | ด่านที่ 2 (เพิ่มใน v1.7.0) — บนอนุกรมสั้น เกณฑ์อัตราส่วนเสื่อมสภาพเป็น "มีจุดหลุดจุดเดียวก็พอ" (1/50 = 2%) จึงต้องทดสอบ*จำนวน*จุดกับ Binomial(n, 0.0027) ด้วย ที่ n=50 false alarm ลดจาก 17.4% เหลือ 4.6% และ n ≥ 100 ไม่ขยับเลย |
| `CAPABILITY_TREND_LAMBDA` | `0.10` | ด่านที่ 3 (เพิ่มใน v1.7.0) — ค่า λ ของ EWMA ที่ใช้จับ drift ค่ามาตรฐานสำหรับ shift ขนาดราว 1σ |
| `CAPABILITY_TREND_RATE_LIMIT` | `0.03` | ด่านที่ 3 — สัดส่วนจุดที่ EWMA หลุดลิมิตของตัวเอง ใช้เป็น**อัตราส่วน** ไม่ใช่ "เตือนอย่างน้อย 1 ครั้ง" เพราะแบบหลังฟันธงข้อมูลปกติ 66.1% ที่ n=1,000 |
| `CAPABILITY_BASELINE_INFLATION_K` | `2.0` | ตัวคูณขยาย σ ที่ใช้*ทดสอบ*ความนิ่ง เมื่อ σ มาจากหน้าต่าง `baseline_n` เพียง N จุด (ขยายเป็น $1 + 2.0/\sqrt{N}$) — **ไม่แตะ `sigma_within` ที่รายงาน จึงไม่มีดัชนี Cp/Cpk ตัวไหนขยับ** ที่ N=40 false alarm ลดจาก 24.8% เหลือ 2.2% |
| `CAPABILITY_TREND_DRIFT_SIGMAS` | `0.75` | ด่านที่ 4 (เพิ่มใน v1.7.0) — เส้นตรงกำลังสองต้องขยับเกิน 0.75σ ตลอดทั้งอนุกรม EWMA มองไม่เห็น drift 0.5σ ไม่ว่าปรับ λ ยังไง เพราะมันไม่เคยออกนอก ±0.25σ ขณะที่ลิมิตอยู่ที่ 0.688σ |
| `CAPABILITY_TREND_ALPHA` | `0.01` | ด่านที่ 4 — ความชันต้องมีนัยสำคัญด้วย **ต้องผ่านทั้งขนาดและนัยสำคัญ** เพราะอนุกรมยาวพอจะทำให้ความชันที่ไม่มีความหมายกลายเป็นมีนัยสำคัญ |

### 3.2 `_validate_fit_inputs(X_train, y_train)` — ด่านตรวจก่อนเทรน
ฟังก์ชันนี้ทำงานที่จุดเริ่มต้นที่สุดของ `fit()` และจะหยุดการทำงานทันที (Raise / Fail-Fast) เนื่องจากข้อมูลเทรนที่ไม่ถูกต้องจะต้องไม่ถูกปล่อยให้สร้างโมเดลที่ผิดพลาดโดยไม่รู้ตัว โดยตรวจสอบตามลำดับดังนี้:
1. `X_train` หรือ `y_train` เป็น `None` -> Raise `ValueError("Training data cannot be None.")`
2. ไม่มีแถวข้อมูลเลย (`len == 0`) -> Raise `ValueError`
3. ไม่มีคอลัมน์ฟีเจอร์ (`shape[1] == 0`) -> Raise `ValueError`
4. แถวข้อมูลน้อยกว่า `MIN_TRAIN_ROWS` (10) -> Raise `ValueError` พร้อมระบุจำนวนแถวจริงที่มี
5. ตรวจพบชื่อคอลัมน์ซ้ำกัน -> บันทึกคำเตือนผ่าน `logger.warning` จะไม่ Raise (เนื่องจาก Pipeline จะจัดการชื่อซ้ำด้วย Collision-proof sanitizer ใน Stage 2)
6. ตรวจพบแถวข้อมูลซ้ำกัน (Duplicate rows) -> บันทึกคำเตือนผ่าน `logger.warning` พร้อมรายงานจำนวนและเปอร์เซ็นต์ เนื่องจากข้อมูลที่ซ้ำกันอาจหลุดเข้าไปอยู่ใน Cross-Validation folds ทั้งสองฝั่งและทำให้ค่าคะแนนโมเดลบวมเกินจริง

### 3.3 `_preprocess_fit_transform(X, y)` — Pipeline 11 ขั้นตอน

```
Raw Sensor Data 
   │
   ├── [Stage 1]  Drop Nested Dicts/Lists
   ├── [Stage 2]  Sanitize Column Names (Collision-Proof & Special Characters)
   ├── [Stage 3]  Drop All-NaN Columns (รวมคอลัมน์ที่เป็น Silent Nulls ทั้งหมด)
   ├── [Stage 4]  Extract Temporal Features from Datetime Strings (UTC)
   ├── [Stage 5]  Auto-Convert Numeric Strings to Continuous Dtypes
   ├── [Stage 6]  Drop High-Cardinality Identifiers (UUIDs, Serial Nos)
   ├── [Stage 7]  Lossless Memory Downcasting & Median/Mode Imputation
   ├── [Stage 8]  Drop Zero-Variance Constant Features
   ├── [Stage 9]  LowSpecML Ordinal Encoding (-1 Unseen Label Safe)
   ├── [Stage 10] Target-Aware Multicollinearity Pruning (Collinear Drop)
   └── [Stage 11] Numerical Feature Scaling (Standard / MinMax)
   │
   ▼
[Guardrails Validation] ──(Defects < 5 samples)──► [Unsupervised Isolation Forest]
   │
   └──(Normal Data)──► [FLAML AutoML Engine] (ค่าเริ่มต้น: lgbm, xgboost, rf)
```

| Stage | หน้าที่ | เกณฑ์เฉพาะ |
|---|---|---|
| 1. Drop Nested Structures | ทิ้งคอลัมน์ที่มีค่าเป็น `list` หรือ `dict` | ตรวจด้วย `isinstance(x, (list, dict))` แม้เจอแค่แถวเดียวก็ตัดทั้งคอลัมน์ |
| 2. Sanitize Column Names | แทนอักขระพิเศษด้วย `_` | regex `[^a-zA-Z0-9_\u0E00-\u0E7F]+` — เก็บภาษาไทย (ช่วง U+0E00-U+0E7F) และตัวเลข/ตัวอักษรไว้ ป้องกัน LightGBM JSON error; ชื่อว่างจะกลายเป็น `feature`; ชื่อซ้ำจะต่อท้าย `_1`, `_2` แบบวนจนไม่ชน |
| 3. Drop All-NaN Columns | ตัดคอลัมน์ที่ว่างทั้งหมด | แปลง `SILENT_NULL_TOKENS` เป็น NaN ก่อน แล้วตัดคอลัมน์ที่เป็น NaN ทั้งหมดทิ้ง พร้อมเก็บรายชื่อไว้ใน `self.all_nan_cols` |
| 4. Extract Temporal Features | แตก datetime เป็น 5 ฟีเจอร์ | สร้าง `_year`, `_month`, `_day`, `_hour`, `_dayofweek`; ตรวจจับ string ที่ตรง regex `^\d{4}[-/]\d{2}[-/]\d{2}` จาก 10 แถวแรก แล้วต้องแปลงสำเร็จเกิน 80% ของข้อมูล; ใช้ `utc=True` รองรับ timezone ผสม |
| 5. Auto-Convert Numeric Strings | แปลง object ที่จริงๆ เป็นตัวเลข | ต้องแปลงสำเร็จเกิน 80% **และ** มีค่าไม่ซ้ำมากกว่า 5 ค่า (กันการแปลงรหัสหมวดหมู่เป็นตัวเลขโดยไม่ตั้งใจ) |
| 6. Drop High-Cardinality IDs | ตัด UUID / Serial / Running number | ฝั่ง string: unique > 50% ของจำนวนแถว และมีมากกว่า 30 แถว หรือ unique > 1000; ฝั่งตัวเลข: unique เท่ากับจำนวนแถวพอดี และชื่อคอลัมน์ตรง regex `^(id\|index\|serial\|uuid\|guid\|code\|trans_id\|row_id\|run_id\|seq)$` หรือลงท้าย `_id`, `_no`, `_num`, `_idx` |
| 7. Imputation + Memory Downcast | เติมค่าว่าง แล้วบีบ RAM | ตัวเลขใช้ median, หมวดหมู่ใช้ mode (ถ้าไม่มี mode ใช้ `"Unknown"`, ถ้ายังเป็น NaN ใช้ `0`); downcast เป็น int8/int16/int32 เฉพาะเมื่อไม่มี null และ float64 -> float32 เมื่อค่าอยู่ในช่วง float32 — เป็น lossless |
| 8. Drop Zero-Variance | ตัดคอลัมน์ค่าคงที่ | `nunique() <= 1`; บันทึกขอบเขต min/max ของทุกคอลัมน์ตัวเลขลง `self.feature_bounds` ตรงจุดนี้ (ก่อน scale) เพื่อใช้ clip ตอน inference |
| 9. Ordinal Encoding | แปลงหมวดหมู่เป็นตัวเลข | `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` — ค่าที่ไม่เคยเห็นตอนเทรนจะกลายเป็น `-1`; `-1` ถูกใส่เข้าไปใน categories ล่วงหน้าเพื่อกัน NaN ตอน LightGBM inference |
| 10. Collinearity Pruning | ตัดฟีเจอร์ที่ซ้ำซ้อนกัน | ทำงานเมื่อ `collinear_threshold < 1.0` และจำนวนคอลัมน์ตัวเลขอยู่ระหว่าง 1-1000 (กัน RAM ระเบิดจาก correlation matrix ขนาดใหญ่); โหมด Target-Aware (ใช้กับ regression และ binary classification) จะเก็บตัวที่ correlate กับ target มากกว่าไว้ ทิ้งอีกตัว; โหมดทั่วไปใช้ upper-triangle ตัดตัวหลัง |
| 11. Feature Scaling | ปรับสเกลตัวเลข | `'standard'` -> StandardScaler, `'minmax'` -> MinMaxScaler; ปิดท้ายด้วยการล็อก `self.column_dtypes` และ `self.feature_names` ไว้บังคับใช้ตอน inference |

### 3.4 `_preprocess_transform(X)` — การบังคับใช้ schema ตอน Inference
เมธอดนี้จะ **ไม่เรียนรู้ค่าสถิติใหม่ใด ๆ ทั้งสิ้น** แต่จะนำค่าสถิติและกฎที่เรียนรู้ไว้แล้วจากขั้นตอน `fit()` มาบังคับใช้ซ้ำ เพื่อให้กระบวนการทำนายมีความแม่นยำและปลอดภัยสูงสุด:
- รับ input ได้หลากหลายรูปแบบ: sparse (`.toarray()`), dict/list (`pd.json_normalize`), Series, ndarray, DataFrame
- แปลง `SILENT_NULL_TOKENS` ในคอลัมน์ Object ให้เป็น `NaN` ก่อนทำการ Impute ด้วยค่าสถิติที่เรียนรู้จากชุดเทรน
- คอลัมน์ที่คาดหวังแต่ไม่มีใน input จะถูกเติมเป็น NaN ก่อน แล้วค่อยถูก impute ด้วยค่าที่เรียนรู้จาก train
- ค่าตัวเลขที่หลุดขอบเขต train จะถูก **clip** ให้อยู่ในช่วง `self.feature_bounds` เมื่อ `apply_bounds_clip=True` (ไม่ตัดทิ้งทั้งแถว)
- ข้อมูลหมวดหมู่ที่ไม่เคยเห็นตอนเทรน -> แปลงเป็น `-1` อย่างปลอดภัย ไม่เกิดข้อผิดพลาด crash
- บังคับประเภทข้อมูลตาม `self.column_dtypes` และบังคับ **ลำดับคอลัมน์** ตาม `self.feature_names` อย่างแม่นยำ ป้องกันปัญหา Tree Split เลื่อนตำแหน่ง
- คอลัมน์ที่ยังขาดหลังผ่านทุกขั้นตอนจะถูกเติมด้วย `0`

### 3.5 `_check_input_anomalies(X_proc_raw)` — ตัวเฝ้าระวังตอน Inference
เมธอดนี้ทำหน้าที่เป็นตัวเฝ้าระวัง (Observer) ซึ่งแตกต่างจาก `_validate_fit_inputs` ตรงที่ **จะไม่ Raise ข้อผิดพลาดหรือหยุดการทำงานโดยเด็ดขาด** เนื่องจากในสภาพแวดล้อม Production สายการผลิตจำเป็นต้องดำเนินต่อไปได้อย่างต่อเนื่อง การที่ระบบหยุดทำงาน (Crash) เพียงเพราะเซนเซอร์ตัวเดียวมีค่าเพี้ยน ย่อมสร้างความเสียหายรุนแรงกว่าการทำนายผลพร้อมส่งสัญญาณเตือน โดยมีกลไกเตือน 2 ระดับ:
- **เตือนที่ 1 — Schema Mismatch:** เมื่อคอลัมน์ที่คาดหวังหายไปเกิน `MISSING_SCHEMA_WARN_RATIO` (50%) จะรายงานจำนวนคอลัมน์ที่หายไป/ทั้งหมด/เปอร์เซ็นต์
- **เตือนที่ 2 — Out-of-bounds:** เมื่อค่าตัวเลขหลุดช่วง `feature_bounds` ที่เคยเรียนรู้มาเกิน `OUT_OF_BOUNDS_WARN_RATIO` (10%) จะรายงานเปอร์เซ็นต์และ 3 คอลัมน์ที่หลุดขอบเขตมากที่สุด เพื่อส่งสัญญาณว่าเซนเซอร์อาจชำรุดหรือเกิด Data Drift
- คำเตือนทั้งหมดจะถูกบันทึกไว้ใน `self.last_predict_warnings` (list) และแสดงผลผ่าน Named Logger `logger.warning` ด้วย prefix `[WARNING]` โดย attribute นี้จะถูกล้างใหม่ทุกครั้งที่มีการเรียก Inference จึงสะท้อนผลเฉพาะรอบล่าสุด และสามารถดึงไปแสดงผลบน Dashboard หน้างานได้ทันที

```python
y_pred = engine.predict(new_batch_df)

# ตรวจคำเตือนของรอบที่เพิ่งทำนายไป
if engine.last_predict_warnings:
    for w in engine.last_predict_warnings:
        print("ตรวจพบความผิดปกติของ input:", w)
```

### 3.6 ตารางสรุปพฤติกรรมเมื่อเจอข้อมูลผิดปกติ (Guardrail Summary)

| สถานการณ์ | พฤติกรรมของระบบ | ผลลัพธ์ |
|---|---|---|
| `X_train` หรือ `y_train` เป็น None | `_validate_fit_inputs` | ValueError (หยุดทันที) |
| ข้อมูลเทรนน้อยกว่า 10 แถว | `_validate_fit_inputs` | ValueError (หยุดทันที) |
| ไม่มีคอลัมน์ฟีเจอร์ | `_validate_fit_inputs` | ValueError (หยุดทันที) |
| ข้อมูลเทรนมีชื่อคอลัมน์ซ้ำกัน | `_validate_fit_inputs` + Stage 2 | เตือนผ่าน `logger.warning` และเปลี่ยนชื่อเป็น `_1`, `_2` แบบ collision-proof ไม่ให้ชนกัน |
| ข้อมูลเทรนมีแถวซ้ำกัน (Duplicate rows) | `_validate_fit_inputs` | เตือนผ่าน `logger.warning` พร้อมระบุจำนวนและ % เพื่อป้องกัน CV leakage |
| ทุกคอลัมน์ถูกตัดทิ้งตอน preprocessing | Empty-feature guard ใน `fit()` | ValueError พร้อมบอกว่าคอลัมน์หายไปเพราะสาเหตุใด |
| จำนวนแถว X และ y ไม่เท่ากัน | Length check ใน `fit()` | ValueError |
| Target มีคลาสเดียว | Class check ใน `fit()` | ValueError |
| Target เป็น NaN บางแถว | Auto-drop ใน `fit()` | ทิ้งเฉพาะแถวนั้น ทำงานต่อ |
| ของเสียน้อยกว่า 5 ชิ้น (binary) | Imbalance guardrail | สลับเป็น IsolationForest อัตโนมัติ |
| Multiclass ที่มีคลาสถึงเกณฑ์ (>= 5 ตัวอย่าง) ไม่ถึง 2 คลาส | Scarcity guardrail | สลับเป็น IsolationForest อัตโนมัติ |
| Multiclass ที่มีบางคลาสน้อยกว่า 5 ตัวอย่าง | Rare-class dropping | ทิ้งเฉพาะคลาสหายาก เตือน แล้วเทรนต่อ |
| ข้อความ Placeholder ในคอลัมน์หมวดหมู่/สตริง (เช่น `"N/A"`, `"-"`) | `_convert_silent_nulls` | แปลงเป็น NaN อัตโนมัติทั้งตอน fit และ inference ก่อน Imputation พร้อมบันทึก log |
| คอลัมน์ที่มีเฉพาะข้อความ Placeholder ทั้งหมด | Stage 3 ใน Preprocessing | แปลงเป็น NaN ทั้งหมดแล้วถูกตัดทิ้งเป็น All-NaN column ป้องกันการมองเป็นค่าคงที่ |
| ค่าตัวเลข Numeric Sentinels (-999, 999999 ฯลฯ) | `audit_data_quality` / `detect_silent_nulls` | รายงานใน Data Quality Audit เท่านั้น ไม่แปลงค่าอัตโนมัติเพื่อป้องกันการกระทบข้อมูลจริง |
| ค่า `inf` / `-inf` ในข้อมูล | แปลงเป็น NaN | ถูก impute ต่อ ไม่ crash |
| คอลัมน์หายตอน predict | เติม NaN + impute | ทำนายต่อได้ พร้อมเตือนถ้าหายเกิน 50% |
| ค่าเซนเซอร์หลุดช่วงตอน predict | Clip + `_check_input_anomalies` | ทำนายต่อได้ พร้อมเตือนถ้าเกิน 10% |
| หมวดหมู่ที่ไม่เคยเห็นตอน predict | Ordinal `unknown_value=-1` | ทำนายต่อได้ ไม่ crash |
| ข้อมูล Batch มีการเคลื่อนตัวของการกระจายตัว (Major / Moderate Drift) | `detect_drift` / `check_data_drift` | ตรวจจับผ่าน PSI และ Std Ratio แจ้งเตือน Warning/Info พร้อมให้คำแนะนำ Recommendation |
| ชุดทดสอบมีเพียงคลาสเดียว (Single-class test set) | `_compute_auc_metrics` ใน `evaluate()` | คำนวณเมทริกซ์อื่นได้ปกติ โดย `roc_auc`, `pr_auc`, `defect_base_rate` คืนค่า `None` ไม่ crash |
| ไฟล์โมเดลถูกดัดแปลง | SHA-256 verification ใน `load()` | RuntimeError `[SECURITY ALERT]` |
| ไฟล์ metadata หายไป | Security check ใน `load()` | FileNotFoundError `[SECURITY ALERT]` |

---

## ระบบคุณภาพข้อมูลและการเฝ้าระวังหลัง Deploy (Data Quality & Monitoring)

### 4.1 Data Quality Score

การเทรนโมเดลบนชุดข้อมูลที่คุณภาพต่ำจะทำให้ได้ตัวเลขผลลัพธ์ที่ดูดีแต่ไม่สามารถนำไปใช้งานจริงได้ในโรงงาน ฟังก์ชัน `audit_data_quality(df, target_col=None)` จึงทำหน้าที่เป็น Quality Gate ตรวจสอบสุขภาพข้อมูล 5 มิติถ่วงน้ำหนักตามหลักวิศวกรรมข้อมูล:

| มิติคุณภาพ (Dimension) | ค่าน้ำหนัก (Weight) | สูตรการคำนวณ (Formula) | คำอธิบาย |
|---|---|---|---|
| **Completeness** | 30% (`0.30`) | $100 \times \left(1 - \frac{\text{explicit nulls} + \text{silent nulls}}{\text{total cells}}\right)$ | สัดส่วนความสมบูรณ์ของข้อมูล โดยรวมทั้ง NaN ชัดเจนและ Silent Nulls |
| **Consistency** | 25% (`0.25`) | $100 \times \left(1 - \frac{\text{mixed-type columns}}{\text{total columns}}\right)$ | ความสม่ำเสมอของประเภทข้อมูล คอลัมน์ object ที่มีค่าตัวเลขปนอยู่ระหว่าง 0% ถึง 100% จะถูกนับเป็น Mixed-type |
| **Validity** | 20% (`0.20`) | $100 \times \left(1 - \frac{\text{flagged outlier cells}}{\text{numeric cells}}\right)$ | ความสมเหตุสมผลทางสถิติ สัดส่วนเซลล์ตัวเลขที่ไม่ถูกตรวจจับเป็น Outliers (Modified Z-score) |
| **Uniqueness** | 15% (`0.15`) | $100 \times \left(1 - \frac{\text{duplicate rows}}{\text{total rows}}\right)$ | ความไม่ซ้ำซ้อนของแถวข้อมูลและชื่อคอลัมน์ |
| **Timeliness** | 10% (`0.10`) | $100 \times \left(1 - \frac{\text{implausible timestamps}}{\text{parsed timestamps}}\right)$ | ความถูกต้องของ Timestamp (ไม่เป็นเวลาในอนาคตและไม่เกิดขึ้นก่อน 1970-01-01) |

> **หมายเหตุเรื่อง Timeliness:** หากชุดข้อมูลไม่มีคอลัมน์ Datetime มิติ Timeliness จะถูกตัดออก (`None`) และค่าน้ำหนักของ 4 มิติที่เหลือจะถูก Renormalise ให้มีผลรวมเท่ากับ 1.0 (100%) โดยอัตโนมัติ

#### ระดับการตัดเกรด (Grade Bands)
- **$\ge 85.0$ (`production_ready`):** ข้อมูลมีคุณภาพสูง พร้อมสำหรับการเทรนและนำขึ้น Production ทันที
- **$\ge 65.0$ (`usable_with_caveats`):** ข้อมูลสามารถใช้งานได้ แต่ต้องบันทึกข้อจำกัด (Caveats) ตามประเด็นปัญหาที่พบ
- **$< 65.0$ (`remediation_required`):** ข้อมูลมีปัญหาคุณภาพวิกฤต ต้องดำเนินการปรับปรุงตาม Remediation Plan ก่อนนำไปเทรน

ระบบจะบันทึกผล Verdict ผ่าน `logger.warning` เมื่อเกรดเป็น `remediation_required` และบันทึกผ่าน `logger.info` ในกรณีอื่น ๆ โดยรายการ Issues จะถูกเรียงลำดับตามความรุนแรง (critical > high > medium > low) และสรุปเป็น Action strings ใน `remediation` สูงสุดไม่เกิน 10 รายการ

#### Missing-Value Playbook
ตารางแนวทางปฏิบัติการจัดการค่าสูญหายรายคอลัมน์ตามระดับสัดส่วนการหาย:

| สัดส่วนค่าสูญหาย | ระดับความรุนแรง (Severity) | แนวทางปฏิบัติ (Action) |
|---|---|---|
| $< 1\%$ | Low | ตัดแถวที่ได้รับผลกระทบทิ้ง หรือ Impute ด้วยค่า Median / Mode |
| $1 - 10\%$ | Medium | Impute ค่า และสร้างคอลัมน์ Indicator `<col>_was_null` เพื่อบอกโมเดล |
| $10 - 30\%$ | High | Impute อย่างระมัดระวัง และสืบหาสาเหตุต้นน้ำในกระบวนการเก็บข้อมูล |
| $> 30\%$ | Critical | ห้าม Impute โดยไม่ตรวจสอบ; ส่งให้ Domain Expert ทบทวนหรือตัดคอลัมน์ทิ้ง |

*(หากคอลัมน์ Target มีค่าสูญหาย จะถูกจัดเป็นระดับ **Critical** เสมอ และต้องตัดแถวนั้นทิ้งเนื่องจากไม่สามารถใช้เทรนแบบ Supervised ได้)*

```python
from potatopt import audit_data_quality

# ตรวจสอบคุณภาพข้อมูลก่อนเทรน
audit = audit_data_quality(df, target_col="Defect_Status")
print(f"DQS: {audit['dqs']} ({audit['grade']})")
print("สรุปประเด็นที่ต้องแก้ไข (Remediation Plan):")
for action in audit["remediation"]:
    print(" -", action)
```

### 4.2 Silent Nulls

ในสภาพแวดล้อมอุตสาหกรรม ข้อมูลสูญหายมักไม่ได้ถูกบันทึกเป็นค่า `NaN` หรือ `None` แต่แฝงมาในรูปแบบข้อความ Placeholder หรือตัวเลข Sentinel:

1. **Placeholder Strings:** ข้อความหลอกที่ระบบหรือผู้ปฏิบัติงานกรอก เช่น `""`, `"-"`, `"--"`, `"?"`, `"n/a"`, `"na"`, `"n.a."`, `"null"`, `"none"`, `"nil"`, `"nan"`, `"missing"`, `"unknown"`, `"undefined"`, `"#n/a"`, `"#value!"`, `"#div/0!"` (ตรงตามชุด `SILENT_NULL_TOKENS` ตรวจสอบแบบ case-insensitive หลังตัด whitespace)
   - หากปล่อยไว้ ข้อความเหล่านี้จะถูก Ordinal Encoding แปลงเป็นตัวเลขหมวดหมู่ปกติ ทำให้โมเดลเรียนรู้จากข้อมูลที่แท้จริงแล้วว่างเปล่า
   - PotatOpt จะแปลง Placeholder Strings ในคอลัมน์ Object ให้เป็น `NaN` โดยอัตโนมัติทั้งในขั้นตอน `fit()` และขั้นตอน Inference ก่อนกระบวนการ Imputation (ชุด Token คงที่ ไม่มีการเรียนรู้ จึงไม่เกิด Data Leakage) พร้อมบันทึกจำนวนที่แปลงผ่าน Log
2. **Numeric Sentinels:** ตัวเลขที่ PLC หรือเซนเซอร์เขียนขึ้นเมื่อเกิดสภาวะ Fault หรือการสื่อสารล้มเหลว ได้แก่ `-999`, `-9999`, `-99999`, `999999`, `-1e30`, `1e30` (ตรงตาม `NUMERIC_SENTINELS`)
   - **หลักการออกแบบ:** ตัวเลข Sentinel จะถูก**รายงานเท่านั้นใน Audit และไม่ถูกแปลงค่าอัตโนมัติ** เนื่องจากในกระบวนการผลิตจริง ค่าอุณหภูมิหรือแรงดันบางจุดอาจมีค่าเป็น -999 ได้อย่างถูกต้องตามธรรมชาติ ซึ่งต้องอาศัย Domain Input ในการตัดสินใจ
3. **การปิดการทำงาน (Opt-Out):** หากในกระบวนการของคุณมีข้อความอย่าง `"NA"` หรือ `"-"` ที่เป็นรหัสหมวดหมู่จริง สามารถปิดระบบแปลงค่าได้โดยกำหนด `handle_silent_nulls=False` ในคอนสตรัคเตอร์ของ `PotatOptEngine`

### 4.3 PSI Drift Detection

**Population Stability Index (PSI)** เป็นสถิติมาตรฐานทางอุตสาหกรรมสำหรับการตรวจจับ Covariate Shift โดยวัดการเปลี่ยนแปลงของการกระจายตัวของข้อมูลในทุก ๆ ช่วงของตัวแปร:

| ค่า PSI | ระดับความรุนแรง (Severity) | ความหมายและคำแนะนำ |
|---|---|---|
| $< 0.10$ | Stable | การกระจายตัวของข้อมูลคงที่และมีเสถียรภาพ โมเดลทำงานได้อย่างมั่นใจ |
| $0.10 - 0.25$ | Moderate Shift | เกิดการเปลี่ยนแปลงปานกลาง ควรเพิ่มความถี่ในการติดตามและวางแผนทบทวนการเทรนใหม่ |
| $> 0.25$ | Major Shift | เกิดการเปลี่ยนแปลงอย่างมีนัยสำคัญ ควรสั่ง Re-train โมเดลก่อนนำผลทำนายไปใช้ต่อ |

#### ทำไม PSI และ Std Ratio จึงตรวจจับ Tool Wear ได้ดีกว่า Mean Shift?
เมื่อเครื่องมือตัดหรือแม่พิมพ์ในโรงงานเริ่มสึกหรอแบบก้าวหน้า (Progressive Tool Wear) ขนาดชิ้นงานเฉลี่ยอาจยังคงอยู่ตรงกลางเท่าเดิม ($\Delta \text{Mean} \approx 0$) แต่ความแปรปรวนของการกระจายตัวจะบานกว้างขึ้นอย่างเห็นได้ชัด การเปรียบเทียบเฉพาะค่าเฉลี่ยแบบเดิมจึงตรวจไม่พบความผิดปกติ แต่ PSI และ `std_ratio` (`batch_std / train_std`) จะแจ้งเตือนทันที

#### การบายพาส Bounds Clipping ในเส้นทางเฝ้าระวัง
ในขั้นตอนการทำนายผล (`predict`) ข้อมูลนำเข้าจะถูก **Clip** ให้อยู่ในช่วง `feature_bounds` ของชุดเทรนเสมอเพื่อความปลอดภัยของโมเดล แต่ในเส้นทางการเฝ้าระวัง `detect_drift(X_batch)` ระบบจะเรียกใช้ Preprocessing โดยกำหนด `apply_bounds_clip=False` โดยเจตนา เพราะหากทำการ Clip ข้อมูล ค่าที่บานออกนอกขอบเขตจะถูกตัดกลับเข้ามาในกรอบและกลบสัญญาณ Drift ที่แท้จริง
> **ผลการทดสอบจริง:** ในชุดข้อมูลทดสอบที่สร้างให้มีความแปรปรวนกว้างขึ้น 4 เท่าของชุดเทรน เมื่อผ่านการ Clip จะรายงาน `std_ratio` เพียง **2.33** แต่เมื่อบายพาส Bounds Clipping จะรายงาน `std_ratio` สูงถึง **3.94** ซึ่งสะท้อนความจริงหน้างานได้อย่างถูกต้อง

ข้อมูลสถิติจะถูกรายงานทั้งในหน่วยวิศวกรรมดั้งเดิมของ Operator (`*_raw`) และในปริภูมิ Scaler Space (`*_scaled`) โดยมีคำแนะนำ Action Logged ที่ระดับ `WARNING` เมื่อพบ Major Drift และ `INFO` เมื่อปกติ

```python
# เฝ้าระวังความเสถียรของโมเดลหลัง Deploy (ไม่จำเป็นต้องมีชุดข้อมูล Train เดิม)
drift_report = engine.detect_drift(live_batch_df)

if drift_report["drift_detected"]:
    print(f"เตือนภัย Data Drift สูงสุด (Max PSI = {drift_report['max_psi']:.4f})")
    print("คำแนะนำ:", drift_report["recommendation"])
    for feat, stats in drift_report["features"].items():
        if stats["severity"] == "major":
            print(f" - ฟีเจอร์ {feat}: PSI={stats['psi']:.4f}, Std Ratio={stats['std_ratio']:.2f}x")
```

### 4.4 Audit Trail & Logging

เพื่อตอบสนองข้อกำหนดการสอบย้อนกลับ (Traceability) ตามมาตรฐาน **ISO 9001**:
1. **Named Logger:** ข้อความแจ้งเตือนทั้งหมดใน PotatOpt ถูกส่งผ่าน Named Logger `logging.getLogger("potatopt")` ที่ระดับ `INFO` พร้อม StreamHandler รูปแบบ `[%(levelname)s] %(message)s` โดยไม่มีคำสั่ง `print()` ตกค้างในโค้ด ทำให้ผู้ดูแลระบบสามารถปรับแต่งและจัดการ Log Stream ได้อย่างอิสระ
2. **File-Based Audit Logging (`enable_audit_log`):** สามารถเปิดการบันทึก Log ลงไฟล์ได้ด้วยคำสั่งเดียว:
   ```python
   import potatopt as po
   po.enable_audit_log("potatopt_audit.log", level=logging.INFO)
   ```
   ระบบจะผูก FileHandler เข้ากับ Logger ด้วยรูปแบบบันทึกเวลา `"%(asctime)s | %(levelname)s | %(message)s"` โดยฟังก์ชันนี้มีคุณสมบัติ Idempotent (เรียกซ้ำด้วยพาธเดิมได้โดยไม่เกิด Handler ซ้ำซ้อน) ช่วยให้วิศวกรโรงงานสามารถสอบย้อนหลังได้ว่า Batch การผลิตใดทำให้เกิด Warning และมีสถานะ Data Drift อย่างไร

---

## การผสานกรอบคิดวิศวกรรมอุตสาหการ (IE Integration)

PotatOpt ออกแบบมาเพื่อเป็นเครื่องมือสนับสนุนวงจร **DMAIC**:
1. **Define & Measure:** ตรวจวัดสุขภาพข้อมูล (`inspect_data`, `audit_data_quality`) และสร้างฐานการวัดกระบวนการด้วย SPC Chart (`calculate_spc_limits`)
2. **Analyze:** วิเคราะห์ปัจจัยสำคัญที่มีผลต่อคุณภาพชิ้นงานด้วย Tri-layer SHAP (`get_shap_values`) และ Feature Importance เพื่อเชื่อมโยงกับผังก้างปลา (Ishikawa/Fishbone Diagram)
3. **Improve:** ปรับปรุงจุดตัดการตัดสินใจด้วย Cost of Quality (`optimize_threshold`) ลดของเสียและลดต้นทุนการตรวจสอบซ้ำซ้อน
4. **Control:** ติดตามความเสถียรของโมเดลหน้างานด้วยการตรวจจับ Data Drift (`detect_drift`, `check_data_drift`) และระบบล็อกโมเดล ISO 9001

---

## ระบบความปลอดภัย ISO 9001 (Model Integrity & Security)

เมื่อสั่ง `.save("model.pkl")` ระบบจะสร้างไฟล์คู่ขนาน `model_metadata.json` ที่บันทึก:
- ค่าแฮช `model_hash_sha256` ของไฟล์โมเดล
- ค่าแฮช `train_data_sha256` ของชุดข้อมูลฝึกสอน
- ประทับเวลา UTC `trained_at_utc` และ `saved_at_utc`
- รุ่นของไลบรารีทุกตัวใน Runtime Stack (`library_versions`)
- รายการคอลัมน์ที่ถูกคัดกรอง, ค่า Imputation, ข้อมูล Profile สำหรับตรวจวัด Drift และผลการประเมิน Data Quality

เมื่อเรียกใช้งาน `.load(filepath, enforce_security=True)`:
- หากไฟล์ `.pkl` ถูกแก้ไข เสียหาย หรือไฟล์ Metadata สูญหาย ระบบจะแจ้งเตือน `RuntimeError: [SECURITY ALERT]` และหยุดการทำงานทันที

**สิ่งที่ตรวจสอบนี้พิสูจน์ได้ และสิ่งที่พิสูจน์ไม่ได้ (v1.4.0):** ค่า SHA-256 ข้างต้นพิสูจน์ได้แค่
**Integrity** — ไบต์ในไฟล์ไม่เปลี่ยนไปจากตอนที่ Save — เท่านั้น **ไม่ใช่ Authenticity** (ไม่สามารถพิสูจน์ได้ว่าใครเป็นผู้สร้างไฟล์)
เพราะค่าแฮชที่ใช้เทียบถูกเก็บเป็น Plaintext อยู่ในไฟล์ Metadata ข้าง ๆ กัน ไม่มี Key หรือลายเซ็นใด ๆ
ผู้ที่เขียนทับ `model.pkl` ได้ ย่อมเขียนทับ `model_metadata.json` และคำนวณแฮชใหม่ให้ตรงกันได้เช่นกัน และเนื่องจาก
`joblib.load()` คือ pickle การโหลดไฟล์จะรันโค้ดที่อยู่ในไฟล์นั้นทันที ไม่ว่าแฮชจะตรงหรือไม่ ดังนั้นกฎที่ต้องยึดคือ
**โหลดเฉพาะไฟล์ที่ตัวเองหรือ Pipeline ของตัวเอง Save ไว้เท่านั้น** การตรวจนี้มีไว้จับไฟล์เสียหาย ไฟล์ถูกตัดตอน
หรือ Deploy ไฟล์ผิดรุ่น ซึ่งเกิดขึ้นบ่อยกว่าการถูกโจมตีมาก ไม่ใช่ใบรับรองว่าไฟล์นั้นปลอดภัยที่จะโหลด

---

## โครงสร้างโปรเจกต์ (Project Structure)

```
├── potatopt/             # ตัวไลบรารี (แตกจากไฟล์เดียวเป็น Package แล้ว)
│   ├── __init__.py       # หน้าบ้านสาธารณะ: re-export, __all__, __version__
│   ├── mcp_server.py     # MCP Server (stdio) ให้ AI ขับได้โดยไม่ต้องเขียน Python
│   ├── engine.py         # PotatOptEngine
│   ├── analysis.py       # auto_analyze, run_seed_sweep
│   ├── data.py           # Profiling / Split / Data Quality
│   ├── spc.py            # Control Chart (SPC, EWMA, CUSUM) + กฎ Nelson/Western Electric
│   ├── quality.py        # Cp/Cpk/Pp/Ppk และ Gauge R&R (MSA) แบบ ANOVA
│   ├── drift.py          # Drift และ PSI
│   ├── reliability.py    # MTBF / MTTR / Availability / OEE / Pareto
│   ├── calibration.py    # check_calibration
│   ├── constants.py      # ค่าคงที่สำหรับการจูน
│   ├── _lazy.py          # Logger + ตัวโหลด FLAML/SHAP แบบ Lazy
│   ├── _utils.py         # Helper ที่ใช้ร่วมกัน, to_jsonable, Audit Log
│   └── py.typed          # PEP 561 marker (บอกว่าแพ็กเกจนี้มี Type Hint ให้ใช้)
├── chart_engine.py       # วาดกราฟจาก dict ที่ potatopt คืนมา (ไม่คำนวณเอง)
├── examples/             # quickstart.py + ตัวโหลดชุดข้อมูล AI4I 2020
├── benchmarks/           # วัด "ค่าเขียน" (token) และ "ค่ารัน" (RAM/เวลา/เงิน)
├── scripts/              # verify_core_install.py พิสูจน์คำเคลม Core 4 แพ็กเกจ
├── tests/                # ชุดทดสอบอัตโนมัติ (422 เคส)
├── pyproject.toml        # Packaging + Optional Extras + การตั้งค่า ruff
├── requirements.txt      # Core & Ecosystem Dependencies (รวม Dev Tools)
├── README.md             # เอกสารฉบับภาษาอังกฤษ
├── LICENSE               # MIT License
└── Agent.md              # Engineering Roadmap & Architecture Progress Log
```

**หมายเหตุเรื่องการแตกเป็น Package:** โครงเดิมเป็นไฟล์เดียว 4,543 บรรทัด การแตกครั้งนี้
**ไม่เปลี่ยน API แม้แต่ชื่อเดียว** (`__all__` ยังเป็น 57 ชื่อชุดเดิมเรียงลำดับเดิม) และพิสูจน์
ด้วยการเทียบ AST ของทุกฟังก์ชันก่อน/หลังว่าเหมือนกันทั้ง 71 ตัว โมเดล `.pkl` ที่เซฟด้วย
เวอร์ชันก่อนแตกยังโหลดได้ปกติ เพราะ `__init__.py` re-export `PotatOptEngine` ไว้

### การรันชุดทดสอบ (Running the tests)

ทดสอบการทำงานของระบบทั้งหมด (422 tests) ด้วยคำสั่ง:
```bash
python -m pytest tests/ -q
```

ตรวจสอบความถูกต้องของสไตล์โค้ดและ Linting:
```bash
python -m ruff check potatopt chart_engine.py tests benchmarks scripts examples
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ต้นทุน Token (Token Cost)

เป้าหมายหลักข้อหนึ่งของไลบรารีนี้คือ **ให้ AI เรียกใช้ได้โดยเขียนโค้ดใหม่น้อยที่สุด** เพราะทุกบรรทัดที่โมเดลภาษาต้องเขียนเองคือทั้งต้นทุน Token และโอกาสผิดพลาด

รันวัดเองได้:

```bash
python benchmarks/token_cost.py
```

สคริปต์เทียบโค้ดสามแบบที่ทำงานเดียวกัน (เทรนโมเดลทำนายเครื่องจักรเสีย + จูน Threshold ตามต้นทุน + รายงานผลเป็นเงิน) และใช้ตัวนับ Token จริงจาก `tiktoken` ถ้าติดตั้งไว้ ถ้าไม่มีจะถอยไปใช้การประมาณจากจำนวนอักขระ **และบอกไว้ในผลลัพธ์เสมอว่าใช้วิธีไหน**

จุดที่ต้องเน้น: โค้ด `scikit-learn` ที่ใช้เป็นฐานเปรียบเทียบนั้น **ยังไม่รวม** AutoML, Memory Downcasting, การตรวจ Drift, Data Quality Gate, SHAP และการเซฟโมเดลพร้อมลายเซ็น SHA-256 — ซึ่งฝั่ง PotatOpt ได้มาทั้งหมดโดยไม่ต้องเขียนเพิ่มอีกบรรทัดเดียว

**การลด Token ไม่ใช่แค่เรื่องประหยัดค่าใช้จ่าย แต่คือการป้องกันความผิดพลาด (Poka-Yoke)** ยิ่ง AI ต้องเขียนโค้ดเองน้อยเท่าไร โอกาสที่มันจะเผลอ Fit Scaler บนชุดทดสอบ หรือจูน Threshold บนชุดเดียวกับที่ใช้รายงานผล ก็ยิ่งน้อยลงเท่านั้น — `auto_analyze()` ทำให้ "เขียนผิดได้ยากตั้งแต่แรก" ส่วน `threshold_leakage_warning` ทำให้ "ถ้าผิดแล้วระบบร้องเตือน" ครบทั้งสองชั้นตามหลัก Poka-Yoke

---

## Type Hints

ทุกฟังก์ชันและเมธอดสาธารณะมี Type Hint ครบ พร้อม `from __future__ import annotations` ทำให้ Annotation ถูกเก็บเป็นสตริงและไม่ถูกประเมินตอน import — ใช้ไวยากรณ์สมัยใหม่อย่าง `str | None` และ `dict[str, Any]` ได้โดยไม่กระทบ Python เวอร์ชันเก่าและไม่เพิ่มเวลา import

เหตุผลที่ทำ: Type Hint คือสิ่งที่ทำให้ Editor หรือ **AI ที่กำลังอ่านซอร์สรู้ได้ว่าฟังก์ชันคืนอะไรโดยไม่ต้องรันโค้ด** ซึ่งลดจำนวนรอบการลองผิดลองถูก — เป็นเป้าหมายเดียวกับการลด Token มีเทสต์บังคับไว้แล้วว่าทุกชื่อใน `__all__` และทุกเมธอดสาธารณะของ `PotatOptEngine` ต้องมี Annotation ครบทั้งพารามิเตอร์และค่าที่คืน

**หมายเหตุ:** ไฟล์ `py.typed` (PEP 561) ที่จะทำให้ Type Checker ภายนอกอ่าน Hint เหล่านี้ได้ **ต้องรอจนกว่าจะแตก `potatopt.py` เป็นแพ็กเกจก่อน** เพราะ PEP 561 กำหนดให้ Marker ต้องวางอยู่ในไดเรกทอรีของแพ็กเกจ ซึ่งการแจกแบบไฟล์เดียว (single module) ทำไม่ได้

---

## การติดตั้ง (Installation)

```bash
pip install potatopt                # Core: numpy, pandas, scipy, scikit-learn เท่านั้น
pip install potatopt[automl]        # + FLAML, LightGBM, XGBoost (จำเป็นสำหรับ .fit())
pip install potatopt[xai]           # + SHAP (จำเป็นสำหรับ .get_shap_values() / .explain_predictions())
pip install potatopt[viz]           # + matplotlib, seaborn
pip install potatopt[all]           # ทุกอย่าง
```

**Core มีแค่ 4 แพ็กเกจโดยเจตนา** — FLAML และ SHAP เป็นสองตัวที่กินพื้นที่ติดตั้งและเวลา import มากที่สุด และทั้งคู่จะไม่ถูกแตะเลยจนกว่าจะเรียก `.fit()` หรือขอคำอธิบายโมเดลจริงๆ ดังนั้นจึงถูกโหลดแบบ Lazy ตอนใช้งานครั้งแรกเท่านั้น

นี่คือความต่างระหว่างไลบรารีที่**รันบนเครื่องสเปกต่ำได้จริง** กับไลบรารีที่แค่อ้างว่าทำได้ — และมีเทสต์บังคับไว้ (`test_importing_potatopt_does_not_load_the_heavy_backends`) ที่รัน `import potatopt` ในโปรเซสสะอาดแล้วยืนยันว่า `flaml` และ `shap` ไม่อยู่ใน `sys.modules`

ถ้าเรียกฟีเจอร์ที่ต้องใช้ Backend ที่ยังไม่ได้ติดตั้ง จะได้ `ImportError` ที่บอกคำสั่งติดตั้งตรงๆ เช่น:

```
PotatOptEngine.fit() needs the FLAML AutoML backend, which is not installed.
Install it with:  pip install potatopt[automl]
```

เวอร์ชันในไฟล์ `pyproject.toml` ถูกตั้งเป็น `dynamic` โดยอ่านจาก `potatopt.__version__` โดยตรง จึงไม่มีทางไม่ตรงกัน

---

## Control Chart สำหรับงานซ่อมบำรุง (EWMA / CUSUM)

Shewhart Chart แบบ $3\sigma$ ตอบได้แค่ว่า "จุดนี้หลุดกรอบไหม" ซึ่งไม่พอสำหรับงาน PdM เพราะ **ตลับลูกปืน ดอกกัด หรือปั๊ม ไม่ได้พังแบบกระโดด แต่ค่อยๆ เสื่อม** EWMA และ CUSUM สะสมการเลื่อนเล็กๆ ที่ต่อเนื่อง จึงเห็นแนวโน้มตั้งแต่ยังไม่มีจุดไหนหลุดกรอบ

#### `calculate_ewma_chart(values, lambda_weight=0.2, n_sigmas=3.0, target=None, sigma=None, baseline_n=None) -> dict`
$$z_i = \lambda x_i + (1-\lambda) z_{i-1}, \qquad z_0 = \text{target}$$

ใช้เส้นควบคุมรูปแบบที่แปรตามเวลา (Exact Time-Varying Form) ซึ่งแคบกว่ารูปแบบ Asymptotic ในช่วงต้น จึงไม่บังสัญญาณผิดปกติที่เกิดเร็ว:
$$\text{target} \pm n_\sigma \cdot \sigma \sqrt{\frac{\lambda}{2-\lambda}\left(1-(1-\lambda)^{2i}\right)}$$

#### `calculate_cusum_chart(values, target=None, sigma=None, slack_k=0.5, decision_h=5.0, baseline_n=None) -> dict`
$$SH_i = \max(0,\; SH_{i-1} + (x_i - \text{target}) - k\sigma), \qquad SL_i = \max(0,\; SL_{i-1} + (\text{target} - x_i) - k\sigma)$$

ส่งสัญญาณเมื่อฝั่งใดฝั่งหนึ่งเกิน $h\sigma$ เพราะผลรวมรีเซ็ตที่ศูนย์ CUSUM จึงเมิน Noise ได้ไม่จำกัด แต่ตอบสนองต่อการเลื่อนที่ต่อเนื่องภายในไม่กี่จุด

### จุดที่สำคัญที่สุด: การประมาณค่า $\sigma$

ทั้งสองฟังก์ชันประมาณ $\sigma$ จาก **ค่าเฉลี่ยของ Moving Range หารด้วย $d_2 = 1.128$** ไม่ใช่จากส่วนเบี่ยงเบนมาตรฐานของกลุ่มตัวอย่าง เพราะสองค่านี้ใช้แทนกันไม่ได้บนกระบวนการที่กำลังเสื่อม — **ข้อมูลที่ไต่ขึ้นเรื่อยๆ จะทำให้ค่า SD ของตัวเองพองขึ้น เส้นควบคุมถ่างออก แล้วบังแนวโน้มที่ Chart มีไว้เพื่อจับพอดี**

วัดจริงบนข้อมูลจำลองการสึกหรอ (ไต่ขึ้น 0.25 ต่อจุด):

| วิธีประมาณ $\sigma$ | ค่าที่ได้ | EWMA จับสัญญาณได้ที่จุดที่ |
|---|---|---|
| Sample Standard Deviation | 2.201 | **13** |
| Moving Range $/\, d_2$ | 0.222 | **3** |

**จับได้เร็วขึ้น 10 รอบการวัด** ซึ่งในงานซ่อมบำรุงคือเวลาที่ใช้วางแผนหยุดเครื่องได้ล่วงหน้าแทนที่จะเป็นการหยุดกะทันหัน

พารามิเตอร์ `baseline_n` จำกัดการประมาณ target และ $\sigma$ ให้ใช้เฉพาะ N จุดแรก (ช่วง Phase I ที่กระบวนการยังนิ่ง) — **สิ่งที่ `baseline_n` ให้ไม่ใช่ "จับได้เร็วขึ้น" แต่คือ "ไม่แจ้งเตือนผิดทิศ"** เพราะถ้าประมาณ target จากทั้งชุดที่กำลังไต่ขึ้น ค่า target จะถูกดึงสูงกว่าจุดที่กระบวนการเริ่มต้นจริง ทำให้จุดแรกๆ ตกใต้เส้นล่างแล้ว Chart รายงานว่า **"decreasing" ตั้งแต่จุดที่ 0 บนข้อมูลที่มีแต่ไต่ขึ้น** วัดจริงบนชุดทดสอบ: ประมาณจากทั้งชุดได้ target 12.44 แจ้งเตือนผิดทิศที่จุด 0 ส่วนใช้ `baseline_n=15` ได้ target 10.00 แจ้งเตือนถูกทิศที่จุด 16

หากช่วง Baseline บังเอิญแบนสนิท (เซนเซอร์ความละเอียดหยาบค้างค่าเดียวทั้งช่วง) ระบบจะถอยไปประมาณ $\sigma$ จากทั้งชุดแทนโดยอัตโนมัติ มีเพียงข้อมูลที่ไม่ขยับเลยจริงๆ เท่านั้นที่จะได้ $\sigma = 0$ — ป้องกันไม่ให้กระบวนการที่ไต่จาก 10 ไป 19.5 ถูกรายงานว่า "ปกติดี"

---

## Per-Asset Drift — ทำไมการรวมหลายเครื่องจักรถึงตอบผิดทั้งสองทาง

เครื่องจักรรุ่นเดียวกันก็ไม่เหมือนกัน เครื่องหนึ่งอยู่ใกล้ประตูจึงเย็นกว่า อีกเครื่องเดินร้อนกว่าเป็นปกติ เมื่อรวมทุกเครื่องเป็นโปรไฟล์เดียว **ความต่างระหว่างเครื่องจะกลายเป็นไม้บรรทัด** แล้วผิดพลาดสองทางพร้อมกัน

### ทางที่หนึ่ง: แจ้งเตือนทั้งที่ไม่มีอะไรเกิดขึ้น

เครื่องจักร 3 ตัวเดินที่ 70°C, 75°C, 80°C สุขภาพดีทุกตัว จากนั้น M-01 ถูกถอดไปซ่อมตามแผน จึงส่งข้อมูลมา 10 แถวแทนที่จะเป็น 200 — **ไม่มีเครื่องไหนเปลี่ยนพฤติกรรมเลยแม้แต่นิดเดียว** แต่

| วิธีตรวจ | ผลลัพธ์ |
|---|---|
| รวมเป็นก้อนเดียว | `drift_detected = True`, PSI 1.26 ← **แจ้งเตือนหลอก** |
| แยกรายเครื่อง | `drift_detected = False`, M-01 ถูกทำเครื่องหมาย `insufficient_data` |

สิ่งที่เปลี่ยนคือ**สัดส่วนของเครื่องที่ส่งข้อมูล ไม่ใช่ตัวกระบวนการ**

### ทางที่สอง: กลบสัญญาณจริงจนเกือบมองไม่เห็น

คราวนี้ M-02 ร้อนขึ้นจริง +3°C

| วิธีตรวจ | Magnitude | บอกได้มั้ยว่าเครื่องไหน |
|---|---|---|
| รวมเป็นก้อนเดียว | **0.244** (เฉียดเกณฑ์ 0.2) | ไม่ได้ |
| แยกรายเครื่อง | **3.084** | ได้ — `assets_drifted = ['M-02']` |

เจือจางไป **12.5 เท่า** และกลไกเป็นตัวเดียวกับที่เจอตอนทำ EWMA เป๊ะ: ส่วนเบี่ยงเบนมาตรฐานรวมเท่ากับ 4.18 เพราะมันบรรจุ*ความต่างระหว่างเครื่อง*เอาไว้ ไม้บรรทัดจึงยาวขึ้น 4.18 เท่า ขณะที่ตัวเศษหดลง 3 เท่าเพราะขยับแค่ 1 ใน 3 เครื่อง

> **บทเรียนที่ซ้ำเป็นครั้งที่สอง:** ความแปรปรวนที่เราไม่ได้แยกออกมา จะไปพองไม้บรรทัดของเราเอง แล้วบังสัญญาณที่เครื่องมือนั้นมีไว้จับพอดี

### กับดักที่การแยกรายเครื่องสร้างขึ้นมาเอง

แยกแล้ว batch ต่อเครื่องย่อมเล็กลง และสถิติบนกลุ่มตัวอย่างเล็กเชื่อไม่ได้ วัดอัตราแจ้งเตือนหลอกของ `check_data_drift` โดยสุ่ม batch จาก**การแจกแจงเดียวกับชุดเทรน** (ทุกครั้งที่แจ้งเตือนคือหลอกล้วนๆ)

| จำนวนแถวใน batch | 5 | 10 | 20 | 30 | 50 | 75 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| แจ้งเตือนหลอก | 100% | 100% | 92% | 81% | 50% | 22% | 10.5% | 1.0% |

การแยกรายเครื่องแบบไร้เดียงสาจึงเป็นการ**แลกคำตอบผิดแบบหนึ่งกับผิดอีกแบบหนึ่ง** ต้องมีด่านกัน 2 ชั้น

**ด่านที่ 1 — ปรับจำนวน bin ของ PSI ตามขนาด batch** `n_bins = min(10, max(2, n // 10))` เพราะ 10 quantile bin ต้องการข้อมูลราว 10 จุดต่อ bin

| แถวใน batch | 10 | 30 | 50 | 100 |
|---|---|---|---|---|
| ตรึง 10 bin | 48.5% | 51.5% | 32.2% | 5.2% |
| ปรับตามขนาด | **9.8%** | **5.0%** | **3.5%** | **3.5%** |

และความผิดปกติจริง +3°C ยังได้ PSI 4.6–6.9 ในทุกขนาด สูงกว่าเส้น 0.25 มาก — **ด่านนี้ไม่ทำให้พลาดของจริงเลย**

**ด่านที่ 2 — เส้นพื้นของสัญญาณรบกวน** การเทียบ $|\Delta\bar{x}|/\sigma$ กับเลข 0.2 ตายตัว มองข้ามว่าค่าเฉลี่ยของ batch เองก็มีความคลาดเคลื่อน $1/\sqrt{n}$ จึงบังคับให้ต้องผ่าน**ทั้งสองเงื่อนไข**

$$\text{effective threshold} = \max\left(\text{threshold\_pct},\; k \sqrt{\tfrac{1}{n_{batch}} + \tfrac{1}{n_{train}}}\right)$$

นี่คือการแยก **Practical Significance** (ใหญ่พอที่จะต้องลงมือมั้ย) ออกจาก **Statistical Significance** (แยกออกจากความบังเอิญได้มั้ย) ซึ่งเป็นวิธีคิดพื้นฐานของงาน IE วัดจริงด้วยชุดเทรน 300 แถว

| แถวใน batch | 10 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| เส้นพื้น | 0.964 | 0.574 | 0.458 | 0.346 | 0.274 |
| แจ้งเตือนหลอก (เดิม → ใหม่) | 59.2% → **0.8%** | 39.3% → **0.7%** | 29.7% → **0.8%** | 24.5% → **1.3%** | 12.0% → **1.2%** |

**ราคาที่ต้องจ่าย พูดกันตรงๆ:** ความเปลี่ยนแปลงขนาด 0.5σ ที่มองผ่านข้อมูลแค่ 30 แถว จะถูกรายงาน 69.8% แทนที่จะเป็น 98.8% — แต่ตัวเลข 98.8% นั้นมาพร้อมอัตราแจ้งเตือนหลอก 39.3% ซึ่งทำให้มันไม่มีความหมายตั้งแต่ต้น ส่วนสัญญาณขนาด 1.0σ ขึ้นไปยังจับได้ **100% ที่ 30 แถวขึ้นไป** คำตอบของ 0.5σ คือเก็บข้อมูลเพิ่ม ไม่ใช่ลดเกณฑ์ลง

### สถานะรายเครื่อง และเหตุผลที่ต้องวนทั้งสองฝั่ง

ทุกเครื่องที่ปรากฏใน **ฝั่งใดฝั่งหนึ่ง** จะมี `status` เสมอ

| สถานะ | ความหมาย |
|---|---|
| `checked` | เทียบตามปกติ |
| `insufficient_data` | ข้อมูลฝั่งใดฝั่งหนึ่งน้อยกว่า `min_rows` |
| `unknown_asset` | มีใน batch แต่ไม่เคยอยู่ในชุดเทรน |
| `missing_from_batch` | **มีในชุดเทรนแต่เงียบหายไป** |

สถานะสุดท้ายคือเหตุผลที่ต้องวน union ของทั้งสองฝั่ง ถ้าวนเฉพาะ batch **เครื่องที่หยุดส่งข้อมูลจะล่องหน** — Gateway ตายกับเครื่องที่สุขภาพดีจะดูเหมือนกันทุกประการ ซึ่งเป็นโหมดพังแบบเดียวกับ Control Chart ที่ไม่มีวันแจ้งเตือน

```python
report = po.check_asset_drift(train_df, batch_df, asset_col="machine_id")

if report["drift_detected"]:
    print("เครื่องที่ต้องไปดู:", report["assets_drifted"])
for asset, reason in report["assets_skipped"].items():
    print(f"ตัดสินไม่ได้ - {asset}: {reason}")
```

---

## กรอบต้นทุนงานซ่อมบำรุง (Maintenance Cost Framing)

`calculate_cost_of_quality` แปลง Confusion Matrix เป็นเงินอยู่แล้ว แต่พูดภาษา **การควบคุมคุณภาพ**: FN คือของเสีย, FP คือ False Alarm, TP คือค่าตรวจสอบ ซึ่งถูกต้องสำหรับสายการผลิตที่ผลิตชิ้นงาน

แต่**ผิดกรอบสำหรับการดูแลเครื่องจักร** ซึ่งเป็นทิศทางของโปรเจกต์นี้ ทั้งสี่ช่องมีความหมายต่างออกไป

| ช่อง | ความหมายในงานซ่อมบำรุง | ต้นทุน |
|---|---|---|
| **TP** | ทำนายได้ จึงซ่อมตอนหยุดเครื่องตามแผน | ค่าเข้าไปตรวจ + ค่าซ่อมตามแผน (เลี่ยงของแพงได้) |
| **FN** | พลาด เครื่องพังกลางกะ | **ค่าเครื่องพังกะทันหัน** ← ตัวที่ครอบงำสมการ |
| **FP** | แจ้งเตือนเครื่องที่ปกติ คนไปดูแล้วไม่เจออะไร | ค่าเข้าไปตรวจอย่างเดียว |
| **TN** | ไม่มีอะไรเกิดขึ้น | 0 |

**FP คิดแค่ค่าเข้าไปตรวจ ไม่ใช่ค่าอะไหล่** เพราะช่างที่ทำงานเป็นย่อมเข้าไปดูก่อนเปลี่ยนของ

และเส้นเปรียบเทียบไม่ใช่ "การตรวจด้วยคน" แต่คือ **Run to Failure**: ถ้าไม่มีโมเดลเลย ทุกความเสียหายที่เกิดขึ้นจะกลายเป็นเครื่องพังกะทันหันทั้งหมด

$$\text{Run to Failure} = (TP + FN) \times C_{breakdown}$$
$$\text{With Model} = FN \cdot C_{breakdown} + TP \cdot (C_{inspection} + C_{planned}) + FP \cdot C_{inspection}$$

### สิ่งที่กรอบนี้ทำให้มองเห็น

ค่าเริ่มต้น: เครื่องพัง 50,000 · ซ่อมตามแผน 8,000 · เข้าไปตรวจ 1,500 (เปลี่ยนได้ และ**ควรเปลี่ยน**ให้ตรงกับตัวเลขจริงของโรงงาน)

| โมเดล | TP | FP | FN | Recall | ประหยัด |
|---|---|---|---|---|---|
| โมเดลที่ดี | 18 | 25 | 2 | 0.90 | **+691,500** (69.15%) |
| แจ้งเตือนทุกเครื่อง | 20 | 980 | 0 | **1.00** | **−660,000** (−66.00%) |
| ไม่แจ้งเตือนอะไรเลย | 0 | 0 | 20 | 0.00 | 0 (เท่ากับ Run to Failure พอดี) |

**แถวที่สองคือเหตุผลทั้งหมดของหัวข้อนี้** โมเดลนั้น Recall เต็ม 100% จับได้ทุกความเสียหาย และ**เผาเงินทิ้ง 660,000** เพราะการวิ่งไปดูเปล่าๆ 980 ครั้งแพงกว่าความเสียหายที่มันป้องกันได้ จุดคุ้มทุนของโมเดลนี้อยู่ที่ 540 ครั้ง — **Recall บอกเรื่องนี้ไม่ได้ มีแต่ต้นทุนที่บอกได้**

นี่คือเหตุผลที่ `breakdown_avoidance_rate` ถูกรายงานคู่กับ `cost_savings` เสมอ เพราะสองค่านี้จะขัดแย้งกันในกรณีที่สำคัญที่สุด

#### `calculate_maintenance_savings(true_positives, false_positives, false_negatives, cost_breakdown=50000.0, cost_planned=8000.0, cost_inspection=1500.0) -> dict`
ฟังก์ชันอิสระ (stateless) เรียกได้โดยไม่ต้องมีโมเดล คืน `run_to_failure_cost`, `predictive_cost`, `cost_savings`, `savings_percentage`, `breakdowns_avoided`, `unplanned_breakdowns`, `breakdown_avoidance_rate`, `wasted_inspections`, `cost_assumptions` และคืน `{"error": ...}` แทนการ raise เสมอ

#### `.calculate_maintenance_cost(X_test, y_test, cost_breakdown=50000.0, cost_planned=8000.0, cost_inspection=1500.0) -> dict`
คู่แฝดของ `.calculate_cost_of_quality()` Confusion Matrix ชุดเดียวกัน แต่ถามคนละคำถาม

#### `.optimize_maintenance_threshold(X_val, y_val, ...) -> float`
หา Threshold ที่ทำให้ต้นทุนซ่อมบำรุงรวมต่ำที่สุด เนื่องจากเครื่องพังกะทันหันมักแพงกว่าการซ่อมตามแผนหลายเท่า **Threshold ที่ถูกที่สุดจึงมักต่ำกว่า 0.5** — ยอมวิ่งไปดูเปล่าหลายครั้งเพื่อเลี่ยงเครื่องพังหนึ่งครั้งนั้นคุ้มกว่า

> **ระเบียบวิธี:** ส่ง **Validation** partition เท่านั้น ไม่ใช่ Test เพราะ Threshold ถูกเลือกให้ถูกที่สุด*บนข้อมูลชุดนั้น* เมธอดนี้บันทึก `threshold_tuning_fingerprint` เหมือน `optimize_threshold` ดังนั้น `evaluate()` จะยังตั้ง `threshold_leakage_warning` ให้ถ้าเผลอใช้ชุดเดียวกัน

---
