import json
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

st.set_page_config(page_title="Loan Prediction Logistic Regression Model (JSON) + Hardrule", layout="centered")

MODEL_PATH = "logistic_regression_model.pkl"
SCALER_PATH = "scaler_model.pkl"

DEFAULT_UNKNOWN = 0

# ===== URUTAN INPUT (final_decision TIDAK ADA) =====
FEATURE_ORDER_UI = [
    "cust_id","Borrow_amt","timeperiod","age","post_code","gender","marital_status",
    "no_of_dependents","residential_status","totalDTI","Savings",
    "monthly_inc_after_enc_n_tax","Total_monthly_finan_commitments","emp_status","emp_type",
    "time_of_work","loans_from_non_bank","ctos_score","bank_ruptcyStausvalue",
    "availablility_legal_record","DishonoredChequesValue","loanbyCC","credit_limit",
    "credit_outstanding_number","creditapp12mthnumber","creditapp12mthapproved",
    "creditApp12mthPending","trade_refrence","special_attention","Litigation_index",
    "secured_facilities","secured_facilities_outstanding_number","unsecured_facilities",
    "unsecured_facilities_outstanding_number","food","loan_credits_card","other",
    "rent_home_loan","transportation","utilities_bills","RELOAN","TOPUP"
]

# ===== KOLOM YANG TIDAK DIPAKAI MODEL SAAT PREDICT =====
DROP_FOR_MODEL = {
    "cust_id", "Litigation_index", "bank_ruptcyStausvalue", "DishonoredChequesValue",
    "post_code", "availablility_legal_record", "timeperiod",
    "RELOAN", "TOPUP"
}

# ===== HARD RULE 11: POSTCODE WHITELIST =====
ALLOWED_POSTCODES = set([
  40000,40100,40150,40160,40170,40200,40300,40400,40450,40460,40470,40500,40502,40503,
  40505,40512,40517,40520,40529,40542,40548,40550,40551,40560,40564,40570,40572,40576,
  40578,40582,40590,40592,40594,40596,40598,40604,40607,40608,40610,40612,40620,40622,
  40626,40632,40646,40648,40660,40664,40670,40672,40673,40674,40675,40676,40680,40690,
  40700,40702,40704,40706,40708,40710,40712,40714,40716,40718,40720,40722,40724,40726,
  40728,40730,40732,40800,40802,40804,40806,40808,40810,40990,41000,41050,41100,41150,
  41200,41250,41300,41400,41506,41560,41586,41672,41700,41710,41720,41900,41902,41904,
  41906,41908,41910,41912,41914,41916,41918,41990,42000,42009,42100,42200,42300,42500,
  42507,42509,42600,42610,42700,42800,42920,42940,42960,43000,43007,43009,43100,43200,
  43207,43300,43400,43500,43558,43600,43650,43700,43800,43807,43900,43950,44000,44010,
  44020,44100,44110,44200,44300,45000,45100,45200,45207,45209,45300,45400,45500,45600,
  45607,45609,45620,45700,45800,46000,46050,46100,46150,46200,46300,46350,46400,46506,
  46547,46549,46551,46564,46582,46598,46662,46667,46668,46672,46675,46700,46710,46720,
  46730,46740,46750,46760,46770,46780,46781,46782,46783,46784,46785,46786,46787,46788,
  46789,46790,46791,46792,46793,46794,46795,46796,46797,46798,46799,46800,46801,46802,
  46803,46804,46805,46806,46860,46870,46960,46962,46964,46966,46968,46970,46972,46974,
  46976,46978,47000,47100,47110,47120,47130,47140,47150,47160,47170,47180,47190,47200,
  47300,47301,47307,47308,47400,47410,47500,47507,47600,47610,47620,47630,47640,47650,
  47800,47810,47820,47830,48000,48010,48020,48050,48100,48200,48300,63000,63100,63200,
  63300,64000,68000,68100,50000,50050,50088,50100,50150,50200,50250,50300,50350,50400,
  50450,50460,50470,50480,50490,50500,50502,50504,50505,50506,50507,50508,50512,50514,
  50515,50519,50528,50529,50530,50532,50534,50536,50540,50544,50546,50548,50550,50551,
  50552,50554,50556,50560,50562,50564,50566,50568,50572,50576,50578,50580,50582,50586,
  50588,50590,50592,50594,50596,50598,50599,50600,50603,50604,50605,50608,50609,50610,
  50612,50614,50620,50621,50622,50623,50626,50632,50634,50636,50638,50640,50642,50644,
  50646,50648,50650,50652,50653,50656,50658,50660,50661,50662,50664,50666,50668,50670,
  50672,50673,50676,50677,50678,50680,50682,50684,50688,50694,50700,50702,50704,50706,
  50708,50710,50712,50714,50716,50718,50720,50722,50724,50726,50728,50730,50732,50734,
  50736,50738,50740,50742,50744,50746,50748,50750,50752,50754,50758,50760,50762,50764,
  50766,50768,50770,50772,50774,50776,50778,50780,50782,50784,50786,50788,50790,50792,
  50794,50796,50798,50800,50802,50804,50806,50808,50810,50812,50814,50816,50818,50901,
  50902,50903,50904,50906,50907,50908,50909,50910,50911,50912,50913,50914,50915,50916,
  50917,50918,50919,50920,50921,50922,50923,50924,50925,50926,50927,50928,50929,50930,
  50931,50932,50933,50934,50935,50936,50937,50938,50939,50940,50941,50942,50943,50944,
  50945,50946,50947,50948,50949,50950,50988,50989,50990,51000,51100,51200,51700,51990,
  52000,52100,52109,52200,53000,53100,53200,53300,53700,53800,53990,54000,54100,54200,
  55000,55100,55188,55200,55300,55700,55710,55720,55900,55902,55904,55906,55908,55910,
  55912,55914,55916,55918,55920,55922,55924,55926,55928,55930,55932,55934,55990,56000,
  56100,57000,57100,57700,57990,58000,58100,58200,58700,58990,59000,59100,59200,59700,
  59800,59990,60000,62000,62007,62050,62100,62150,62200,62250,62300,62502,62504,62505,
  62506,62510,62512,62514,62516,62517,62518,62519,62520,62522,62524,62526,62527,62530,
  62532,62536,62540,62542,62546,62550,62551,62570,62574,62576,62582,62584,62590,62592,
  62596,62602,62604,62605,62606,62616,62618,62620,62623,62624,62628,62630,62632,62648,
  62652,62654,62662,62668,62670,62674,62675,62676,62677,62686,62692,62988
])

# ===== MAPPING ENCODER KATEGORI (UNTUK MODEL) =====
CATEGORY_ENCODERS: Dict[str, Dict[str, float]] = {
    "gender": {"0": 0, "male": 1, "lelaki": 1, "female": 2, "perempuan": 2},
    "marital_status": {"0": 0, "single": 4, "bujang": 4, "married": 3, "berkahwin": 3, "divorced": 2, "bercerai": 2, "widowed": 1, "balu atau duda": 1},
    "residential_status": {"0": 0, "company property": 1, "quarters company": 1, "rumah kepunyaan syarikat": 1, "own property": 3, "rumah sendiri": 3,
                           "parents property": 4, "parents house": 4, "rumah ibu bapa": 4, "relatives property": 5, "relative property": 5, "rumah saudara": 5,
                           "renting": 6, "rental": 6, "sewa": 6, "others": 2, "lain-lain": 2},
    "emp_status": {"0": 0, "government employee or civil servant": 1, "government employee/civil servant": 1, "kakitangan atau penjawat sektor awam": 1,
                   "private sector employee": 2, "pekerja sektor swasta": 2, "self-employed with workers": 3, "bekerja sendiri dengan pekerja": 3,
                   "self-employed without workers": 4, "bekerja sendiri tanpa pekerja": 4, "retired or pensioner": 5, "pesara atau berpencen": 5,
                   "housewife": 6, "unemployed": 7, "tidah berkerja": 7, "student": 8},
    "emp_type": {"0": 0, "contract": 1, "pekerja kontrak": 1, "permanent": 2, "pekerja tetap": 2, "temporary": 3, "pekerja sementara": 3},
    "time_of_work": {"0": 0, "1 to 2 years": 1, "1to2years": 1, "1 - 2 tahun": 1, "1-2 years": 1,
                     "1 - 3 bulan": 2, "1 - 3 months": 2, "1-3 months": 2,
                     "2 - 3 tahun": 3, "2 to 3 years": 3, "2to3years": 3, "2-3 years": 3,
                     "3 years": 4, "3 years+": 5, "3+ years": 5, "lebih daripada 3 tahun": 5,
                     "4 - 6 bulan": 6, "4-6 months": 6, "7 - 12 bulan": 7, "7 - 12 months": 7, "7-12 months": 7}
}
CATEGORICAL_COLS = list(CATEGORY_ENCODERS.keys())

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def normalize(x: Any) -> str:
    return str(x).lower().strip()

def parse_flag_01(x: Any) -> int:
    s = normalize(x)
    if s in ["1", "yes", "y", "true", "on"]:
        return 1
    if s in ["0", "no", "n", "false", "", "none", "null"]:
        return 0
    try:
        return 1 if int(float(s)) == 1 else 0
    except Exception:
        return 0

def get_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def get_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def encode_one_value(col: str, raw_value: Any) -> float:
    if col not in CATEGORY_ENCODERS:
        return get_float(raw_value, 0.0)
    try:
        return float(raw_value)
    except Exception:
        pass
    return float(CATEGORY_ENCODERS[col].get(normalize(raw_value), DEFAULT_UNKNOWN))

def auto_encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in CATEGORY_ENCODERS.items():
        if col in out.columns:
            def encode_cell(v: Any) -> float:
                try:
                    return float(v)
                except Exception:
                    pass
                return float(mapping.get(normalize(v), DEFAULT_UNKNOWN))
            out[col] = out[col].apply(encode_cell).astype(float)
    return out

def normalize_label(pred_value: Any) -> str:
    if isinstance(pred_value, str):
        p = pred_value.strip().lower()
        if p in ["approved", "approve", "yes", "y", "1", "true"]:
            return "Approved"
        if p in ["disapproved", "disapprove", "no", "n", "0", "false"]:
            return "Disapproved"
        return pred_value
    try:
        return "Approved" if int(pred_value) == 1 else "Disapproved"
    except Exception:
        return str(pred_value)

def parse_postcode(raw_pc: Any) -> int | None:
    s = str(raw_pc).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def check_hardrules(full_input: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    TOPUP = parse_flag_01(full_input.get("TOPUP", "0"))
    RELOAN = parse_flag_01(full_input.get("RELOAN", "0"))

    totalDTI = get_float(full_input.get("totalDTI", 0))
    DishonoredChequesValue = get_float(full_input.get("DishonoredChequesValue", 0))
    creditApp12mthPending = get_float(full_input.get("creditApp12mthPending", 0))
    bank_ruptcyStausvalue = get_float(full_input.get("bank_ruptcyStausvalue", 0))
    availablility_legal_record = get_float(full_input.get("availablility_legal_record", 0))
    loanbyCC = get_float(full_input.get("loanbyCC", 0))
    special_attention = get_int(full_input.get("special_attention", 0))
    trade_refrence = get_float(full_input.get("trade_refrence", 0))
    monthly_inc_after_enc_n_tax = get_float(full_input.get("monthly_inc_after_enc_n_tax", 0))
    ctos_score = get_float(full_input.get("ctos_score", 0))
    residential_status_enc = encode_one_value("residential_status", full_input.get("residential_status", "0"))

    # 2
    if DishonoredChequesValue > 0 and totalDTI > 95.0:
        reasons.append("Hardrule 2: DishonoredChequesValue>0 & totalDTI>95.0")
    # 3
    if creditApp12mthPending > 0 and totalDTI > 95.0:
        reasons.append("Hardrule 3: creditApp12mthPending>0 & totalDTI>95.0")
    # 4
    if totalDTI > 95.0:
        reasons.append("Hardrule 4: totalDTI>95.0")
    # 5
    if bank_ruptcyStausvalue >= 1:
        reasons.append("Hardrule 5: Bankruptcy Status >= 1")
    # 6
    if availablility_legal_record > 12:
        reasons.append("Hardrule 6: Avialibity of legal record>12")
    # 7
    if loanbyCC > 95.0:
        reasons.append("Hardrule 7: loanbyCC>95.0")
    # 8
    if special_attention == 1:
        reasons.append("Hardrule 8: Special_Attention==1")
    # 9
    if trade_refrence > 1.0:
        reasons.append("Hardrule 9: trade_refrence>1.0")
    # 10
    if (monthly_inc_after_enc_n_tax <= 2000) and (ctos_score == 0) and (residential_status_enc == 6):
        reasons.append("Hardrule 10: monthly_inc_after_enc_n_tax<=2000 & ctos_score==0 & residential_status==6")

    # 11 - postcode harus ada di whitelist
    pc = parse_postcode(full_input.get("post_code", ""))
    if pc is None or pc not in ALLOWED_POSTCODES:
        reasons.append("Hardrule 11: invalid post_code (not in allowed list)")

    # 1 (gating note)
    if (TOPUP == 1 or RELOAN == 1) and len(reasons) > 0:
        reasons.insert(0, "Hardrule 1: TOPUP/RELOAN=1 + other hardrule hits -> model not called")
    return reasons

def build_defaults() -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for k in FEATURE_ORDER_UI:
        if k in ["cust_id", "timeperiod", "post_code"]:
            d[k] = ""
        elif k in ["RELOAN", "TOPUP"]:
            d[k] = "0"
        elif k in CATEGORICAL_COLS:
            d[k] = "0"
        else:
            d[k] = 0.0
    return d

# =========================
# UI
# =========================
st.title("Loan Prediction Logistic Regression Model (JSON Input) + Hardrule 1-11")
st.caption("Paste JSON input → hardrule check → if no hardrule hits, call model.")

model, scaler = load_artifacts()

default_payload = build_defaults()

with st.expander("Template JSON (copy)"):
    st.code(json.dumps(default_payload, indent=2), language="json")

raw_json = st.text_area("JSON Input", value=json.dumps(default_payload, indent=2), height=420)
run = st.button("Predict")

if run:
    try:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("JSON must in object/dict.")

        # merge defaults -> missing key tidak bikin error
        full_input = build_defaults()
        full_input.update(payload)

        X_full = pd.DataFrame([full_input])

        # ===== 1) Hardrule =====
        hardrule_hits = check_hardrules(full_input)
        if hardrule_hits:
            st.error("REJECTED (Hardrule) — model not called")
            st.write("Reasons:")
            for r in hardrule_hits:
                st.write(f"- {r}")

            with st.expander("Input FULL (audit)"):
                st.dataframe(X_full[FEATURE_ORDER_UI])
            st.stop()

        # ===== 2) Lolos hardrule -> Model =====
        X_model = X_full.drop(columns=list(DROP_FOR_MODEL), errors="ignore")
        X_model_enc = auto_encode_categories(X_model)

        if hasattr(scaler, "feature_names_in_"):
            missing = [c for c in scaler.feature_names_in_ if c not in X_model_enc.columns]
            if missing:
                raise ValueError(f"Kolom input kurang untuk scaler: {missing}")
            X_model_enc = X_model_enc[list(scaler.feature_names_in_)]

        X_scaled = scaler.transform(X_model_enc)

        pred = model.predict(X_scaled)[0]
        label = normalize_label(pred)

        proba = None
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X_scaled)[0]
            classes = list(getattr(model, "classes_", []))
            if "Approved" in classes:
                proba = float(proba_all[classes.index("Approved")])
            elif len(proba_all) > 1:
                proba = float(proba_all[1])

        st.success(f"Prediction: **{label}**")
        if proba is not None:
            st.info(f"Probability Approved: **{proba:.4f}**")

        with st.expander("Input FULL (audit)"):
            st.dataframe(X_full[FEATURE_ORDER_UI])

        with st.expander("Input MODEL (drop + encode)"):
            st.dataframe(X_model_enc)

    except json.JSONDecodeError as e:
        st.error(f"JSON Not Valid: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
