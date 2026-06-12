# ============================================================
# app.py — NIT Calicut Face Recognition Attendance System
# Streamlit UI | Tesseract OCR | MongoDB Atlas | dlib
# ============================================================

import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import face_recognition
import re
import pandas as pd
from pymongo import MongoClient, errors
from datetime import datetime
import sys

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NIT Calicut — Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark navy sidebar, teal accent — NIT Calicut color palette */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2d45 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f4f8 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 1.05rem; }

    /* Main area */
    .main { background-color: #f7fafc; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0d6efd, #0dcaf0);
        color: white !important;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Status boxes */
    .success-box {
        background: #d1fae5; border-left: 4px solid #10b981;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0;
    }
    .error-box {
        background: #fee2e2; border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0;
    }
    .info-box {
        background: #dbeafe; border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0;
    }

    /* Card container */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# DATABASE CONNECTION (cached so it runs only once)
# ════════════════════════════════════════════════════════════

MONGO_URI             = "mongodb+srv://namanghoti1:Naman4112005@cluster0.e8pzwh4.mongodb.net/?appName=Cluster0"
DB_NAME               = "College_Platform_DB"
STUDENTS_COLLECTION   = "Registered_Students"
ATTENDANCE_COLLECTION = "Daily_Attendance"

@st.cache_resource
def get_db():
    """
    Cached MongoDB connection — Streamlit re-uses this across
    all reruns instead of opening a new socket every time.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=6000)
        client.admin.command("ping")
        db = client[DB_NAME]

        # Indexes (idempotent — safe to call on every startup)
        db[STUDENTS_COLLECTION].create_index("roll_number", unique=True)
        db[ATTENDANCE_COLLECTION].create_index(
            [("roll_number", 1), ("subject_code", 1), ("date", 1)],
            unique=True
        )
        return db
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()

db             = get_db()
students_col   = db[STUDENTS_COLLECTION]
attendance_col = db[ATTENDANCE_COLLECTION]


# ════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ════════════════════════════════════════════════════════════

class NoFaceDetectedError(Exception):
    pass

class MultipleFacesDetectedError(Exception):
    pass


# ════════════════════════════════════════════════════════════
# HELPER: OCR EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_id_card_info(pil_image: Image.Image) -> dict:
    """
    Runs Tesseract OCR on the ID card image and uses regex to
    extract Name and Roll Number.

    NIT Calicut roll number formats supported:
        B21CE001  /  B21CS042  /  M22ME010  /  22CE001  /  21CS0042
    The regex is intentionally broad to catch slight OCR noise.

    Returns:
        dict with keys: 'raw_text', 'name', 'roll_no'
        Values are empty strings if not found.
    """
    # Upscale image for better OCR accuracy (2x)
    w, h = pil_image.size
    pil_upscaled = pil_image.resize((w * 2, h * 2), Image.LANCZOS)

    # Tesseract config: PSM 6 = assume a single uniform block of text
    custom_config = r"--oem 3 --psm 6"
    raw_text = pytesseract.image_to_string(pil_upscaled, config=custom_config)

    # ── Roll Number Regex ────────────────────────────────────
    # Matches: optional letter prefix + 2-digit year + 2-letter branch + 3-4 digits
    # Examples: B21CE001, 22CE0042, M23CS010
    roll_pattern = re.compile(
        r"\b([A-Z]?\d{2}[A-Z]{2,3}\d{3,4})\b",
        re.IGNORECASE
    )
    roll_match = roll_pattern.search(raw_text)
    roll_no = roll_match.group(1).upper() if roll_match else ""

    # ── Name Regex ───────────────────────────────────────────
    # Strategy: look for a line that starts after "Name:" or "Student:"
    # Falls back to the longest all-caps line (common on ID cards).
    name = ""
    name_pattern = re.compile(
        r"(?:Name|Student Name|Student)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+)",
        re.IGNORECASE
    )
    name_match = name_pattern.search(raw_text)
    if name_match:
        name = name_match.group(1).strip()
    else:
        # Fallback: longest ALL-CAPS line (names on ID cards are often printed in caps)
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        caps_lines = [l for l in lines if l.isupper() and len(l.split()) >= 2 and len(l) > 5]
        if caps_lines:
            name = max(caps_lines, key=len).title()

    return {"raw_text": raw_text, "name": name, "roll_no": roll_no}


# ════════════════════════════════════════════════════════════
# HELPER: FACE ENCODING EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_face_encoding(pil_image: Image.Image):
    """
    Detects exactly one face in the image and returns its
    128-D encoding as a Python list (MongoDB-serializable).

    Raises NoFaceDetectedError or MultipleFacesDetectedError.
    """
    img_array = np.array(pil_image.convert("RGB"))

    face_locations = face_recognition.face_locations(img_array, model="hog")
    n = len(face_locations)

    if n == 0:
        raise NoFaceDetectedError(
            "No face detected. Ensure the ID card has a clear, unobstructed photo."
        )
    if n > 1:
        raise MultipleFacesDetectedError(
            f"{n} faces found. Please crop the ID card so only the student portrait is visible."
        )

    encodings = face_recognition.face_encodings(
        img_array, known_face_locations=face_locations, num_jitters=2
    )
    return encodings[0].tolist()


# ════════════════════════════════════════════════════════════
# HELPER: BULK ATTENDANCE PROCESSING
# ════════════════════════════════════════════════════════════

def process_attendance(pil_image: Image.Image, subject_code: str) -> dict:
    """
    Detects all faces in the classroom image, matches against DB,
    and bulk-inserts Present records.

    Returns a summary dict with keys:
        total_detected, matched, unknown, duplicates, inserted
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now()

    all_students = list(students_col.find({}))
    if not all_students:
        return {"error": "No registered students in database."}

    known_encodings = [np.array(s["face_encoding"]) for s in all_students]

    img_array      = np.array(pil_image.convert("RGB"))
    face_locations = face_recognition.face_locations(img_array, model="hog")

    if not face_locations:
        return {"error": "No faces detected in the uploaded image."}

    detected_encodings = face_recognition.face_encodings(img_array, face_locations)

    TOLERANCE       = 0.5
    records         = []
    already_counted = set()
    unknown_count   = 0
    duplicate_count = 0

    for detected_enc in detected_encodings:
        distances     = face_recognition.face_distance(known_encodings, detected_enc)
        best_idx      = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance > TOLERANCE:
            unknown_count += 1
            continue

        student = all_students[best_idx]
        roll_no = student["roll_number"]
        name    = student["student_name"]

        if roll_no in already_counted:
            continue
        already_counted.add(roll_no)

        # Duplicate check against DB
        if attendance_col.find_one({"roll_number": roll_no,
                                    "subject_code": subject_code,
                                    "date": today_str}):
            duplicate_count += 1
            continue

        records.append({
            "date"        : today_str,
            "timestamp"   : timestamp,
            "subject_code": subject_code,
            "student_name": name,
            "roll_number" : roll_no,
            "status"      : "Present"
        })

    inserted = 0
    if records:
        try:
            result  = attendance_col.insert_many(records, ordered=False)
            inserted = len(result.inserted_ids)
        except errors.BulkWriteError as bwe:
            inserted = bwe.details.get("nInserted", 0)

    return {
        "total_detected": len(face_locations),
        "matched"       : len(already_counted),
        "unknown"       : unknown_count,
        "duplicates"    : duplicate_count,
        "inserted"      : inserted
    }


# ════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 NIT Calicut")
    st.markdown("**Attendance Management System**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📋 Register Student", "📷 Mark Attendance", "📊 View Records"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    total_students = students_col.count_documents({})
    today_str      = datetime.now().strftime("%Y-%m-%d")
    today_present  = attendance_col.count_documents({"date": today_str})

    st.metric("Registered Students", total_students)
    st.metric("Present Today", today_present)
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %H:%M')}")


# ════════════════════════════════════════════════════════════
# PAGE 1: STUDENT REGISTRATION (OCR FLOW)
# ════════════════════════════════════════════════════════════

if page == "📋 Register Student":
    st.title("📋 Student Registration")
    st.markdown("Upload the student's **College ID Card**. The system will auto-extract the name and roll number via OCR.")

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">Step 1 — Upload ID Card</div>', unsafe_allow_html=True)
        id_card_file = st.file_uploader(
            "Upload ID Card Image",
            type=["jpg", "jpeg", "png"],
            help="A clear, flat photo of the college ID card. The face on the card will be used for recognition.",
            label_visibility="collapsed"
        )

    # ── OCR + Face extraction on upload ──────────────────────
    if id_card_file:
        pil_img = Image.open(id_card_file)

        with col_preview:
            st.markdown('<div class="section-header">ID Card Preview</div>', unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True)

        # Run OCR
        with st.spinner("🔍 Running OCR on ID card..."):
            ocr_result = extract_id_card_info(pil_img)

        # Show raw OCR text in expander for debugging
        with st.expander("🔎 View Raw OCR Text (for debugging)"):
            st.code(ocr_result["raw_text"], language=None)

        # ── Step 2: Verify & Edit extracted info ─────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Step 2 — Verify Extracted Details</div>', unsafe_allow_html=True)
        st.info("ℹ️ OCR may not be 100% accurate. Please verify and correct the fields below before saving.")

        col1, col2, col3 = st.columns(3)

        with col1:
            extracted_name = st.text_input(
                "Student Name",
                value=ocr_result["name"],
                placeholder="e.g. Naman Ghoti"
            )
        with col2:
            extracted_roll = st.text_input(
                "Roll Number",
                value=ocr_result["roll_no"],
                placeholder="e.g. B21CE001"
            )
        with col3:
            branch = st.selectbox(
                "Branch",
                [
                    "Civil Engineering",
                    "Computer Science & Engineering",
                    "Electronics & Communication",
                    "Electrical Engineering",
                    "Mechanical Engineering",
                    "Chemical Engineering",
                    "Architecture",
                    "Other"
                ]
            )

        # ── Step 3: Extract face encoding from card ───────────
        st.markdown("---")
        st.markdown('<div class="section-header">Step 3 — Face Encoding from ID Card Photo</div>', unsafe_allow_html=True)

        face_status_placeholder = st.empty()

        with st.spinner("👤 Detecting face on ID card..."):
            try:
                encoding_list = extract_face_encoding(pil_img)
                face_status_placeholder.markdown(
                    '<div class="success-box">✅ Face detected and encoded successfully from ID card photo.</div>',
                    unsafe_allow_html=True
                )
                face_ok = True
            except NoFaceDetectedError as e:
                face_status_placeholder.markdown(
                    f'<div class="error-box">❌ No Face Found: {e}</div>',
                    unsafe_allow_html=True
                )
                face_ok = False
            except MultipleFacesDetectedError as e:
                face_status_placeholder.markdown(
                    f'<div class="error-box">❌ Multiple Faces: {e}</div>',
                    unsafe_allow_html=True
                )
                face_ok = False

        # ── Step 4: Confirm & Save ────────────────────────────
        st.markdown("---")
        confirm_col, _ = st.columns([1, 2])

        with confirm_col:
            register_btn = st.button(
                "✅ Confirm & Register Student",
                use_container_width=True,
                disabled=not face_ok
            )

        if register_btn:
            if not extracted_name.strip():
                st.error("❌ Student name cannot be empty.")
            elif not extracted_roll.strip():
                st.error("❌ Roll number cannot be empty.")
            else:
                student_doc = {
                    "student_name" : extracted_name.strip(),
                    "roll_number"  : extracted_roll.strip().upper(),
                    "branch"       : branch,
                    "face_encoding": encoding_list,
                    "registered_at": datetime.now()
                }
                try:
                    result = students_col.insert_one(student_doc)
                    st.balloons()
                    st.success(f"🎉 **{extracted_name}** ({extracted_roll}) registered successfully!")
                    st.caption(f"MongoDB _id: `{result.inserted_id}`")
                except errors.DuplicateKeyError:
                    st.warning(f"⚠️ Roll number **{extracted_roll}** is already registered. No duplicate created.")
                except Exception as e:
                    st.error(f"❌ Database error: {e}")


# ════════════════════════════════════════════════════════════
# PAGE 2: MARK DAILY ATTENDANCE
# ════════════════════════════════════════════════════════════

elif page == "📷 Mark Attendance":
    st.title("📷 Mark Daily Attendance")
    st.markdown("Upload a **wide-angle classroom photo**. The system will identify all registered students automatically.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Attendance Settings</div>', unsafe_allow_html=True)

        subject_code = st.text_input(
            "Subject Code",
            placeholder="e.g. CE301, CS201",
            help="Enter the subject code for today's class."
        ).strip().upper()

        classroom_file = st.file_uploader(
            "Upload Classroom Photo",
            type=["jpg", "jpeg", "png"],
            help="A wide-angle photo of the classroom with students seated."
        )

    with col_right:
        if classroom_file:
            st.markdown('<div class="section-header">Classroom Photo Preview</div>', unsafe_allow_html=True)
            st.image(classroom_file, use_column_width=True)

    if classroom_file and subject_code:
        st.markdown("---")
        process_btn = st.button("🚀 Process Attendance", use_container_width=False)

        if process_btn:
            pil_classroom = Image.open(classroom_file)

            with st.spinner("🔄 Detecting and matching faces... This may take 30–60 seconds."):
                summary = process_attendance(pil_classroom, subject_code)

            if "error" in summary:
                st.error(f"❌ {summary['error']}")
            else:
                # ── Result metrics ────────────────────────────
                st.markdown("### 📊 Processing Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Faces Detected",  summary["total_detected"])
                m2.metric("✅ Matched",       summary["matched"])
                m3.metric("❓ Unknown",        summary["unknown"])
                m4.metric("💾 Records Saved", summary["inserted"])

                if summary["duplicates"] > 0:
                    st.info(f"ℹ️ {summary['duplicates']} student(s) were already marked present today — skipped.")

                # ── Attendance table ──────────────────────────
                st.markdown("### 📋 Today's Attendance — " + subject_code)
                today_str = datetime.now().strftime("%Y-%m-%d")
                records   = list(attendance_col.find(
                    {"date": today_str, "subject_code": subject_code},
                    {"_id": 0}
                ))

                if records:
                    df = pd.DataFrame(records)
                    col_order = ["roll_number", "student_name", "subject_code", "date", "timestamp", "status"]
                    df = df[[c for c in col_order if c in df.columns]]
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
                    df = df.sort_values("roll_number").reset_index(drop=True)
                    df.index += 1
                    st.dataframe(df, use_container_width=True)
                    st.success(f"✅ Total Present: **{len(df)}** student(s)")
                else:
                    st.warning("No attendance records found for today.")
    elif classroom_file and not subject_code:
        st.warning("⚠️ Please enter a Subject Code before processing.")


# ════════════════════════════════════════════════════════════
# PAGE 3: VIEW RECORDS
# ════════════════════════════════════════════════════════════

elif page == "📊 View Records":
    st.title("📊 Attendance Records")

    tab1, tab2 = st.tabs(["📅 By Date & Subject", "🎓 Registered Students"])

    with tab1:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_date = st.date_input("Date", value=datetime.now())
        with fc2:
            filter_subject = st.text_input("Subject Code (leave blank for all)").strip().upper()
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍 Search Records", use_container_width=True)

        if search_btn:
            query = {"date": filter_date.strftime("%Y-%m-%d")}
            if filter_subject:
                query["subject_code"] = filter_subject

            records = list(attendance_col.find(query, {"_id": 0}))
            if records:
                df = pd.DataFrame(records)
                col_order = ["roll_number", "student_name", "subject_code", "date", "timestamp", "status"]
                df = df[[c for c in col_order if c in df.columns]]
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
                df = df.sort_values(["subject_code", "roll_number"]).reset_index(drop=True)
                df.index += 1
                st.dataframe(df, use_container_width=True)
                st.info(f"Total records: **{len(df)}**")
            else:
                st.warning("No records found for the selected filters.")

    with tab2:
        students = list(students_col.find({}, {"_id": 0, "face_encoding": 0}))
        if students:
            df_s = pd.DataFrame(students)
            df_s["registered_at"] = pd.to_datetime(df_s["registered_at"]).dt.strftime("%d %b %Y %H:%M")
            df_s = df_s.sort_values("roll_number").reset_index(drop=True)
            df_s.index += 1
            st.dataframe(df_s, use_container_width=True)
            st.info(f"Total registered: **{len(df_s)}** student(s)")
        else:
            st.warning("No students registered yet.")
