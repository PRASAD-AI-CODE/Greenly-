import streamlit as st
import tensorflow as tf
import numpy as np
import json
import sqlite3
import hashlib
from datetime import datetime
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Greenly",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #9aa0a6;
    margin-bottom: 25px;
}

.card {
    padding: 22px;
    border-radius: 15px;
    background: #1e1f26;
    border: 1px solid #30313a;
    margin-bottom: 20px;
}

.success-card {
    padding: 20px;
    border-radius: 15px;
    background: #123b2a;
    border: 1px solid #247a52;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    background: #162d46;
    border: 1px solid #28547d;
}

.warning-card {
    padding: 20px;
    border-radius: 15px;
    background: #403c13;
    border: 1px solid #777021;
}

.metric-box {
    padding: 18px;
    border-radius: 12px;
    background: #20212a;
    text-align: center;
    border: 1px solid #343640;
}

.metric-number {
    font-size: 28px;
    font-weight: 700;
}

.metric-label {
    color: #a0a0a0;
}


.login-card {
    max-width: 520px;
    margin: 60px auto;
    padding: 35px;
    border-radius: 20px;
    background: #1e1f26;
    border: 1px solid #343640;
}
.user-badge {
    padding: 10px 12px;
    border-radius: 10px;
    background: #162d46;
    border: 1px solid #28547d;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model_89.keras")


@st.cache_data
def load_classes():
    with open("class_names_89.json", "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
class_names = load_classes()


# =========================================================
# DATABASE + AUTHENTICATION
# =========================================================
DB_FILE = "plantcare.db"


def get_db():
    return sqlite3.connect(DB_FILE)


def init_database():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            filename TEXT,
            image_data BLOB,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migrate an existing database created by an older Greenly version.
    # This adds image storage without deleting any existing prediction records.
    columns = [row[1] for row in cur.execute("PRAGMA table_info(predictions)").fetchall()]
    if "image_data" not in columns:
        cur.execute("ALTER TABLE predictions ADD COLUMN image_data BLOB")

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(name, username, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (
                name.strip(),
                username.strip().lower(),
                hash_password(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_db()
    row = conn.execute(
        "SELECT id, name, username FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password)),
    ).fetchone()
    conn.close()
    return row


def save_prediction(user_id, disease, confidence, filename, image_bytes=None):
    conn = get_db()
    conn.execute(
        """
        INSERT INTO predictions
        (user_id, disease, confidence, filename, image_data, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            disease,
            float(confidence),
            filename,
            sqlite3.Binary(image_bytes) if image_bytes else None,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def get_predictions(user_id):
    conn = get_db()
    rows = conn.execute(
        """
        SELECT disease, confidence, filename, predicted_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 50
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def clear_predictions(user_id):
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_admin_users():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, username, created_at FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def get_all_predictions():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT predictions.id, users.name, users.username, predictions.disease,
               predictions.confidence, predictions.filename, predictions.image_data,
               predictions.predicted_at
        FROM predictions
        LEFT JOIN users ON predictions.user_id = users.id
        ORDER BY predictions.id DESC
        """
    ).fetchall()
    conn.close()
    return rows


init_database()


# =========================================================
# LOGIN / REGISTER
# =========================================================
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
    st.session_state["user_name"] = None
    st.session_state["username"] = None
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False


def show_auth_page():
    st.markdown(
        """
        <div style="
            max-width: 760px;
            margin: 55px auto 20px auto;
            text-align: center;
        ">
            <div style="font-size: 60px;">🌱</div>
            <h1 style="font-size: 44px; margin-bottom: 5px;">Greenly</h1>
            <p style="font-size: 19px; color: #9aa0a6;">
                AI-Powered Plant Disease Detection System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])

    with center:
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Create Account"])

        with tab_login:
            st.markdown("### Welcome back 👋")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "🔐 Login",
                    use_container_width=True,
                )

            if submitted:
                if not username or not password:
                    st.error("Please enter your username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state["user_id"] = user[0]
                        st.session_state["user_name"] = user[1]
                        st.session_state["username"] = user[2]
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_register:
            st.markdown("### Create your Greenly account 🌱")
            with st.form("register_form"):
                name = st.text_input("Full Name")
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                )
                registered = st.form_submit_button(
                    "📝 Create Account",
                    use_container_width=True,
                )

            if registered:
                if not name or not new_username or not new_password:
                    st.error("Please fill in all fields.")
                elif len(new_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    created = create_user(name, new_username, new_password)
                    if created:
                        st.success("Account created successfully. Please login.")
                    else:
                        st.error("That username already exists.")


if st.session_state["user_id"] is None:
    show_auth_page()
    st.stop()


# =========================================================
# IMAGE QUALITY CHECK
# =========================================================
def check_image_quality(image):
    """Check whether a captured/uploaded image is suitable for prediction."""
    image = image.convert("RGB")

    width, height = image.size
    issues = []

    # Very small images do not provide enough visual detail.
    if width < 300 or height < 300:
        issues.append("The image resolution is too low. Please take a closer photo of the leaf.")

    gray = np.array(image.convert("L").resize((256, 256)), dtype=np.float32)

    # Average brightness check.
    brightness = float(gray.mean())
    if brightness < 35:
        issues.append("The photo is too dark. Move to a brighter area and retake it.")
    elif brightness > 225:
        issues.append("The photo is too bright. Avoid direct glare and retake it.")

    # Low contrast check.
    contrast = float(gray.std())
    if contrast < 18:
        issues.append("The image has very low contrast. Make sure the leaf is clearly visible.")

    # Lightweight blur estimate using image gradients; no extra package is needed.
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float(dx.var() + dy.var())
    if sharpness < 20:
        issues.append("The photo appears blurry. Hold the camera steady and retake it.")

    return len(issues) == 0, issues


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_disease(image):

    
    image = image.convert("RGB")
    image = image.resize((224, 224))

    # Keep pixel values in 0-255 because the model
    # already applies MobileNetV2 preprocessing
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:
        results.append(
            (class_names[index], float(predictions[index]) * 100)
        )

    return results


# =========================================================
# DISEASE INFORMATION
# =========================================================
def get_guidance(disease):

    disease_lower = disease.lower()

    if "healthy" in disease_lower:
        return (
            "Healthy Plant 🌿",
            "The model classified this leaf as healthy.",
            "Continue regular watering, adequate sunlight and good plant care.",
            "Monitor the plant regularly for changes in leaf color, spots or damage."
        )

    if "blight" in disease_lower:
        return (
            "Possible Blight 🦠",
            "The model detected a class associated with blight.",
            "Remove severely affected leaves and maintain good airflow around the plant.",
            "Avoid prolonged leaf wetness and monitor nearby plants for similar symptoms."
        )

    if "mildew" in disease_lower:
        return (
            "Possible Mildew 🦠",
            "The model detected a mildew-related class.",
            "Improve ventilation and avoid excessive moisture around the leaves.",
            "Regularly inspect new growth and remove severely affected material."
        )

    if "rust" in disease_lower:
        return (
            "Possible Rust 🦠",
            "The model detected a rust-related class.",
            "Remove affected leaves where appropriate and maintain good air circulation.",
            "Monitor the plant regularly for new symptoms."
        )

    if "spot" in disease_lower:
        return (
            "Possible Leaf Spot 🦠",
            "The model detected a leaf-spot-related class.",
            "Remove severely affected leaves and avoid unnecessary moisture on foliage.",
            "Keep the growing area clean and monitor surrounding plants."
        )

    if "virus" in disease_lower:
        return (
            "Possible Viral Disease 🦠",
            "The model detected a virus-related disease class.",
            "Separate affected plants where practical and control insect vectors.",
            "For confirmation and treatment decisions, consult an agricultural expert."
        )

    return (
        "Disease-related Class Detected 🦠",
        "The AI model detected a disease-related class.",
        "Inspect the plant carefully and maintain appropriate watering, sunlight and airflow.",
        "For accurate diagnosis and treatment, consult a local agricultural expert."
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("# 🌱 Greenly")

    st.markdown(
        "**AI-Powered Plant Disease Detection**"
    )

    st.markdown(
        f"""
        <div class="user-badge">
            👤 <b>{st.session_state["user_name"]}</b><br>
            <small>@{st.session_state["username"]}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["user_id"] = None
        st.session_state["user_name"] = None
        st.session_state["username"] = None
        st.session_state["admin_authenticated"] = False
        st.session_state.pop("results", None)
        st.session_state.pop("guidance", None)
        st.rerun()

    st.divider()

    page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Disease Prediction",
        "🕘 Prediction History",
        "📈 Analytics",
        "📊 Model Information",
        "ℹ️ About Project",
        "🔐 Admin Dashboard"
    ]
)

    st.divider()

    st.markdown("### 🧠 Technology")

    st.write("TensorFlow + Keras")
    st.write("MobileNetV2")
    st.write("Streamlit")

    st.divider()

    st.caption("Greenly • Final Year Project")


# =========================================================
# ADMIN DASHBOARD
# =========================================================
if page == "🔐 Admin Dashboard":

    st.markdown(
        '<div class="main-title">🔐 Admin Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">View Greenly users and prediction records.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state["admin_authenticated"]:
        st.info("Admin access is required to view the database.")
        admin_password = st.text_input("Admin Password", type="password")
        if st.button("🔓 Open Admin Dashboard", use_container_width=True):
            try:
                expected_password = st.secrets["ADMIN_PASSWORD"]
            except Exception:
                expected_password = None

            if not expected_password:
                st.error("Admin access is not configured. Add ADMIN_PASSWORD to Streamlit Secrets.")
            elif admin_password == expected_password:
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect admin password.")
    else:
        if st.button("🔒 Lock Admin Dashboard"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

        users = get_admin_users()
        predictions = get_all_predictions()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("👥 Total Users", len(users))
        with c2:
            st.metric("🌿 Total Predictions", len(predictions))

        st.divider()
        st.markdown("### 👥 Registered Users")
        if users:
            import pandas as pd
            users_df = pd.DataFrame(
                users, columns=["ID", "Name", "Username", "Created At"]
            )
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users registered yet.")

        st.divider()
        st.markdown("### 🌿 All Predictions")
        if predictions:
            import pandas as pd
            predictions_df = pd.DataFrame(
                [row[:6] + (row[7],) for row in predictions],
                columns=["ID", "Name", "Username", "Disease", "Confidence", "Filename", "Predicted At"],
            )
            predictions_df["Confidence"] = predictions_df["Confidence"].map(lambda x: f"{x:.2f}%")
            st.dataframe(predictions_df, use_container_width=True, hide_index=True)

            st.markdown("### 🖼️ Submitted Images")
            st.caption("Images are stored in the predictions database and are visible only in this protected Admin Dashboard.")

            for row in predictions:
                prediction_id, name, username, disease, confidence, filename, image_data, predicted_at = row
                with st.expander(
                    f"#{prediction_id} • {name} (@{username}) • {disease} • {confidence:.2f}%"
                ):
                    info_col, image_col = st.columns([1, 1])

                    with info_col:
                        st.write(f"**User:** {name} (@{username})")
                        st.write(f"**Disease:** {disease}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        st.write(f"**Filename:** {filename or 'N/A'}")
                        st.write(f"**Predicted At:** {predicted_at}")

                    with image_col:
                        if image_data:
                            st.image(
                                image_data,
                                caption=f"Submitted image • {filename or 'image'}",
                                use_container_width=True,
                            )
                        else:
                            st.info("No image was stored for this older prediction.")
        else:
            st.info("No predictions recorded yet.")


# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":

    # Hero section
    st.html("""
    <div style="
        padding: 45px 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #123524, #162d46);
        border: 1px solid #315a48;
        margin-bottom: 30px;
    ">
        <h1 style="font-size: 48px; margin-bottom: 10px;">
            🌱 Welcome to Greenly
        </h1>

        <h3 style="font-weight: 400; color: #c8d8d0;">
            AI-Powered Plant Disease Detection System
        </h3>

        <p style="font-size: 18px; color: #b8c5c0; max-width: 850px;">
            Upload a plant leaf image and let our deep learning model
            analyze it to identify the most likely disease class,
            confidence score, and useful plant-care guidance.
        </p>
    </div>
    """)


    # Statistics
    st.markdown("## 📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-number">89</div>
            <div class="metric-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-number">89</div>
            <div class="metric-label">Supported Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-number">5</div>
            <div class="metric-label">Training Epochs</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-number">AI</div>
            <div class="metric-label">Deep Learning</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # Main features
    st.markdown("## 🚀 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>📷 Image Analysis</h2>
            <p>
            Upload a clear image of a plant leaf and the AI model
            analyzes its visual characteristics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h2>🧠 AI Prediction</h2>
            <p>
            MobileNetV2 analyzes the image and predicts the most
            likely plant disease class.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h2>📊 Confidence Score</h2>
            <p>
            The system displays the prediction confidence along
            with the top three possible classes.
            </p>
        </div>
        """, unsafe_allow_html=True)


    # How it works
    st.markdown("## 🔄 How Greenly Works")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="card">
            <h2>1️⃣</h2>
            <h3>Upload</h3>
            <p>Upload an image or take a real-time photo.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h2>2️⃣</h2>
            <h3>Process</h3>
            <p>The image is resized and normalized.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h2>3️⃣</h2>
            <h3>Predict</h3>
            <p>MobileNetV2 performs disease classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">
            <h2>4️⃣</h2>
            <h3>Result</h3>
            <p>View predictions, confidence and guidance.</p>
        </div>
        """, unsafe_allow_html=True)


    # Technology
    st.markdown("## 🛠️ Technologies Used")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("🐍 Python")

    with col2:
        st.info("🧠 TensorFlow / Keras")

    with col3:
        st.info("📱 MobileNetV2")

    with col4:
        st.info("🌐 Streamlit")


    # Project objective
    st.markdown("## 🎯 Project Objective")

    st.markdown("""
    <div class="info-card">

    <h3>Why Greenly?</h3>

    <p>
    Plant diseases can affect crop health and productivity.
    Greenly demonstrates how deep learning and image
    classification can assist with preliminary identification
    of plant diseases from leaf images.
    </p>

    <p>
    The system provides a quick, interactive way to analyze
    plant images and display possible disease classes.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # CTA
    st.markdown("<br>", unsafe_allow_html=True)

    st.html("""
<div style="
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #123d2f, #163b55);
    border: 1px solid #315a48;
    text-align: center;
    margin-top: 20px;
">

<h2 style="color: white;">
🌱 Ready to analyze a plant?
</h2>

<p style="color: #c8d8d0; font-size: 16px;">
Open <b>🔍 Disease Prediction</b> from the sidebar
and upload a leaf image.
</p>

</div>
""")


# =========================================================
# DISEASE PREDICTION
# =========================================================
elif page == "🔍 Disease Prediction":

    st.markdown(
        '<div class="main-title">🔍 Disease Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload a clear plant leaf image for AI analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    # Choose between a normal upload and a live camera capture.
    upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📷 Take Photo"])

    uploaded_file = None
    camera_file = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "📷 Upload Leaf Image",
            type=["jpg", "jpeg", "png", "webp", "bmp"]
        )

    with camera_tab:
        st.info("Allow camera access in your browser, position the plant leaf clearly, and take a photo.")
        camera_file = st.camera_input("📷 Take a real-time plant photo")

    selected_file = camera_file if camera_file is not None else uploaded_file

    if selected_file is not None:

        image = Image.open(selected_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Camera Photo" if camera_file is not None else "Uploaded Plant Leaf",
                use_container_width=True
            )

        with col2:
            st.markdown("### 🧠 AI Verification")

            button_label = (
                "📷 Verify Captured Photo"
                if camera_file is not None
                else "🔍 Predict Disease"
            )

            if st.button(button_label, use_container_width=True):

                quality_ok, quality_issues = check_image_quality(image)

                if not quality_ok:
                    st.warning("⚠️ Photo quality needs improvement before AI verification.")
                    for issue in quality_issues:
                        st.write(f"• {issue}")
                    st.info("Please retake the photo and make sure the plant leaf is well lit, in focus, and fills a good part of the frame.")
                    st.stop()

                st.success("✅ Photo quality looks good. Starting AI verification...")

                with st.spinner("AI is analyzing the leaf..."):
                    results = predict_disease(image)

                disease = results[0][0]
                confidence = results[0][1]

                title, description, treatment, prevention = get_guidance(disease)

                st.markdown(
                    f"""
                    <div class="success-card">
                    <h3>🌿 Detected Class</h3>
                    <h2>{disease}</h2>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.session_state["results"] = results
                st.session_state["guidance"] = (
                    title,
                    description,
                    treatment,
                    prevention
                )

                save_prediction(
                    st.session_state["user_id"],
                    disease,
                    confidence,
                    getattr(selected_file, "name", "camera_photo.jpg"),
                    selected_file.getvalue(),
                )


        # =================================================
        # RESULTS
        # =================================================
        if "results" in st.session_state:

            results = st.session_state["results"]

            st.divider()

            st.markdown("## 📊 Top 3 Predictions")

            for disease, probability in results:

                st.write(
                    f"**{disease} — {probability:.2f}%**"
                )

                st.progress(
                    min(int(probability), 100)
                )

            # =================================================
            # GUIDANCE
            # =================================================
            title, description, treatment, prevention = \
                st.session_state["guidance"]

            st.divider()

            st.markdown("## 🌱 Plant Guidance")

            st.markdown(
                f"""
                <div class="info-card">
                <h3>{title}</h3>
                <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 💊 General Care")

                st.write(treatment)

            with col2:

                st.markdown("### 🛡️ Prevention")

                st.write(prevention)

            st.markdown(
                """
                <div class="warning-card">
                ⚠️ <b>Important:</b>
                This AI prediction is for educational and preliminary
                identification purposes. For confirmation and treatment
                decisions, consult a qualified agricultural expert.
                </div>
                """,
                unsafe_allow_html=True
            )


elif page == "🕘 Prediction History":

    st.markdown(
        '<div class="main-title">🕘 Prediction History</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Your saved plant disease predictions.</div>',
        unsafe_allow_html=True,
    )

    history = get_predictions(st.session_state["user_id"])

    if not history:
        st.info(
            "📭 No predictions saved yet. Go to Disease Prediction and analyze a leaf image."
        )
    else:
        st.markdown(f"### 📋 {len(history)} recent prediction(s)")

        for disease, confidence, filename, predicted_at in history:
            st.markdown(
                f"""
                <div class="card">
                    <h3>🌿 {disease}</h3>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                    <p><b>Image:</b> {filename}</p>
                    <p><b>Date & Time:</b> {predicted_at}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        if st.button("🗑️ Clear My Prediction History"):
            clear_predictions(st.session_state["user_id"])
            st.success("Prediction history cleared.")
            st.rerun()


# =========================================================
# ANALYTICS
# =========================================================
elif page == "📈 Analytics":

    st.markdown(
        '<div class="main-title">📈 Prediction Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">A summary of your Greenly activity.</div>',
        unsafe_allow_html=True,
    )

    history = get_predictions(st.session_state["user_id"])

    if not history:
        st.info("📊 Make a prediction first to see your analytics.")
    else:
        total = len(history)
        healthy_count = sum(1 for row in history if "healthy" in row[0].lower())
        diseased_count = total - healthy_count
        avg_confidence = sum(row[1] for row in history) / total

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-number">{total}</div>'
                '<div class="metric-label">Total Predictions</div></div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-number">{healthy_count}</div>'
                '<div class="metric-label">Healthy</div></div>',
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-number">{diseased_count}</div>'
                '<div class="metric-label">Disease Classes</div></div>',
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-number">{avg_confidence:.1f}%</div>'
                '<div class="metric-label">Average Confidence</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("### 🦠 Detected Disease Classes")

        counts = {}
        for disease, _, _, _ in history:
            counts[disease] = counts.get(disease, 0) + 1

        chart_data = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        st.bar_chart(chart_data)

        st.markdown("### 🎯 Confidence by Recent Prediction")

        confidence_data = {
            f"{i + 1}. {row[0][:25]}": row[1]
            for i, row in enumerate(history[:10])
        }
        st.bar_chart(confidence_data)


elif page == "📊 Model Information":

    st.markdown(
        '<div class="main-title">📊 Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Technical details of the Greenly model'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🤖 Model")

        st.markdown("""
        <div class="card">

        **Architecture:** MobileNetV2

        **Framework:** TensorFlow / Keras

        **Input:** Plant leaf image

        **Image Size:** 224 × 224 pixels

        **Output:** Disease classification

        **Classes:** 89

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("### 📈 Training")

        st.markdown("""
        <div class="card">

        **Training Epochs:** 5

        **Training Metrics:** Available from the 89-class training run

        **Validation Metrics:** Not hard-coded from the previous 71-class model

        **Loss Function:** Sparse Categorical Crossentropy

        **Optimizer:** Adam

        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 Prediction Process")

    st.code("""
Leaf Image
    ↓
Image Resize (224 × 224)
    ↓
MobileNetV2 Preprocessing
    ↓
MobileNetV2
    ↓
Feature Extraction
    ↓
Disease Classification
    ↓
Top 3 Predictions
    ↓
Confidence Score
    """)


# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About Greenly</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎯 Project Objective

    Greenly is a deep learning based plant disease detection
    system designed to identify possible diseases from plant leaf images.

    The system uses image classification to analyze visual patterns
    in leaves and predict the most likely disease class.

    ### 🧠 Artificial Intelligence

    The project uses **MobileNetV2**, a lightweight deep learning
    architecture suitable for image classification.

    ### 🛠️ Technologies Used

    - Python
    - TensorFlow
    - Keras
    - MobileNetV2
    - NumPy
    - Pillow
    - Streamlit

    ### 📌 Main Features

    ✅ Leaf image upload

    ✅ AI-based disease prediction

    ✅ Confidence percentage

    ✅ Top 3 predictions

    ✅ Disease guidance

    ✅ Prevention information

    ✅ Interactive web interface

    ### 🌱 Future Scope

    The project can be extended with:

    - Mobile application
    - Weather-based disease prediction
    - Farmer dashboard
    - Disease history
    - Multilingual support
    - Treatment recommendation system
    - Cloud deployment
    """)

    st.success(
        "Greenly — AI-powered plant disease detection 🌱"
    )
