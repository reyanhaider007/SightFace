from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

import cv2
import numpy as np
import base64
import os
import sqlite3
from datetime import datetime


# =========================================================
# APP
# =========================================================

app = FastAPI()


# =========================================================
# PATHS
# =========================================================

DATA_DIR = "face_data"
DB_FILE = "attendance.db"
MODEL_FILE = os.path.join(DATA_DIR, "face_model.yml")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================

def get_db():

    conn = sqlite3.connect(DB_FILE)

    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(student_id, date)
        )
    """)

    conn.commit()
    conn.close()


init_database()


# =========================================================
# FACE DETECTOR
# =========================================================

# Use the Haar Cascade XML file stored in the SightFace project folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_detector.empty():

    raise RuntimeError(
        "Could not load Haar Cascade."
    )

print("Face detector loaded successfully!")


# =========================================================
# FACE RECOGNIZER
# =========================================================

if not hasattr(cv2, "face"):

    raise RuntimeError(
        "cv2.face is missing. "
        "Install opencv-contrib-python."
    )


recognizer = cv2.face.LBPHFaceRecognizer_create()

print("LBPH recognizer loaded!")


# =========================================================
# REQUEST MODELS
# =========================================================

class RegisterRequest(BaseModel):

    student_id: str
    name: str
    images: List[str]


class RecognizeRequest(BaseModel):

    image: str


# =========================================================
# IMAGE DECODER
# =========================================================

def decode_image(data):

    if "," in data:

        data = data.split(",", 1)[1]

    image_bytes = base64.b64decode(data)

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Could not decode image."
        )

    return image


# =========================================================
# FACE DETECTION
# =========================================================

def get_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) == 0:

        raise ValueError(
            "No face detected. Please look at the camera."
        )

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    face = gray[
        y:y + h,
        x:x + w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    return face


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    conn = get_db()

    students = conn.execute(
        "SELECT * FROM students"
    ).fetchall()

    conn.close()

    faces = []
    labels = []

    for student in students:

        internal_id = student["id"]

        folder = os.path.join(
            DATA_DIR,
            f"user_{internal_id}"
        )

        if not os.path.exists(folder):

            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(".jpg"):

                continue

            path = os.path.join(
                folder,
                filename
            )

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:

                continue

            image = cv2.resize(
                image,
                (200, 200)
            )

            faces.append(image)
            labels.append(internal_id)

    if not faces:

        print("No training images found.")

        return False

    recognizer.train(
        faces,
        np.array(labels)
    )

    recognizer.write(
        MODEL_FILE
    )

    print(
        f"Model trained with {len(faces)} samples."
    )

    return True


# =========================================================
# LOAD EXISTING MODEL
# =========================================================

if os.path.exists(MODEL_FILE):

    try:

        recognizer.read(
            MODEL_FILE
        )

        print("Existing model loaded.")

    except Exception as error:

        print(
            "Model loading error:",
            error
        )


# =========================================================
# MAIN PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>SightFace Attendance</title>


<style>

/* =====================================================
   RESET
===================================================== */

* {
    box-sizing: border-box;
}


html,
body {

    margin: 0;

    width: 100%;
    height: 100%;

    overflow: hidden;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: white;

    background:

        radial-gradient(
            circle at 10% 10%,
            #173b78,
            transparent 35%
        ),

        radial-gradient(
            circle at 90% 10%,
            #432174,
            transparent 35%
        ),

        #050b18;
}


/* =====================================================
   PAGE
===================================================== */

.page {

    width: 100%;
    height: 100vh;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: flex-start;

    padding: 12px 18px;
}


/* =====================================================
   HEADER
===================================================== */

.header {

    text-align: center;

    margin-bottom: 10px;

    flex-shrink: 0;
}


.logo {

    font-size: 30px;

    font-weight: 700;

    letter-spacing: -1px;
}


.logo span {

    color: #60a5fa;
}


.subtitle {

    color: #94a3b8;

    font-size: 12px;

    margin-top: 3px;
}


/* =====================================================
   MAIN CARD
===================================================== */

.card {

    width: min(1100px, 96vw);

    height: min(
        540px,
        calc(100vh - 88px)
    );

    display: flex;

    gap: 15px;

    padding: 14px;

    border-radius: 24px;

    background:
        rgba(15, 23, 42, .72);

    border:
        1px solid rgba(255,255,255,.1);

    backdrop-filter:
        blur(30px);

    box-shadow:
        0 30px 80px rgba(0,0,0,.4);

    flex-shrink: 1;

    min-height: 0;
}


/* =====================================================
   CAMERA
===================================================== */

.camera {

    flex: 1;

    min-width: 0;
    min-height: 0;

    position: relative;

    overflow: hidden;

    border-radius: 20px;

    background: #020617;
}


video {

    width: 100%;
    height: 100%;

    display: block;

    object-fit: cover;

    background: #020617;
}


.camera-label {

    position: absolute;

    top: 15px;
    left: 15px;

    padding: 8px 13px;

    border-radius: 30px;

    background:
        rgba(0,0,0,.55);

    backdrop-filter:
        blur(10px);

    font-size: 12px;
}


/* =====================================================
   CONTROLS
===================================================== */

.controls {

    width: 300px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    gap: 9px;

    padding: 16px;

    border-radius: 20px;

    background:
        rgba(30,41,59,.5);

    min-width: 0;
}


.mode-title {

    text-align: center;

    font-size: 16px;

    font-weight: 600;

    color: #e2e8f0;

    margin-bottom: 2px;
}


.mode-subtitle {

    text-align: center;

    font-size: 11px;

    color: #64748b;

    margin-bottom: 3px;
}


/* =====================================================
   INPUTS
===================================================== */

input {

    width: 100%;

    padding: 12px;

    border-radius: 13px;

    border:
        1px solid #334155;

    background:
        #0f172a;

    color: white;

    outline: none;

    font-size: 13px;
}


input:focus {

    border-color:
        #3b82f6;
}


input::placeholder {

    color: #64748b;
}


/* =====================================================
   BUTTONS
===================================================== */

button {

    width: 100%;

    padding: 12px;

    border: none;

    border-radius: 13px;

    color: white;

    font-weight: 600;

    cursor: pointer;

    transition: .2s;

    font-size: 13px;
}


button:hover {

    transform:
        translateY(-2px);

    filter:
        brightness(1.08);
}


button:active {

    transform:
        scale(.98);
}


/* =====================================================
   BUTTON COLORS
===================================================== */

.open {

    background:
        linear-gradient(
            135deg,
            #06b6d4,
            #2563eb
        );
}


.register {

    background:
        linear-gradient(
            135deg,
            #22c55e,
            #14b8a6
        );
}


.attendance {

    background:
        linear-gradient(
            135deg,
            #8b5cf6,
            #6366f1
        );
}


.stop {

    background:
        #334155;
}


.dashboard {

    background:
        linear-gradient(
            135deg,
            #f59e0b,
            #ef4444
        );
}


/* =====================================================
   REGISTRATION AREA
===================================================== */

.registration-area {

    display: none;

    flex-direction: column;

    gap: 8px;

    padding: 10px;

    border-radius: 15px;

    background:
        rgba(15,23,42,.8);

    border:
        1px solid rgba(255,255,255,.08);
}


.registration-title {

    text-align: center;

    color: #86efac;

    font-size: 11px;

    margin-bottom: 1px;
}


/* =====================================================
   STATUS
===================================================== */

#progress {

    color: #60a5fa;

    text-align: center;

    font-size: 11px;

    min-height: 15px;
}


.status {

    margin-top: 3px;

    padding: 12px;

    border-top:
        1px solid #334155;

    color: #cbd5e1;

    font-size: 12px;

    text-align: center;

    line-height: 1.4;
}


/* =====================================================
   MOBILE
===================================================== */

@media(max-width:750px) {

    .page {

        padding: 8px;
    }


    .header {

        margin-bottom: 6px;
    }


    .logo {

        font-size: 24px;
    }


    .subtitle {

        font-size: 10px;
    }


    .card {

        flex-direction: column;

        height:
            calc(100vh - 65px);

        padding: 9px;

        gap: 9px;
    }


    .camera {

        height: 53%;

        flex-shrink: 0;
    }


    .controls {

        width: 100%;

        height: 47%;

        display: flex;

        overflow: hidden;

        padding: 9px;

        gap: 6px;
    }


    button {

        padding: 8px;

        font-size: 11px;
    }


    input {

        padding: 8px;

        font-size: 11px;
    }


    .registration-area {

        padding: 7px;

        gap: 5px;
    }

}


/* =====================================================
   VERY SMALL HEIGHT
===================================================== */

@media(max-height:650px) {

    .header {

        margin-bottom: 5px;
    }


    .logo {

        font-size: 23px;
    }


    .subtitle {

        display: none;
    }


    .card {

        height:
            calc(100vh - 48px);
    }


    .controls {

        gap: 5px;

        padding: 9px;
    }


    button {

        padding: 8px;
    }

}

</style>

</head>


<body>


<div class="page">


<!-- =================================================
     HEADER
================================================= -->

<div class="header">

    <div class="logo">

        Sight<span>Face</span>

    </div>


    <div class="subtitle">

        AI Face Recognition Attendance System

    </div>

</div>


<!-- =================================================
     MAIN CARD
================================================= -->

<div class="card">


<!-- =================================================
     CAMERA
================================================= -->

<div class="camera">

    <video
        id="video"
        autoplay
        playsinline>
    </video>


    <div class="camera-label">

        🟢 Camera

    </div>

</div>


<!-- =================================================
     CONTROLS
================================================= -->

<div class="controls">


<div class="mode-title">

    Attendance Mode

</div>


<div class="mode-subtitle">

    Open camera and mark attendance

</div>


<!-- CAMERA -->

<button
    class="open"
    onclick="openCamera()">

    📷 Open Camera

</button>


<!-- ATTENDANCE -->

<button
    class="attendance"
    onclick="markAttendance()">

    ✓ Mark Attendance

</button>


<!-- REGISTER -->

<button
    class="register"
    onclick="showRegistration()">

    👤 Register New Student

</button>


<!-- REGISTRATION AREA -->

<div
    id="registrationArea"
    class="registration-area">


    <div class="registration-title">

        New Student Registration

    </div>


    <input
        id="studentId"
        placeholder="Student ID"
    />


    <input
        id="name"
        placeholder="Student Name"
    />


    <button
        class="register"
        onclick="registerStudent()">

        📸 Capture & Register

    </button>


</div>


<!-- DASHBOARD -->

<button
    class="dashboard"
    onclick="openDashboard()">

    📊 Attendance Dashboard

</button>


<!-- STOP -->

<button
    class="stop"
    onclick="stopCamera()">

    ■ Stop Camera

</button>


<div id="progress"></div>


<div
    id="status"
    class="status">

    Ready — Open camera to take attendance

</div>


</div>


</div>


<canvas
    id="canvas"
    style="display:none;">
</canvas>


<script>


// =====================================================
// VARIABLES
// =====================================================

let stream = null;


const video =
    document.getElementById(
        "video"
    );


const canvas =
    document.getElementById(
        "canvas"
    );


const status =
    document.getElementById(
        "status"
    );


const progress =
    document.getElementById(
        "progress"
    );


const registrationArea =
    document.getElementById(
        "registrationArea"
    );


// =====================================================
// OPEN CAMERA
// =====================================================

async function openCamera() {

    try {

        // If camera is already running
        if (stream) {

            status.innerText =
                "Camera is already open.";

            return;
        }


        stream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        width: 640,

                        height: 480

                    },

                    audio: false

                });


        video.srcObject =
            stream;


        status.innerText =
            "Camera ready. Look at the camera.";

    }


    catch(error) {

        console.error(error);


        status.innerText =
            "Camera permission denied.";

    }

}


// =====================================================
// CAPTURE IMAGE
// =====================================================

function captureImage() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {

        throw new Error(
            "Camera is not ready."
        );

    }


    canvas.width =
        video.videoWidth;


    canvas.height =
        video.videoHeight;


    const context =
        canvas.getContext(
            "2d"
        );


    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );


    return canvas.toDataURL(
        "image/jpeg",
        0.95
    );

}


// =====================================================
// SHOW / HIDE REGISTRATION
// =====================================================

function showRegistration() {

    if (
        registrationArea.style.display ===
        "flex"
    ) {

        registrationArea.style.display =
            "none";


        status.innerText =
            "Attendance mode ready.";

        return;
    }


    registrationArea.style.display =
        "flex";


    status.innerText =
        "Enter details for a new student.";

}


// =====================================================
// REGISTER STUDENT
// =====================================================

async function registerStudent() {

    const studentId =
        document
            .getElementById(
                "studentId"
            )
            .value
            .trim();


    const name =
        document
            .getElementById(
                "name"
            )
            .value
            .trim();


    if (!studentId || !name) {

        status.innerText =
            "Enter Student ID and Name.";

        return;
    }


    if (!stream) {

        status.innerText =
            "Open the camera first.";

        return;
    }


    const images = [];


    status.innerText =
        "Keep your face visible.";


    // Capture 7 samples

    for (
        let i = 0;
        i < 7;
        i++
    ) {

        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    500
                )
        );


        try {

            images.push(
                captureImage()
            );

        }

        catch(error) {

            status.innerText =
                error.message;

            return;

        }


        progress.innerText =
            "Capturing " +
            (i + 1) +
            " / 7";

    }


    progress.innerText =
        "Training face model...";


    try {

        const response =
            await fetch(
                "/register",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            student_id:
                                studentId,

                            name:
                                name,

                            images:
                                images

                        })

                }
            );


        const data =
            await response.json();


        status.innerText =
            data.message;


        progress.innerText =
            "";


        if (data.success) {

            // Hide registration
            registrationArea.style.display =
                "none";


            // Clear fields

            document
                .getElementById(
                    "studentId"
                )
                .value = "";


            document
                .getElementById(
                    "name"
                )
                .value = "";

        }

    }


    catch(error) {

        console.error(error);


        status.innerText =
            "Registration failed.";


        progress.innerText =
            "";

    }

}


// =====================================================
// MARK ATTENDANCE
// =====================================================

async function markAttendance() {

    if (!stream) {

        status.innerText =
            "Open the camera first.";

        return;
    }


    status.innerText =
        "Recognizing face...";


    progress.innerText =
        "Please look at the camera";


    try {

        // Capture ONE picture

        const image =
            captureImage();


        // Send it to recognition API

        const response =
            await fetch(
                "/recognize",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            image:
                                image

                        })

                }
            );


        const data =
            await response.json();


        status.innerText =
            data.message;


        progress.innerText =
            "";


    }


    catch(error) {

        console.error(error);


        status.innerText =
            "Attendance failed.";


        progress.innerText =
            "";

    }

}


// =====================================================
// DASHBOARD
// =====================================================

function openDashboard() {

    window.location.href =
        "/dashboard";

}


// =====================================================
// STOP CAMERA
// =====================================================

function stopCamera() {

    if (!stream) {

        status.innerText =
            "Camera is already stopped.";

        return;
    }


    stream
        .getTracks()
        .forEach(
            track =>
                track.stop()
        );


    stream = null;


    video.srcObject =
        null;


    status.innerText =
        "Camera stopped.";


    progress.innerText =
        "";

}


// =====================================================
// INITIAL STATE
// =====================================================

registrationArea.style.display =
    "none";

</script>


</body>

</html>
"""


# =========================================================
# REGISTER STUDENT API
# =========================================================

@app.post("/register")
def register_student(
    request: RegisterRequest
):

    try:

        student_id = (
            request.student_id
            .strip()
        )

        name = (
            request.name
            .strip()
        )


        # -----------------------------------------------
        # VALIDATION
        # -----------------------------------------------

        if not student_id:

            return {
                "success": False,
                "message":
                    "Student ID is required."
            }


        if not name:

            return {
                "success": False,
                "message":
                    "Student name is required."
            }


        if len(request.images) == 0:

            return {
                "success": False,
                "message":
                    "No camera images received."
            }


        # -----------------------------------------------
        # CHECK EXISTING STUDENT
        # -----------------------------------------------

        conn = get_db()


        existing = conn.execute(
            """
            SELECT * FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()


        if existing:

            conn.close()


            return {
                "success": False,
                "message":
                    "Student ID already exists."
            }


        # -----------------------------------------------
        # CREATE STUDENT
        # -----------------------------------------------

        conn.execute(
            """
            INSERT INTO students
            (student_id, name)
            VALUES (?, ?)
            """,
            (
                student_id,
                name
            )
        )


        conn.commit()


        student = conn.execute(
            """
            SELECT * FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()


        conn.close()


        internal_id = student["id"]


        # -----------------------------------------------
        # CREATE FACE FOLDER
        # -----------------------------------------------

        folder = os.path.join(
            DATA_DIR,
            f"user_{internal_id}"
        )


        os.makedirs(
            folder,
            exist_ok=True
        )


        # -----------------------------------------------
        # SAVE FACE SAMPLES
        # -----------------------------------------------

        saved = 0


        for index, image_data in enumerate(
            request.images
        ):

            try:

                image = decode_image(
                        image_data
                    )


                face = get_face(
                        image
                    )


                path = os.path.join(
                    folder,
                    f"face_{index + 1}.jpg"
                )


                cv2.imwrite(
                    path,
                    face
                )


                saved += 1


            except Exception as error:

                print(
                    "Sample error:",
                    error
                )


        # -----------------------------------------------
        # CHECK SAMPLES
        # -----------------------------------------------

        if saved < 2:

            return {
                "success": False,
                "message":
                    "Could not capture enough face samples."
            }


        # -----------------------------------------------
        # TRAIN
        # -----------------------------------------------

        trained = train_model()


        if not trained:

            return {
                "success": False,
                "message":
                    "Student saved but model training failed."
            }


        return {

            "success": True,

            "message":
                f"✓ {name} registered successfully."

        }


    except Exception as error:

        print(
            "REGISTER ERROR:",
            error
        )


        return {

            "success": False,

            "message":
                f"Registration error: {error}"

        }


# =========================================================
# RECOGNIZE + MARK ATTENDANCE
# =========================================================

@app.post("/recognize")
def recognize(
    request: RecognizeRequest
):

    try:

        # -----------------------------------------------
        # MODEL CHECK
        # -----------------------------------------------

        if not os.path.exists(
            MODEL_FILE
        ):

            return {

                "success": False,

                "message":
                    "No registered students found."

            }


        # -----------------------------------------------
        # DECODE IMAGE
        # -----------------------------------------------

        image = decode_image(
                request.image
            )


        # -----------------------------------------------
        # GET FACE
        # -----------------------------------------------

        face = get_face(
                image
            )


        # -----------------------------------------------
        # LOAD MODEL
        # -----------------------------------------------

        recognizer.read(
            MODEL_FILE
        )


        # -----------------------------------------------
        # PREDICT
        # -----------------------------------------------

        label, distance = recognizer.predict(
                face
            )


        print(
            "Face:",
            label,
            "Distance:",
            distance
        )


        # -----------------------------------------------
        # MATCH THRESHOLD
        # -----------------------------------------------

        if distance > 75:

            return {

                "success": False,

                "message":
                    "Face not recognized."

            }


        # -----------------------------------------------
        # MATCH SCORE
        # -----------------------------------------------

        match_score = max(
            0,
            min(
                100,
                round(
                    100 - distance
                )
            )
        )


        # -----------------------------------------------
        # FIND STUDENT
        # -----------------------------------------------

        conn = get_db()


        student = conn.execute(
            """
            SELECT * FROM students
            WHERE id = ?
            """,
            (int(label),)
        ).fetchone()


        if not student:

            conn.close()


            return {

                "success": False,

                "message":
                    "Registered student not found."

            }


        # -----------------------------------------------
        # DATE / TIME
        # -----------------------------------------------

        now = datetime.now()


        today = now.strftime(
                "%Y-%m-%d"
            )


        current_time = now.strftime(
                "%I:%M %p"
            )


        # -----------------------------------------------
        # CHECK ALREADY PRESENT
        # -----------------------------------------------

        existing = conn.execute(
            """
            SELECT * FROM attendance
            WHERE student_id = ?
            AND date = ?
            """,
            (
                student["student_id"],
                today
            )
        ).fetchone()


        if existing:

            conn.close()


            return {

                "success": True,

                "already_present": True,

                "message":
                    f"✓ {student['name']} "
                    f"already marked present today. "
                    f"Match: {match_score}%"

            }


        # -----------------------------------------------
        # MARK ATTENDANCE
        # -----------------------------------------------

        conn.execute(
            """
            INSERT INTO attendance
            (student_id, date, time)
            VALUES (?, ?, ?)
            """,
            (
                student["student_id"],
                today,
                current_time
            )
        )


        conn.commit()
        conn.close()


        return {

            "success": True,

            "already_present": False,

            "message":
                f"✓ {student['name']} Present | "
                f"Match: {match_score}%"

        }


    except Exception as error:

        print(
            "RECOGNITION ERROR:",
            error
        )


        return {

            "success": False,

            "message":
                f"Recognition error: {error}"

        }


# =========================================================
# DASHBOARD PAGE
# =========================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse
)
def dashboard():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
SightFace Attendance Dashboard
</title>


<style>

* {
    box-sizing: border-box;
}


html,
body {

    margin: 0;

    min-height: 100%;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: white;

    background:

        radial-gradient(
            circle at 10% 10%,
            #173b78,
            transparent 35%
        ),

        radial-gradient(
            circle at 90% 10%,
            #432174,
            transparent 35%
        ),

        #050b18;
}


.page {

    padding: 30px;

    max-width: 1200px;

    margin: auto;
}


.header {

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    margin-bottom: 25px;
}


.title {

    font-size: 30px;

    font-weight: 700;
}


.title span {

    color: #60a5fa;
}


.subtitle {

    color: #94a3b8;

    margin-top: 3px;
}


.back {

    padding: 10px 18px;

    border-radius: 12px;

    background:
        rgba(255,255,255,.08);

    color: white;

    text-decoration: none;

    border:
        1px solid rgba(255,255,255,.08);
}


.stats {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;

    margin-bottom: 20px;
}


.stat {

    padding: 22px;

    border-radius: 20px;

    background:
        rgba(30,41,59,.6);

    border:
        1px solid rgba(255,255,255,.08);

    backdrop-filter:
        blur(20px);
}


.stat-title {

    color: #94a3b8;

    font-size: 13px;
}


.stat-number {

    font-size: 35px;

    font-weight: 700;

    margin-top: 8px;
}


.present {

    color: #4ade80;
}


.percentage {

    color: #60a5fa;
}


.table-box {

    padding: 20px;

    border-radius: 20px;

    background:
        rgba(30,41,59,.6);

    border:
        1px solid rgba(255,255,255,.08);

    backdrop-filter:
        blur(20px);
}


table {

    width: 100%;

    border-collapse:
        collapse;
}


th,
td {

    padding: 14px;

    text-align: left;

    border-bottom:
        1px solid #334155;
}


th {

    color: #94a3b8;

    font-size: 12px;
}


.badge {

    padding: 5px 10px;

    border-radius: 20px;

    background:
        rgba(34,197,94,.15);

    color: #4ade80;

    font-size: 12px;
}


@media(max-width:700px) {

    .page {

        padding: 15px;
    }


    .stats {

        grid-template-columns: 1fr;
    }


    .header {

        gap: 10px;
    }


    table {

        font-size: 12px;
    }

}

</style>

</head>


<body>


<div class="page">


<!-- HEADER -->

<div class="header">


<div>

    <div class="title">

        Sight<span>Face</span>

    </div>


    <div class="subtitle">

        Attendance Dashboard

    </div>

</div>


<a
    class="back"
    href="/">

    ← Face Recognition

</a>


</div>


<!-- STATISTICS -->

<div class="stats">


<div class="stat">

    <div class="stat-title">

        Total Students

    </div>


    <div
        id="total"
        class="stat-number">

        0

    </div>

</div>


<div class="stat">

    <div class="stat-title">

        Present Today

    </div>


    <div
        id="present"
        class="stat-number present">

        0

    </div>

</div>


<div class="stat">

    <div class="stat-title">

        Today's Attendance

    </div>


    <div
        id="percentage"
        class="stat-number percentage">

        0%

    </div>

</div>


</div>


<!-- TABLE -->

<div class="table-box">


<table>


<thead>

<tr>

<th>
Student ID
</th>


<th>
Name
</th>


<th>
Present Days
</th>


<th>
Attendance %
</th>


<th>
Today
</th>

</tr>

</thead>


<tbody
    id="students">

</tbody>


</table>


</div>


</div>


<script>


async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/dashboard"
            );


        const data =
            await response.json();


        document
            .getElementById(
                "total"
            )
            .innerText =
                data.total_students;


        document
            .getElementById(
                "present"
            )
            .innerText =
                data.present_today;


        document
            .getElementById(
                "percentage"
            )
            .innerText =
                data.today_percentage +
                "%";


        const table =
            document.getElementById(
                "students"
            );


        table.innerHTML = "";


        data.students.forEach(
            student => {

                const row =
                    document.createElement(
                        "tr"
                    );


                row.innerHTML = `

                    <td>
                        ${student.student_id}
                    </td>

                    <td>
                        ${student.name}
                    </td>

                    <td>
                        ${student.present_days}
                        /
                        ${data.total_days}
                    </td>

                    <td>
                        ${student.percentage}%
                    </td>

                    <td>

                        ${
                            student.present_today

                            ?

                            '<span class="badge">Present</span>'

                            :

                            'Absent'
                        }

                    </td>

                `;


                table.appendChild(
                    row
                );

            }
        );

    }


    catch(error) {

        console.error(
            "Dashboard error:",
            error
        );

    }

}


// Initial load

loadDashboard();


// Refresh every 5 seconds

setInterval(
    loadDashboard,
    5000
);


</script>


</body>

</html>
"""


# =========================================================
# DASHBOARD API
# =========================================================

@app.get("/api/dashboard")
def dashboard_data():

    conn = get_db()


    # -----------------------------------------------
    # STUDENTS
    # -----------------------------------------------

    students = conn.execute(
        """
        SELECT *
        FROM students
        ORDER BY name
        """
    ).fetchall()


    total_students = len(students)


    # -----------------------------------------------
    # TODAY
    # -----------------------------------------------

    today = datetime.now().strftime(
            "%Y-%m-%d"
        )


    present_today = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM attendance
            WHERE date = ?
            """,
            (today,)
        ).fetchone()["count"]


    # -----------------------------------------------
    # TOTAL ATTENDANCE DAYS
    # -----------------------------------------------

    total_days = conn.execute(
            """
            SELECT COUNT(
                DISTINCT date
            )
            FROM attendance
            """
        ).fetchone()[0]


    if total_days == 0:

        total_days = 1


    students_data = []


    # -----------------------------------------------
    # EACH STUDENT
    # -----------------------------------------------

    for student in students:

        present_days = conn.execute(
                """
                SELECT COUNT(*)
                FROM attendance
                WHERE student_id = ?
                """,
                (
                    student["student_id"],
                )
            ).fetchone()[0]


        percentage = round(
                (
                    present_days /
                    total_days
                ) * 100
            )


        today_present = conn.execute(
                """
                SELECT COUNT(*)
                FROM attendance
                WHERE student_id = ?
                AND date = ?
                """,
                (
                    student["student_id"],
                    today
                )
            ).fetchone()[0] > 0


        students_data.append({

            "student_id":
                student["student_id"],

            "name":
                student["name"],

            "present_days":
                present_days,

            "percentage":
                percentage,

            "present_today":
                today_present

        })


    conn.close()


    # -----------------------------------------------
    # TODAY PERCENTAGE
    # -----------------------------------------------

    today_percentage = 0


    if total_students > 0:

        today_percentage = round(
                (
                    present_today /
                    total_students
                ) * 100
            )


    return {

        "total_students":
            total_students,

        "present_today":
            present_today,

        "today_percentage":
            today_percentage,

        "total_days":
            total_days,

        "students":
            students_data

    }