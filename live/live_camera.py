import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "models"
    / "final_model.pth"
)

LANDMARKER_PATH = (
    BASE_DIR
    / "live"
    / "models"
    / "face_landmarker.task"
)

TEMP_DIR = (
    BASE_DIR
    / "live"
    / "temp"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(
    0,
    str(BASE_DIR)
)

from src.inference import load_model


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)


# ============================================================
# ASSESSMENT STEPS
# ============================================================

STEPS = [
    {
        "name": "Neutral Face",
        "instruction": "Look straight at the camera",
        "duration": 4,
    },
    {
        "name": "Smile",
        "instruction": "SMILE naturally",
        "duration": 4,
    },
    {
        "name": "Eyebrows",
        "instruction": "RAISE BOTH EYEBROWS",
        "duration": 4,
    },
    {
        "name": "Eyes",
        "instruction": "CLOSE BOTH EYES",
        "duration": 4,
    },
    {
        "name": "Mouth",
        "instruction": "OPEN YOUR MOUTH",
        "duration": 4,
    },
]


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [
    33, 133,
    160, 159, 158,
    157, 173,
    144, 145, 153,
    154, 155,
]

RIGHT_EYE = [
    362, 263,
    387, 386, 385,
    384, 398,
    373, 374, 380,
    381, 382,
]

LEFT_EYEBROW = [
    70,
    63,
    105,
    66,
    107,
]

RIGHT_EYEBROW = [
    336,
    296,
    334,
    293,
    300,
]

MOUTH = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]


# ============================================================
# LANDMARK HELPERS
# ============================================================

def landmark_points(
    landmarks,
    indices,
    width,
    height,
):

    points = []

    for index in indices:

        lm = landmarks[index]

        x = lm.x * width
        y = lm.y * height

        points.append(
            (x, y)
        )

    return np.array(
        points,
        dtype=np.float32
    )


def center(points):

    if points is None or len(points) == 0:
        return None

    return np.mean(
        points,
        axis=0
    )


def distance(a, b):

    if a is None or b is None:
        return 0.0

    return float(
        np.linalg.norm(a - b)
    )


# ============================================================
# FACIAL MEASUREMENTS
# ============================================================

def calculate_measurements(
    landmarks,
    width,
    height,
):

    left_eye = landmark_points(
        landmarks,
        LEFT_EYE,
        width,
        height
    )

    right_eye = landmark_points(
        landmarks,
        RIGHT_EYE,
        width,
        height
    )

    left_brow = landmark_points(
        landmarks,
        LEFT_EYEBROW,
        width,
        height
    )

    right_brow = landmark_points(
        landmarks,
        RIGHT_EYEBROW,
        width,
        height
    )

    mouth = landmark_points(
        landmarks,
        MOUTH,
        width,
        height
    )

    left_eye_center = center(
        left_eye
    )

    right_eye_center = center(
        right_eye
    )

    left_brow_center = center(
        left_brow
    )

    right_brow_center = center(
        right_brow
    )

    # ========================================================
    # EYE OPENING
    # ========================================================

    left_eye_top = np.mean(
        left_eye[
            [2, 3, 4]
        ],
        axis=0
    )

    left_eye_bottom = np.mean(
        left_eye[
            [7, 8, 9]
        ],
        axis=0
    )

    right_eye_top = np.mean(
        right_eye[
            [2, 3, 4]
        ],
        axis=0
    )

    right_eye_bottom = np.mean(
        right_eye[
            [7, 8, 9]
        ],
        axis=0
    )

    left_eye_opening = distance(
        left_eye_top,
        left_eye_bottom
    )

    right_eye_opening = distance(
        right_eye_top,
        right_eye_bottom
    )

    # ========================================================
    # EYEBROW HEIGHT
    # ========================================================

    left_brow_height = (
        left_eye_center[1]
        - left_brow_center[1]
    )

    right_brow_height = (
        right_eye_center[1]
        - right_brow_center[1]
    )

    # ========================================================
    # MOUTH DIMENSIONS
    # ========================================================

    mouth_y = mouth[:, 1]

    mouth_x = mouth[:, 0]

    mouth_opening = (
        float(mouth_y.max())
        - float(mouth_y.min())
    )

    mouth_width = (
        float(mouth_x.max())
        - float(mouth_x.min())
    )

    return {

        "left_eye_opening":
            left_eye_opening,

        "right_eye_opening":
            right_eye_opening,

        "left_brow_height":
            float(left_brow_height),

        "right_brow_height":
            float(right_brow_height),

        "mouth_opening":
            mouth_opening,

        "mouth_width":
            mouth_width,
    }


# ============================================================
# ASYMMETRY
# ============================================================

def asymmetry_score(
    left,
    right,
):

    denominator = (
        abs(left)
        + abs(right)
        + 1e-6
    )

    return abs(
        left - right
    ) / denominator


def calculate_step_score(
    step_name,
    measurements,
):

    if not measurements:

        return 0.0


    # --------------------------------------------------------
    # Eyes
    # --------------------------------------------------------

    if step_name == "Eyes":

        return asymmetry_score(
            measurements[
                "left_eye_opening"
            ],
            measurements[
                "right_eye_opening"
            ],
        )


    # --------------------------------------------------------
    # Eyebrows
    # --------------------------------------------------------

    if step_name == "Eyebrows":

        return asymmetry_score(
            measurements[
                "left_brow_height"
            ],
            measurements[
                "right_brow_height"
            ],
        )


    # --------------------------------------------------------
    # Smile
    # --------------------------------------------------------

    if step_name == "Smile":

        # Currently no smile-specific
        # asymmetry calculation.
        return 0.0


    # --------------------------------------------------------
    # Mouth
    # --------------------------------------------------------

    if step_name == "Mouth":

        # Currently no mouth-specific
        # asymmetry calculation.
        return 0.0


    return 0.0


# ============================================================
# DRAW REGION
# ============================================================

def draw_points(
    frame,
    landmarks,
    indices,
    width,
    height,
    label,
):

    points = landmark_points(
        landmarks,
        indices,
        width,
        height
    )

    if len(points) == 0:

        return


    pts = np.array(
        points,
        dtype=np.int32
    )

    x, y, w, h = cv2.boundingRect(
        pts
    )

    cv2.rectangle(
        frame,
        (x - 5, y - 5),
        (x + w + 5, y + h + 5),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        label,
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# ============================================================
# DRAW HEADER
# ============================================================

def draw_header(
    frame,
    step_index,
    countdown,
):

    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 115),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "NEURO.AI",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    step = STEPS[step_index]

    cv2.putText(
        frame,
        f"{step_index + 1}/"
        f"{len(STEPS)}  "
        f"{step['name']}",
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        step["instruction"],
        (25, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    if countdown is not None:

        cv2.putText(
            frame,
            str(countdown),
            (width - 75, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 255, 255),
            3
        )


# ============================================================
# DRAW PROGRESS
# ============================================================

def draw_progress(
    frame,
    completed,
):

    height, width = frame.shape[:2]

    y = height - 45

    for i, step in enumerate(STEPS):

        x = 30 + i * 145

        if i < completed:

            symbol = "✓"
            color = (0, 255, 0)

        elif i == completed:

            symbol = "●"
            color = (0, 255, 255)

        else:

            symbol = "○"
            color = (150, 150, 150)

        cv2.putText(
            frame,
            symbol,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            step["name"],
            (x + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )


# ============================================================
# FINAL FACIAL RESULT
# ============================================================

def calculate_final_asymmetry(
    results,
):

    scores = []

    for name, data in results.items():

        score = data.get(
            "asymmetry",
            0.0
        )

        if score > 0:

            scores.append(
                score
            )

    if not scores:

        return 0.0

    return float(
        np.mean(scores)
    )


def severity_from_asymmetry(
    score,
):

    # IMPORTANT:
    # These are prototype thresholds.
    # They are NOT clinically validated thresholds.

    if score < 0.05:

        return "Low asymmetry"

    if score < 0.12:

        return "Mild asymmetry"

    if score < 0.20:

        return "Moderate asymmetry"

    return "High asymmetry"


def get_facial_result(
    results
):

    """
    Convert the step-wise facial measurements
    into the final Neuro.ai facial result.
    """

    score = calculate_final_asymmetry(
        results
    )

    assessment = severity_from_asymmetry(
        score
    )

    return {

        "facial_assessment":
            assessment,

        "asymmetry_score":
            round(
                score,
                4
            ),

        "asymmetry_percentage":
            round(
                score * 100,
                2
            ),

        "step_results":
            results
    }


# ============================================================
# RESULT SCREEN
# ============================================================

def show_final_result(
    frame,
    results,
):

    height, width = frame.shape[:2]

    facial_result = get_facial_result(
        results
    )

    score = facial_result[
        "asymmetry_score"
    ]

    assessment = facial_result[
        "facial_assessment"
    ]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (80, 80),
        (width - 80, height - 80),
        (15, 15, 15),
        -1
    )

    frame[:] = overlay

    cv2.putText(
        frame,
        "ASSESSMENT COMPLETE",
        (
            width // 2 - 220,
            145
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Facial asymmetry: "
        f"{score * 100:.1f}%",
        (
            width // 2 - 210,
            210
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        assessment,
        (
            width // 2 - 180,
            270
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    y = 335

    for name, data in results.items():

        asym = data.get(
            "asymmetry",
            0.0
        )

        cv2.putText(
            frame,
            f"{name}: "
            f"{asym * 100:.1f}%",
            (
                width // 2 - 180,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2
        )

        y += 42

    cv2.putText(
        frame,
        "Press ENTER to accept",
        (
            width // 2 - 150,
            height - 110
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press R to restart",
        (
            width // 2 - 140,
            height - 75
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (
            width // 2 - 120,
            height - 45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


# ============================================================
# LIVE FACIAL ASSESSMENT
# ============================================================

def run_live_assessment():

    print("=" * 70)

    print(
        "NEURO.AI - LIVE FACIAL PALSY ASSESSMENT"
    )

    print("=" * 70)


    # ========================================================
    # VALIDATE FILES
    # ========================================================

    if not MODEL_PATH.exists():

        print(
            f"ERROR: Model not found:\n"
            f"{MODEL_PATH}"
        )

        return None


    if not LANDMARKER_PATH.exists():

        print(
            f"ERROR: MediaPipe model not found:\n"
            f"{LANDMARKER_PATH}"
        )

        return None


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    print(
        "\nLoading trained model..."
    )

    try:

        model = load_model()

        print(
            "✓ Model loaded"
        )

    except Exception as error:

        print(
            f"WARNING: Facial classifier "
            f"could not be loaded:\n{error}"
        )

        model = None


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                LANDMARKER_PATH
            )
        ),

        running_mode=(
            VisionRunningMode.VIDEO
        ),

        num_faces=1,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )


    landmarker = (
        FaceLandmarker.create_from_options(
            options
        )
    )


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print(
            "ERROR: Webcam could not be opened."
        )

        try:

            landmarker.close()

        except Exception:

            pass

        return None


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print(
        "\n✓ Camera started"
    )

    print(
        "\nControls:"
    )

    print(
        "  SPACE = start/capture step"
    )

    print(
        "  R     = restart"
    )

    print(
        "  Q     = quit"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # STATE
    # ========================================================

    step_index = 0

    results = {}

    state = "READY"

    capture_start = None

    assessment_complete = False

    final_result = None

    timestamp_start = time.time()


    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            print(
                "ERROR: Camera frame unavailable."
            )

            break


        frame = cv2.flip(
            frame,
            1
        )


        height, width = frame.shape[:2]


        # ====================================================
        # MEDIAPIPE DETECTION
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb
        )


        timestamp_ms = int(
            (
                time.time()
                - timestamp_start
            ) * 1000
        )


        detection = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        landmarks = None


        if detection.face_landmarks:

            landmarks = (
                detection.face_landmarks[0]
            )


        # ====================================================
        # FINAL RESULT SCREEN
        # ====================================================

        if assessment_complete:

            show_final_result(
                frame,
                results
            )


            cv2.imshow(
                "Neuro.ai - Facial Palsy Assessment",
                frame
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # -----------------------------------------------
            # ACCEPT RESULT
            # -----------------------------------------------

            if key == 13:

                final_result = (
                    get_facial_result(
                        results
                    )
                )

                print()

                print(
                    "=" * 60
                )

                print(
                    "FINAL FACIAL RESULT"
                )

                print(
                    "=" * 60
                )

                print(
                    f"Facial Assessment: "
                    f"{final_result['facial_assessment']}"
                )

                print(
                    f"Asymmetry Score: "
                    f"{final_result['asymmetry_score']}"
                )

                print(
                    f"Asymmetry Percentage: "
                    f"{final_result['asymmetry_percentage']}%"
                )

                print(
                    "=" * 60
                )

                break


            # -----------------------------------------------
            # QUIT
            # -----------------------------------------------

            if key == ord("q"):

                final_result = None

                break


            # -----------------------------------------------
            # RESTART
            # -----------------------------------------------

            if key == ord("r"):

                step_index = 0

                results = {}

                state = "READY"

                capture_start = None

                assessment_complete = False

                final_result = None

                timestamp_start = (
                    time.time()
                )

                continue


            continue


        # ====================================================
        # COUNTDOWN
        # ====================================================

        countdown = None


        if state == "CAPTURING":

            elapsed = (
                time.time()
                - capture_start
            )


            remaining = (
                STEPS[
                    step_index
                ]["duration"]
                - elapsed
            )


            countdown = max(
                0,
                int(
                    np.ceil(
                        remaining
                    )
                )
            )


        # ====================================================
        # HEADER
        # ====================================================

        draw_header(
            frame,
            step_index,
            countdown
        )


        draw_progress(
            frame,
            step_index
        )


        # ====================================================
        # FACE NOT DETECTED
        # ====================================================

        if landmarks is None:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (
                    width // 2 - 180,
                    height // 2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        else:

            # =================================================
            # DRAW FACIAL REGIONS
            # =================================================

            draw_points(
                frame,
                landmarks,
                LEFT_EYE,
                width,
                height,
                "L EYE"
            )


            draw_points(
                frame,
                landmarks,
                RIGHT_EYE,
                width,
                height,
                "R EYE"
            )


            draw_points(
                frame,
                landmarks,
                LEFT_EYEBROW,
                width,
                height,
                "L BROW"
            )


            draw_points(
                frame,
                landmarks,
                RIGHT_EYEBROW,
                width,
                height,
                "R BROW"
            )


            draw_points(
                frame,
                landmarks,
                MOUTH,
                width,
                height,
                "MOUTH"
            )


            cv2.putText(
                frame,
                "FACE DETECTED",
                (
                    width - 230,
                    35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


        # ====================================================
        # READY STATE
        # ====================================================

        if state == "READY":

            cv2.putText(
                frame,
                "Press SPACE to begin",
                (
                    width // 2 - 170,
                    height - 100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


        # ====================================================
        # CAPTURING STATE
        # ====================================================

        elif state == "CAPTURING":

            cv2.putText(
                frame,
                "HOLD POSITION...",
                (
                    width // 2 - 140,
                    height - 100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            elapsed = (
                time.time()
                - capture_start
            )


            duration = (
                STEPS[
                    step_index
                ]["duration"]
            )


            # =================================================
            # CAPTURE STEP
            # =================================================

            if elapsed >= duration:

                if landmarks is not None:

                    measurements = (
                        calculate_measurements(
                            landmarks,
                            width,
                            height
                        )
                    )


                    step_name = (
                        STEPS[
                            step_index
                        ]["name"]
                    )


                    asymmetry = (
                        calculate_step_score(
                            step_name,
                            measurements
                        )
                    )


                    results[
                        step_name
                    ] = {

                        "measurements":
                            measurements,

                        "asymmetry":
                            asymmetry,
                    }


                    print()

                    print(
                        f"✓ {step_name} captured"
                    )


                    print(
                        f"  Asymmetry: "
                        f"{asymmetry * 100:.2f}%"
                    )


                    # ========================================
                    # NEXT STEP
                    # ========================================

                    step_index += 1


                    # ========================================
                    # ALL STEPS COMPLETED
                    # ========================================

                    if step_index >= len(STEPS):
                       assessment_complete = True
                       final_result = get_facial_result(results)

                       print("\n============================================================")
                       print("ASSESSMENT COMPLETE")
                       print("============================================================")
                       print(f"Facial Assessment: {final_result['facial_assessment']}")
                       print(f"Asymmetry Score: {final_result['asymmetry_score']}")
                       print(f"Asymmetry Percentage: {final_result['asymmetry_percentage']}%")

                       print("\n✓ Facial assessment completed successfully.")
                       print("Returning result to Neuro.ai...")
    
                       # Automatically finish the assessment
                       break


                    else:

                        state = "READY"

                        capture_start = None


                else:

                    print(
                        "No face detected. "
                        "Please try again."
                    )


                    state = "READY"

                    capture_start = None


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        cv2.imshow(
            "Neuro.ai - Facial Palsy Assessment",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ====================================================
        # QUIT
        # ====================================================

        if key == ord("q"):

            final_result = None

            break


        # ====================================================
        # RESTART
        # ====================================================

        elif key == ord("r"):

            step_index = 0

            results = {}

            state = "READY"

            capture_start = None

            assessment_complete = False

            final_result = None

            timestamp_start = (
                time.time()
            )

            print(
                "\nAssessment restarted."
            )


        # ====================================================
        # START CAPTURE
        # ====================================================

        elif key == 32:

            if landmarks is None:

                print(
                    "Cannot start: "
                    "no face detected."
                )

                continue


            if state == "READY":

                state = "CAPTURING"

                capture_start = (
                    time.time()
                )


                print()

                print(
                    f"Starting: "
                    f"{STEPS[step_index]['name']}"
                )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


    try:

        landmarker.close()

    except Exception:

        pass


    print(
        "\nCamera closed."
    )


    return final_result


# ============================================================
# MAIN
# ============================================================

def main():

    return run_live_assessment()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    result = run_live_assessment()


    if result is not None:

        print()

        print(
            "=" * 60
        )

        print(
            "FACIAL ASSESSMENT RETURNED"
        )

        print(
            "=" * 60
        )

        print(
            f"Facial Assessment: "
            f"{result['facial_assessment']}"
        )

        print(
            f"Asymmetry Score: "
            f"{result['asymmetry_score']}"
        )

        print(
            f"Asymmetry Percentage: "
            f"{result['asymmetry_percentage']}%"
        )

        print(
            "=" * 60
        )

    else:

        print(
            "\nFacial assessment was cancelled."
        )