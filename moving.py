import cv2
import time

# Start webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)
cap.set(4, 480)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=700,
    varThreshold=40,
    detectShadows=True
)

# FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for better speed
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(blur)

    # Remove shadows
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Fill gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Dilate object
    dilated = cv2.dilate(closing, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Ignore small movements
        if area < 1200:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(cnt)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display object area
        cv2.putText(
            frame,
            f"Area: {int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Motion status
    status = "Motion Detected" if motion_detected else "No Motion"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Show outputs
    cv2.imshow("Moving Object Detection", frame)
    cv2.imshow("Threshold", dilated)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()