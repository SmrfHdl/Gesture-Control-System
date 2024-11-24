import cv2
import mediapipe as mp
import math
from pynput.keyboard import Controller
import time

def speed_up(hand_landmarks, image_height):
    """Check if is only index finger up."""
    index_tip_y = hand_landmarks.landmark[8].y * image_height
    index_dip_y = hand_landmarks.landmark[6].y * image_height
    thumb_tip_y = hand_landmarks.landmark[4].y * image_height
    middle_tip_y = hand_landmarks.landmark[12].y * image_height
    ring_tip_y = hand_landmarks.landmark[16].y * image_height
    pinky_tip_y = hand_landmarks.landmark[20].y * image_height

    return (
        index_tip_y < index_dip_y
        and thumb_tip_y > index_tip_y
        and middle_tip_y > index_tip_y
        and ring_tip_y > index_tip_y
        and pinky_tip_y > index_tip_y
    )


def process_hands(image, results, keyboard, width, height):
    """Hands processing and key controlling."""
    index_fingers_up = 0
    hands_center = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hands_center.append(
                [
                    int(hand_landmarks.landmark[9].x * width),
                    int(hand_landmarks.landmark[9].y * height),
                ]
            )
            if speed_up(hand_landmarks, height):
                index_fingers_up += 1

    if len(hands_center) == 2:
        # Calculate angle between diameter and Ox
        angle = math.degrees(
            math.atan2(
                hands_center[1][1] - hands_center[0][1],
                hands_center[1][0] - hands_center[0][0],
            )
        )
        if angle < 0:
            angle += 360

        # Key control
        if angle > 180:
            keyboard.press("a")
            time.sleep(0.1)
            keyboard.release("a")
            print("Pressing 'a' (left)")
        else:
            keyboard.press("d")
            time.sleep(0.1)
            keyboard.release("d")
            print("Pressing 'd' (right)")

        if index_fingers_up == 2:
            keyboard.press("w")
            time.sleep(0.1)
            keyboard.release("w")
            print("Pressing 'w' (speed up)")

    return hands_center


def main():
    keyboard = Controller()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Configuring MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Preprocess image
            image = cv2.resize(image, (640, 480))
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw results and handle landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width, _ = image.shape

            hands_center = process_hands(image, results, keyboard, width, height)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Draw a line connect 2 middle points
                if len(hands_center) == 2:
                    center_x = (hands_center[0][0] + hands_center[1][0]) // 2
                    center_y = (hands_center[0][1] + hands_center[1][1]) // 2
                    radius = int(
                        math.sqrt(
                            (hands_center[0][0] - hands_center[1][0]) ** 2
                            + (hands_center[0][1] - hands_center[1][1]) ** 2
                        ) / 2
                    )
                    cv2.line(
                        image,
                        tuple(hands_center[0]),
                        tuple(hands_center[1]),
                        (0, 255, 0),
                        5,
                    )
                    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 5)

            # Display the image
            cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))

            # Exit condition
            if cv2.waitKey(5) & 0xFF == 27:
                break

            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.03 - elapsed_time))

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
