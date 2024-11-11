import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: "Delete", 27: "Nothing", 28: "Space"
}

st.title("Sign Language Recognition")

# Streamlit camera input
stframe = st.empty()
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Convert the image to OpenCV format
    image = Image.open(camera_input)
    bytes_data = camera_input.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = cv2_img.shape
    frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make predictions using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the processed frame in Streamlit
    stframe.image(frame_rgb, channels='RGB')