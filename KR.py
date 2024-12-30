import cv2
import numpy as np
import face_recognition
import os
import csv
import time
from datetime import datetime
import requests
import pygame

# Define base directories
base_dir = os.getcwd()
recording_dir = os.path.join(base_dir, "Recordings")
photo_dir = os.path.join(base_dir, "Photos")
known_faces_dir = os.path.join(base_dir, "KnownFaces")

# Create directories if they do not exist
if not os.path.exists(recording_dir):
    os.makedirs(recording_dir)
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

# Load Haar cascades
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Paths to models
age_model = os.path.join(base_dir, "Models", "age_net.caffemodel")
age_proto = os.path.join(base_dir, "Models", "age_deploy.prototxt")
gender_model = os.path.join(base_dir, "Models", "gender_net.caffemodel")
gender_proto = os.path.join(base_dir, "Models", "gender_deploy.prototxt")

# Load the models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Age and gender categories
AGE_LIST = ['(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Initialize pygame for playing sound
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound(r"C:\Users\om\Desktop\2\alert.wav")

# Function to send message to Telegram
def send_telegram_message(bot_token, chat_id, message, video_path=None, photo_path=None):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=data)
    print("Send message response:", response.json())

    if photo_path:
        with open(photo_path, 'rb') as photo:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            files = {'photo': photo}
            data = {'chat_id': chat_id}
            response = requests.post(url, files=files, data=data)
            print("Send photo response:", response.json())

    if video_path:
        with open(video_path, 'rb') as video:
            url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
            files = {'video': video}
            data = {'chat_id': chat_id}
            response = requests.post(url, files=files, data=data)
            print("Send video response:", response.json())

# Function to predict age and gender
def predict_age_and_gender(face_img):
    if face_img.size == 0:
        print("Empty face image!")
        return None, None

    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    if gender_preds.size == 0:
        print("Gender predictions are empty!")
        return None, None
    gender = GENDER_LIST[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    if age_preds.size == 0:
        print("Age predictions are empty!")
        return None, None
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age

# Function to describe face with gender and age
def describe_face_with_age_gender(img, faceLoc):
    y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
    face_img = img[y1:y2, x1:x2]

    gender, age = predict_age_and_gender(face_img)
    if gender is None or age is None:
        return "Unknown gender and age"
    return f"{gender}, aged around {age}"

# Function to describe clothing color
def describe_clothing_color(img, faceLoc):
    y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
    clothing_region = img[y2:y2 + int((y2 - y1) / 2), x1:x2]  # Region below the face
    average_color = cv2.mean(clothing_region)[:3]
    return get_clothing_color_name(average_color)

def get_clothing_color_name(average_color):
    blue, green, red = average_color
    if red > green and red > blue:
        return "red"
    elif green > red and green > blue:
        return "green"
    elif blue > red and blue > green:
        return "blue"
    elif red > 100 and green > 100 and blue < 100:
        return "yellow"
    elif red < 100 and green > 100 and blue > 100:
        return "cyan"
    elif red > 100 and green < 100 and blue > 100:
        return "magenta"
    else:
        return "dark color"

# Function to check if the person is wearing glasses
def wearing_glasses(img, faceLoc):
    y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
    gray_face = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    glasses = glasses_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    return len(glasses) > 0

# Function to detect facial hair and hair length
def detect_facial_hair_and_hair_length(face_img):
    facial_hair = "no facial hair"
    hair_length = "short hair"
    height, width = face_img.shape[:2]

    if height > 80:
        facial_hair = "has facial hair"

    forehead_region = face_img[0:int(height / 3), int(width / 4):int(3 * width / 4)]
    if np.mean(forehead_region) < 120:
        hair_length = "long hair"

    return facial_hair, hair_length

# Function to save prompt to CSV file
def save_prompt(recording_name, prompt):
    with open('prompts.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([recording_name, prompt])

# Load known faces
encodeListKnown = []
classNames = []

for filename in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, filename)
    img = face_recognition.load_image_file(img_path)
    img_encoding = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(img_encoding)
    classNames.append(os.path.splitext(filename)[0])

# Initialize camera
cap = cv2.VideoCapture(0)
unknown_detected = False
initial_notification_time = 5  # Time for initial notification
alarm_notification_time = 20  # Time for alarm notification (10 seconds after initial)
security_notification_time = 5  # Time after alarm for security notification
duration = 0
start_time = 0
recording = False
out = None
initial_notification_sent = False
alarm_notification_sent = False
alarm_sounding = False
security_notification_sent = False

# Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = '7384217991:AAG6HXvs-HkdEAlLD1KbyiQ66cWc_bkiaFs'
TELEGRAM_CHAT_ID = '6707952441'
TELEGRAM_BOT_TOKEN_SEC = '7644487145:AAG7vb7NA3fHIBh_AtlMWxveVops0Y2HBOY'  # sec bot token
TELEGRAM_CHAT_ID_SEC = '6707952441'  # sec bot chat ID
car_image_path = r"C:\Users\om\Desktop\2\map.png"  # Replace with your car image path

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    unknown_face_detected = False

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            cv2.putText(img, name, (faceLoc[3] * 4, faceLoc[0] * 4 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (faceLoc[3] * 4, faceLoc[0] * 4), (faceLoc[1] * 4, faceLoc[2] * 4), (0, 255, 0), 2)
        else:
            cv2.putText(img, "Unknown", (faceLoc[3] * 4, faceLoc[0] * 4 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (faceLoc[3] * 4, faceLoc[0] * 4), (faceLoc[1] * 4, faceLoc[2] * 4), (0, 0, 255), 2)
            unknown_face_detected = True

    if unknown_face_detected:
        if not unknown_detected:
            unknown_detected = True
            start_time = time.time()
        
        duration = time.time() - start_time

        if duration > initial_notification_time and not initial_notification_sent:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_name = f"unknown_{timestamp}.avi"
            recording_path = os.path.join(recording_dir, recording_name)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(recording_path, fourcc, 20.0, (640, 480))

            snapshot_name = f"unknown_{timestamp}.jpg"
            snapshot_path = os.path.join(photo_dir, snapshot_name)
            cv2.imwrite(snapshot_path, img)

            gender_age = describe_face_with_age_gender(img, faceLoc)
            clothing_color = describe_clothing_color(img, faceLoc)
            glasses = "wearing glasses" if wearing_glasses(img, faceLoc) else "not wearing glasses"
            facial_hair, hair_length = detect_facial_hair_and_hair_length(img[faceLoc[0]*4:faceLoc[2]*4, faceLoc[3]*4:faceLoc[1]*4])

            current_datetime = datetime.now().strftime("%d-%m-%Y")
            current_time = datetime.now().strftime("%I:%M %p")

            prompt = f"**ALERT: \n Unknown person detected for {initial_notification_time} seconds**\n" \
                     f"- Date: {current_datetime}\n" \
                     f"- Time: {current_time}\n" \
                     f"- {gender_age}.\n" \
                     f"- {glasses}.\n" \
                     f"- {facial_hair}, with {hair_length}.\n" \
                    #  f"- Wearing {clothing_color} clothing.\n"

            print(prompt)
            save_prompt(recording_name, prompt)
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, prompt, recording_path, snapshot_path)

            initial_notification_sent = True
            recording = True

        elif duration > alarm_notification_time and not alarm_notification_sent:
            alarm_prompt = "**ALERT:\n Unknown person still present. Sounding alarm now!**"
            print(alarm_prompt)
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alarm_prompt)
            alarm_notification_sent = True
            alarm_sounding = True
            alarm_sound.play(-1)  # Play the sound on loop
            
            # Record the time when the alarm started
            alarm_start_time = time.time()

        
    else:
        if unknown_detected:  # If previously detected but now not
            object_secure_prompt = "**Object of interest is secure.**"
            print(object_secure_prompt)
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, object_secure_prompt)

        # Stop the alarm if it is sounding
        if alarm_sounding:
            alarm_sound.stop()
            alarm_sounding = False  # Reset alarm state

        # Reset detection states
        unknown_detected = False
        duration = 0
        initial_notification_sent = False
        alarm_notification_sent = False
    

    # Check if the alarm has sounded and 5 seconds have passed
    if alarm_sounding and 'alarm_start_time' in locals():
        if time.time() - alarm_start_time >= 5:  # 5 seconds after the alarm sounded
            security_alert = "**SECURITY ALERT: \n Sending security to location **"
            securit_alert = "**ALERT !!: \n Go to location immediately \n Tresspasser: Catch Him !!**"
            print(security_alert)
            print(securit_alert)
            send_telegram_message(TELEGRAM_BOT_TOKEN_SEC, TELEGRAM_CHAT_ID_SEC, securit_alert, photo_path=snapshot_path)  # Attach car image
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, security_alert, photo_path=car_image_path)  # Attach car image
            alarm_start_time = None  # Reset the alarm start time to prevent repeated messages

    if recording and out is not None:
        out.write(img)

    cv2.imshow("Camera", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        if alarm_sounding:
            alarm_sound.stop()

if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()
