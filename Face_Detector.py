import cv2
from random import randrange

# Load some pre-trained data on face frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces
# img =  cv2.imread('SSR.jpg')

# To capture the video
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grey_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces 
    face_coordinates = trained_face_data.detectMultiScale(grey_scaled_img)

    # Draw rectangles (B, G, R) ==> color
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)


    # Display the image
    cv2.imshow('Image Display', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break 

webcam.release()


'''
# Convert image to greyscale
grey_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces 
face_coordinates = trained_face_data.detectMultiScale(grey_scaled_img)

# Draw rectangles (B, G, R) ==> color
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)

# print(face_coordinates)

# Display the image
cv2.imshow('Image Display', img)
cv2.waitKey()
'''

print("Code Completed!!")