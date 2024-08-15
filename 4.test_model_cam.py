import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained Keras model
model = load_model('trained_model.h5')

# Load the label array
labels= np.load('train_labels.npy')

# Create a cascade classifier object for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define a function to recognize face
def recognize_face(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        # Extract the face from the image
        face = img[y:y+h, x:x+w]

        # Resize the face to match the input size of the CNN model
        face = cv2.resize(face, (128,128))

        # Normalize the pixel values to be between 0 and 1
        face = face / 255.0

        # Add a batch dimension to the face array
        face = np.expand_dims(face, axis=0)
            
        # Predict the identity of the face ROI
        preds = model.predict(face)
    
        # Get the index of the predicted class
        pred_idx = np.argmax(preds)
        print("pred index:",pred_idx," ")
    
        pred_label = labels[pred_idx]
        print("pred label:",pred_label,"\n")
    
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Write the predicted label below the rectangle
        cv2.putText(img, pred_label, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  

    return img

# Start the main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Call the recognize_face function to recognize faces in the frame
    frame = recognize_face(frame)

    # Show the frame in a window
    cv2.imshow('Real-time Face Recognition', frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
