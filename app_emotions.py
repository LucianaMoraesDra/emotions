import numpy as np
import cv2
import pandas as pd
from sklearn.svm import SVC
import mediapipe as mp
from sklearn.metrics import pairwise_distances
import pickle

text = 'How are you?'
color=(255,0,0)

# mp_solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# distance function 
def dist(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

# load model
model = pickle.load(open('./model_smile_sad_neutre.sav', 'rb'))

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

    compt=0
    
    while cap.isOpened():
        success, image = cap.read()
        compt=compt+1
      
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
        if compt%12 == 0 :
            compt=0
        
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # parameterizing distances
                    ref_dist = dist((face_landmarks.landmark[127].x,
                                    face_landmarks.landmark[127].y,
                                    face_landmarks.landmark[127].z),
                                    (face_landmarks.landmark[356].x,
                                    face_landmarks.landmark[356].y,
                                    face_landmarks.landmark[356].z)
                    )

                    distances = pairwise_distances([(i.x, i.y, i.z) for i in face_landmarks.landmark])

                    dataset = np.array([distances[np.triu_indices(len(distances), k=1)]/ref_dist])

                    df = pd.DataFrame(dataset)
                    
                    # predictions
                    pred = model.predict(df)
                    
                    # print(pred)
                    if pred[0] == 0 :
                        text='Heureux :-D'
                        color=(255,140,0)

                    if pred[0] == 1 :
                        text='Souriant.e :-)'
                        color=(106,90,205)

                    if pred[0] == 2 :
                        text='Neutre'
                        color=(64,64,64)

                    if pred[0] == 3 :
                        text='Pas content :-('
                        color=(0,128,255)

                    if pred[0] == 4 :
                        text="Guillaume devant un boxplot :-z"
                        color=(0,128,255)

        image = cv2.flip(image, 1)
        
        cv2.putText(image,str(text), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)
            # Flip the image horizontally for a selfie-view display.
                
        cv2.imshow('How are you?', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


