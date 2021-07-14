import cv2
import mediapipe as mp

# Drawing utility
mp_drawing = mp.solutions.drawing_utils
# Face detection utility
mp_face_detection = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec((255, 220, 0), thickness=1, circle_radius=1)

# Face mash
mp_face_mesh = mp.solutions.face_mesh

# Model for detecting the face
# model_detection = mp_face_detection.FaceDetection()

# Model facemash
model_facemesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while cap.isOpened():
	flag, frame = cap.read()
	if not flag:
		print("Could not access the camera.")
		break


	########################For detection only##################
	# results = model_detection.process(frame)
	# for landmark in results.detections:

	# 	print(mp_face_detection.get_key_point(landmark, mp_face_detection.FaceKeyPoint.NOSE_TIP))
	# 	mp_drawing.draw_detection(frame, landmark)
	
	########################Meshing only##########################
	results = model_facemesh.process(frame)
	for landmark in results.multi_face_landmarks:
		print(landmark)
		mp_drawing.draw_landmarks(
			image=frame,
			landmark_list=landmark,
			connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)





	cv2.imshow('Frame', frame)
	if cv2.waitKey(10) & 0xff == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()




backgroundapp.py


import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
bg_image = cv2.imread("goku.jpg")


model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
	flag, frame = cap.read()
	if not flag:
		print("Could not access the camera")

	results = model.process(frame)
	condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
	if bg_image is None:
		bg_image = np.zeros(frame.shape, dtype=np.uint8)
		bg_image[:] = (0, 255, 0)
	bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
	output_image = np.where(condition, frame, bg_image)
	cv2.imshow("Frame", output_image)
	if cv2.waitKey(10) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
