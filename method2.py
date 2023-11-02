import cv2
import time 

vidCap = cv2.VideoCapture('MOT17-13-SDP-raw.mp4')

#initializing OpenCV Background subtraction for KNN and MOG2
BS_KNN = cv2.createBackgroundSubtractorKNN()
BS_MOG2 = cv2.createBackgroundSubtractorMOG2()

while vidCap.isOpened(): 
    ret, frame = vidCap.read() #reads next frame 

    if not ret: 
        break

    cv2.imshow('Original Video', frame)

    knn_FGMask = BS_KNN.apply(frame)
    cv2.imshow('KNN-Foreground Mask', knn_FGMask)

    mog2_FGMask = BS_MOG2.apply(frame)
    cv2.imshow('MOG2-Foreground Mask', mog2_FGMask)

    #waiting for a key to be pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

#releasing video capture 
cv2.destroyAllWindows()
vidCap.release()