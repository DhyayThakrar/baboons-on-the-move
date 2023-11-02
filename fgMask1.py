import cv2
import time 

vidCap = cv2.VideoCapture('MOT17-13-SDP-raw.mp4')


#reading the first frame of the video 
ret, frame1 = vidCap.read() 


while vidCap.isOpened(): 
    #reading frame by frame
    ret, frame2 = vidCap.read() 

    if not ret: 
        break

    cv2.imshow('Raw Video', frame2)

    #extracting the foreground mask 
    fgMask = cv2.absdiff(frame1, frame2)

    #applying the threshold for increasing white foreground
    _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('Foreground Mask', thresh)

    frame1 = frame2

    #waiting for a key to be pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

#releasing video capture 
cv2.destroyAllWindows()
vidCap.release()

