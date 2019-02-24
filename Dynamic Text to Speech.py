
#import libraries
import cv2
import numpy as np
from pytesseract import image_to_string
import math
import win32com.client as wincl



cap = cv2.VideoCapture(0)

#Part1
#Finding the finger

while(1):
    
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 60, 100])
    upper_blue = np.array([30, 200, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    if cv2.waitKey(33) == ord('q'):
        mask_selected=mask
        break



(_, contour_finger, _) = cv2.findContours(mask_selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

number_contours_finger=len(contour_finger)
max_area_contour_finger = max(contour_finger, key = cv2.contourArea)
topmost_finger = tuple(max_area_contour_finger[max_area_contour_finger[:,:,1].argmin()][0])
tf1,tf2=topmost_finger
cv2.circle(mask_selected,(tf1,tf2), 10, (150,150,150), -1)

cv2.imshow("finger_location",mask_selected)
cv2.waitKey(0)

img_parameters = mask_selected.shape
height , width = img_parameters
#Part 2
#Countouring

while 1:
    _, frame = cap.read()
    
    crop = frame[tf2-50:tf2,0:width]
    img_parameters = crop.shape
    print img_parameters
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    (_, contours, _) = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    no_of_cnt=len(contours)
    i=0
    dist=[]
    #discard contours with lower than threshold area
    reduction=0
    threshold=400
    no_of_cnt_temp = no_of_cnt
    print no_of_cnt
    while i<(no_of_cnt_temp):
        print i
        cnt_delete = contours[i]
        (not_x ,not_y ),(area_1 ,area_2), not_angle = cv2.minAreaRect(cnt_delete)
        area=area_1*area_2
        if area < threshold:
            contours.pop(i)
            reduction=reduction+1
            no_of_cnt_temp=no_of_cnt_temp-1
        i=i+1       
    no_of_cnt = no_of_cnt - reduction
    

    
            
    print no_of_cnt        
            
    #all contours
    i=0
    while i<(no_of_cnt):
        cnt_word = contours[i]
        bottommost = tuple(cnt_word[cnt_word[:,:,1].argmax()][0])
        b1,b2=bottommost
        dist.append(math.sqrt((tf2 - b1)**2 + (tf1 - b2)**2))
        min_dist=min(dist)
        index_min_dist=dist.index(min_dist)
        
        rect_word = cv2.minAreaRect(cnt_word)
        box_word = cv2.boxPoints(rect_word)
        box_word = np.int0(box_word)
        cv2.drawContours(crop,[box_word],0,(0,0,255),2)
        i=i+1

        
    #selecting closest contour     
    chosen_one=contours[index_min_dist]
    rect = cv2.minAreaRect(chosen_one)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(crop,[box],0,(0,255,0),2)
    
    #masking everything other than contour
    stencil = np.zeros(crop.shape).astype(crop.dtype)
    contours_a = [np.array(chosen_one)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours_a, color)
    result = cv2.bitwise_and(crop, stencil)
    (xc,yc),(w,h),angle=cv2.minAreaRect(chosen_one)
    #result has the image

    if angle<0 and angle>-45:
        angle=angle
    if angle<-45:
        angle=angle+90
    if angle>0 and angle<45:
        angle=angle
    if angle>45:
        angle=90-angle
    
    M = cv2.getRotationMatrix2D((xc,yc),angle,1)
    final = cv2.warpAffine(result,M,(640,50))

    gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    gray_final = cv2.bitwise_not(gray_final)
    thresh_final = cv2.threshold(gray_final, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((8,8), np.uint8)
    dilation_final = cv2.dilate(thresh_final, kernel, iterations=1)
    
    (_, contour_of_mask, _) = cv2.findContours(dilation_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    length_of_mask=len(contour_of_mask)
    i=0
    while(i<length_of_mask):
        cnt_final=contour_of_mask[i]
        rect_final = cv2.minAreaRect(cnt_final)
        (xcf,ycf),(wf,hf),angle_final=cv2.minAreaRect(chosen_one)
        
        left_most=np.absolute(int(xcf-(wf/2)))
        right_most=int(xcf+(wf/2))
        top_most=np.absolute(int(ycf-(hf/2)))
        bottom_most=int(ycf+(hf/2))
##        left_most = tuple(cnt_final[cnt_final[:,:,0].argmin()][0])
##        right_most = tuple(cnt_final[cnt_final[:,:,0].argmax()][0])
##        top_most = tuple(cnt_final[cnt_final[:,:,1].argmin()][0])
##        bottom_most = tuple(cnt_final[cnt_final[:,:,1].argmax()][0])


        
        print left_most,right_most,top_most,bottom_most
        i=i+1
        crop_final = crop[top_most:bottom_most,left_most:right_most]

        string=image_to_string(crop_final,lang='eng')
        print "string "+ string
        
	speak = wincl.Dispatch("SAPI.SpVoice")
    	speak.Speak(string)
        
        cv2.imshow("final",final)
        cv2.waitKey(0)
        cv2.imshow("finalcropped",crop_final)
        cv2.waitKey(0)
        
        
        

