import cv2
import ctypes
import numpy as np
import math
import collections as coll
X=[]
Y=[]
X2=[]
Y2=[]
m=[]
cap = cv2.VideoCapture(0)
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float

while(1):
    areas=[0.0]
    # Take each frame
    ret, frame = cap.read()
    kernel = np.ones((5,5),np.uint8)
    frame=cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame=cv2.medianBlur(frame,5)
    frame=cv2.GaussianBlur(frame,(5,5),0)
    # Convert BGR to HSV
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50], dtype=np.uint8)
    upper_blue = np.array([130,255,255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)
    mask2=cv2.GaussianBlur(mask2,(5,5),0)
    #mask=mask.smoothGaussian(9)
    im2 = mask2
    ret,thresh2 = cv2.threshold(mask2,127,255,cv2.cv.CV_THRESH_BINARY)
    contours2, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas2 = [cv2.contourArea(c) for c in contours2]





    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # define range of blue color in HSV
    lower_blue = np.array([0,151,80], dtype=np.uint8)
    upper_blue = np.array([255,185,155], dtype=np.uint8)
   
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #mask=mask.smoothGaussian(9)
    im = mask
    rect_12 = cv2.getStructuringElement(cv2.MORPH_RECT,(10,5))
    rect_6 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,3))
    
    #mask= cv2.erode(mask,rect_12,iterations = 1)
    #mask= cv2.dilate(mask,rect_6,iterations = 2)
    mask=cv2.GaussianBlur(mask,(5,5),0)
    
    ret,thresh = cv2.threshold(mask,127,255,0)

    
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    
    try:
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(frame,(x+w/2,y+h/2),(x+w/2,y+h/2),(0,255,0),2)
        X.insert(0,x*1.5)
       
        #print X,x+w
        if len(Y)>10:
            Y.pop()
            
            y=math.ceil(sum(Y)/10)
            #print Y,y
    except:
        pass    
    try: 
        max_index2 = np.argmax(areas2)
        cnt2=contours2[max_index2]
        x2,y2,w2,h2 = cv2.boundingRect(cnt2)
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(10,10,10),2)
        cv2.rectangle(frame,(x2+w2/2,y2+h2/2),(x2+w2/2,y2+h2/2),(10,255,255),2)
        X2.insert(0,x2*1.5)
        Y2.insert(0,y2)
        if len(X2)>10:
            X2.pop()
            
            x2=math.floor(sum(X2)/10)
            #print X2,x2
        if len(Y2)>10:
            Y2.pop()
            
            y2=math.floor(sum(Y2)/10)
            #print Y2,y2
        if 1366-int(x2)*2 < 0:
            x2=1364/2
        
        ctypes.windll.user32.SetCursorPos(1366-int(x2)*2,int(y2)*2)
        #ctypes.windll.user32.SetCursorPos(1286,545)

        m.insert(0,(1366-int(x2)*2,int(y2)*2))
        if len(m)>50:
            m.pop()
        c=coll.Counter(m)
        n=[k for k,v in c.items() if v==2]
        pp=False
        print pp
        kk=0
##        i=n[0]
        if n!=[]:
            for i in n:
                x=i[0]
                y=i[1]
                y1=y
                ind = m.index(i)
                if m[ind]!=m[ind+1]!=m[ind+2]!=m[ind+3]!=m[ind+4]:
                    while kk<50:
                        kk=kk+1
                        y1=y1+1
                        if (x,y1) in m:
                            y=y+(y1-y)/2
                            ctypes.windll.user32.SetCursorPos(x,y)
                            print c
                            print n
                          #  ctypes.windll.user32.mouse_event(6,0,0,0,0)
                            pp=True
                            m.remove(i)
                            
                            break
##                    if pp==True:
##                        break
        
    except:
        pass
##    try:   
##        if x+w>x2>x:
##            if y+h>y2>y:
##                ctypes.windll.user32.mouse_event(6,0,0,0,0)
##        elif x2+w>x>x2:
##            if y2+h>y>y2:
##                ctypes.windll.user32.mouse_event(6,0,0,0,0)
##    except:
##        pass
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask2)
    #cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
    #cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    blank_image = np.zeros((height,width,3), np.uint8)
    
    cv2.imshow('test',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
##i=1
##>>> while 1:
##	end
##
##	
##
##Traceback (most recent call last):
##  File "<pyshell#24>", line 2, in <module>
##    end
##NameError: name 'end' is not defined
##>>> y=524
##>>> while y<769:
##	y=y+1
##	if (1286,y) in m: print y
##
##	
##566
##>>> 566-524
##42
##>>> (566-524)/2
##21
##>>> 524+21
##545
