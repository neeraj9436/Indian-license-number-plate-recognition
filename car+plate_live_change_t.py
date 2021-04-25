import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import operator
import pytesseract
import time
import os

model = tf.keras.models.load_model('mosaic_ps2_infinity1.h5')
##model.summary()

thresh_plate=140
thresh_car=0
def nothing(x):
    pass


def detect(img,th):

    img = cv2.resize(img,(800,200))
##    frame = img.copy()
    


    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1,(5,5),0)
    cv2.namedWindow('plate_thresh')
    cv2.createTrackbar("t",'plate_thresh',100,255,nothing)
    while True:
        frame = img.copy()
        thr = cv2.getTrackbarPos('t','plate_thresh',)
        th=thr
        ret,im_th = cv2.threshold(img1,thr,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('plate_trackbar',im_th)
##        car_detect(img,thr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break    
        

        
        ##cv2.imshow('img',img)
        ##print(img)
        ret,im_th = cv2.threshold(img1,th,255,cv2.THRESH_BINARY_INV)

        kernel = np.ones((1,1), np.uint8)
        im_th = cv2.erode(im_th, kernel) 
        im_th = cv2.dilate(im_th,(7,7))

        ##cv2.imshow('img_t',cv2.bitwise_not(im_th))
        ##cv2.imshow('img_th',im_th)

        thr = im_th.copy()
        thr[0:20,0:800]=0
        thr[179:200,0:800]=0
        thr[0:200,0:25]=0
        thr[0:200,775:800]=0

        cv2.imshow('thr',thr)
        _,ctrs,hierachy = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        rects.sort(key=lambda x: x[0])

        ans=''
        for rect in rects:
            if (rect[2]*rect[3]<3000 or rect[2]*rect[3]>13000) or (rect[3]/rect[2]<1):
                    continue        
##            cv2.putText(img,str(rect[2]*rect[3]), (rect[0],rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    ##        print(rect[2]*rect[3],rect[3]/rect[2])

                   
            cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,255),3)
            
    ##        cv2.imshow('img with rect',img)
            pic_new = im_th[rect[1]-15:rect[3]+rect[1]+15,rect[0]-15:rect[2]+rect[0]+15]
    ##        cv2.imshow(str(rect),pic_new)               
            
            pic_new = cv2.resize(pic_new,(32,32))
            pic_new = cv2.GaussianBlur(pic_new,(5,5),0)
        ##    text=text+pic_new
    ##        cv2.imshow(str(rect[0]),pic_new)
            pic_new = np.array(pic_new,dtype='float32')
            pic_new=pic_new/255
            pic_new = pic_new.astype(np.float64)
            pic_new=pic_new.reshape(1,32,32,1)
            result = (model.predict(pic_new))

            prediction =  {'0': result[0][0], 
                           '1': result[0][1], 
                           '2': result[0][2],
                           '3': result[0][3],
                           '4': result[0][4], 
                           '5': result[0][5], 
                           '6': result[0][6],
                           '7': result[0][7],
                           '8': result[0][8], 
                           '9': result[0][9], 
                           'A': result[0][10],
                           'B': result[0][11],
                           'C': result[0][12], 
                           'D': result[0][13], 
                           'E': result[0][14],
                           'F': result[0][15],
                           'G': result[0][16], 
                           'H': result[0][17], 
                           'I': result[0][18],
                           'J': result[0][19],
                           'K': result[0][20], 
                           'L': result[0][21], 
                           'M': result[0][22],
                           'N': result[0][23],
                           'O': result[0][24], 
                           'P': result[0][25], 
                           'Q': result[0][26],
                           'R': result[0][27],
                           'S': result[0][28], 
                           'T': result[0][29], 
                           'U': result[0][30],
                           'V': result[0][31],
                           'W': result[0][32], 
                           'X': result[0][33], 
                           'Y': result[0][34],
                           'Z': result[0][35],
                                      }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        ##  print("ans",prediction[0][0],)
            ans=ans+(prediction[0][0])
            cv2.putText(frame,prediction[0][0], (rect[0],rect[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),5)
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(255,0,0),3)
            cv2.imshow('frame_plate',frame)

    ##    print('Plate No:',ans)
##        return ans







def car_detect(img,th):
    img = cv2.resize(img,(700,700))
##    cv2.imshow('img',img)
    frame = img.copy()



    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cv2.namedWindow('th')
    cv2.createTrackbar("t",'th',100,255,nothing)
    while True:
        thr = cv2.getTrackbarPos('t','th',)
        th=thr
        ret,im_th = cv2.threshold(img,thr,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('trackbar',im_th)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break    

    
    img = cv2.GaussianBlur(img,(5,5),0)
    ##print(img)
    ret,im_th = cv2.threshold(img,th,255,cv2.THRESH_BINARY_INV)
    im_th=cv2.bitwise_not(im_th) 
    kernel = np.ones((1,1), np.uint8)
    im_th = cv2.erode(im_th, kernel) 
    im_th = cv2.dilate(im_th,(7,7))
##    cv2.imshow('imth',im_th)
    _,ctrs,hierachy = cv2.findContours(im_th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects.sort(key=lambda x: x[0])
##    print(len(rects))
    ans=''
    for rect in rects:
        if (rect[2]*rect[3]<3000 or rect[2]*rect[3]>25000) or rect[2]/rect[3]<2.5 or rect[2]/rect[3]>5 :
                continue        
        
        print(rect[2]*rect[3],rect[2]/rect[3])

               
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,255),3)
        cv2.imshow('frame',frame)
        roi = frame[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
        
##        cv2.imshow('roi',roi)
        op=detect(roi,100)
        
##        if len(op)>=4:
        cv2.putText(frame,op, (rect[0],rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
        cv2.imshow('img with rect',frame)



##img = cv2.imread('car1.jpg')
##
##car_detect(img,100)
fol = r'C:\Users\user\Desktop\Programming\mosaic_ps2\test_multipleCar'

for i in os.listdir(fol):
    print(i)
    img = cv2.imread(fol+'/'+i)
    
    car_detect(img,100)


