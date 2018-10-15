import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
import sys

nameLabel = ["Hiroshi Takahashi", "Hiroshi Tamaki"]

def detect_face(image):
    print(image.shape)
    img_height = image.shape[0]
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./lib/haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)
            name = detect_who(img)
            font_height = int(height/10)
            cv2.putText(image,name,(x,y+height+font_height),cv2.FONT_HERSHEY_DUPLEX,font_height/22,(255,0,0),2)
    #顔が検出されなかった時
    else:
        print("no face")
    return image

def detect_who(img):
    #予測
    result = model.predict(img)
    for i in range(len(result[0])):
        print(nameLabel[i] + ": " + str(round(result[0][i]*100, 2)) + "%")
    nameNumLabel=np.argmax(result)
    return nameLabel[nameNumLabel] + " (" + str(round(result[0][nameNumLabel]*100, 2)) + "%)"


if __name__ == '__main__':
    model = load_model('./face_recognition_cnn.h5')
    if len(sys.argv) != 2:
        print('invalid argment')
        sys.exit()
    else:
        im_jpg = sys.argv[1]
        image=cv2.imread(im_jpg)
        if image is None:
            print("Not open:")
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        whoImage = detect_face(image)

        plt.imshow(whoImage)
        plt.show()
