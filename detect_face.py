import glob
import os
import cv2

names = ["takahashi"]
out_dir = "./face"
os.makedirs(out_dir, exist_ok=True)

for i in range(len(names)):
    in_dir = "./data/" + names[i] + "/*.jpg"
    print(in_dir)
    in_jpg = glob.glob(in_dir)
    os.makedirs(out_dir + "/" + names[i], exist_ok=True)

    print(len(in_jpg))
    count = 1
    for num in range(len(in_jpg)):
        # Load images
        print("Loading" + str(in_jpg[num]))
        image = cv2.imread(str(in_jpg[num]))
        if image is None:
            print("Not open: ", num)
            continue

        # Start face recognition
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("./lib/haarcascade_frontalface_alt.xml")
        face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64,64))

        # In case if multiple faces detected
        if len(face_list)>0:
            for rect in face_list:
                x,y,width,height = rect
                image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                image = cv2.resize(image, (64,64))
                # save
                filename = os.path.join(out_dir + "/" + names[i], str(count) + ".jpg")
                count += 1
                cv2.imwrite(str(filename), image)
                print("Saved " + str(count) + ".jpg")
        #in case when no faces detected
        else:
            print("No face detected")
            continue
        print(image.shape)
