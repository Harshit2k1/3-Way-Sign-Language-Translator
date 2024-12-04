import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
from PIL import Image, ImageTk
import tkinter as tk
import string
import keras
from pygame import mixer 
from gtts import gTTS
from transformers import pipeline



def func1():
    r = sr.Recognizer()

    arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        i = 0
        while True:
            print("Recording....")
            audio = r.listen(source)
            
            try:
                a = r.recognize_google(audio)
                a = a.lower()
                print('You Said: ' + a.lower())

                for c in string.punctuation:
                    a = a.replace(c, "")

                if(a.lower() == 'goodbye' or a.lower() == 'good bye' or a.lower() == 'bye' or a.lower() == 'stop translating please'):
                    print("It was a pleasure translating for you! See you soon")
                    break
                else:
                    for i in range(len(a)):
                        if(a[i] in arr):

                            ImageAddress = 'assets/letters/'+a[i]+'.jpg'
                            ImageItself = Image.open(ImageAddress)
                            ImageNumpyFormat = np.asarray(ImageItself)
                            plt.imshow(ImageNumpyFormat)
                            plt.draw()
                            plt.pause(0.8)
                        else:
                            continue

            except:
                print(" ")
            plt.close()


def func2():
    model = keras.models.load_model("asl_classifier_new.h5")
    pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    labels_dict = {
    0: '1', 1: 'w', 2: '0', 3: '6', 4: 'd', 5: '3', 6: 'u', 7: 'p', 8: 'x', 9: '5', 10: '8', 11: '4', 12: 'z', 13: '7', 14: 't', 15: 'i', 16: 'c', 17: 'h', 18: 'o', 19: 'k', 20: '2', 21: 'n', 22: 'l', 23: 'v', 24: 'b', 25: 'y', 26: 'f', 27: 'm', 28: 'g', 29: 's', 30: 'r', 31: 'e', 32: 'a', 33: '9', 34: 'j', 35: 'q'
}

    color_dict = (0, 255, 0)
    img_size = 128
    minValue = 70
    source = cv2.VideoCapture(0)
    count = 0
    string = " "
    prev = " "
    prev_val = 0
    i=0
    sentiment=''
    result1=''
    while(True):
        i=i+1
        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img, (24, 24), (250, 250), color_dict, 2)
        crop_img = gray[24:250, 24:250]
        count = count + 1
        if(count % 25 == 0):
            prev_val = count
        cv2.putText(img, str(prev_val//25), (300, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        blur = cv2.GaussianBlur(crop_img, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(
            th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        resized = cv2.resize(res, (img_size, img_size))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
        result = model.predict(reshaped)
        # print(result)
        label = np.argmax(result, axis=1)[0]


        if(count == 50):
            count = 10
            prev = labels_dict[label]
            if(label == 0):
                string = string + " "
                # if(len(string)==1 or string[len(string)] != " "):

            else:
                string = string + prev


                if i%10 == 0:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    result1 = pipe(images=Image.fromarray(img_rgb))
                    #Image.fromarray(img_rgb).save('img_rgb.jpg')

                    print(result1) 

                if(result1[0]['label'] == 'happy'):
                    sentiment='Happy'
                elif(result1[0]['label'] == 'sad'):
                    sentiment='Sad'
                elif(result1[0]['label'] == 'angry'):
                    sentiment='Angry'
                elif(result1[0]['label'] == 'surprise'):
                    sentiment='Surprised'
                elif(result1[0]['label'] == 'neutral'):
                    sentiment='Neutral'

                #print(sentiment)

        cv2.putText(img, prev, (24, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.putText(img, string, (275, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(img, sentiment, (275, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2)

        cv2.imshow("Gray", res)
        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)

        if(key == 27):  # press Esc. to exit
            break
    print(string)
    cv2.destroyAllWindows()
    source.release()

    cv2.destroyAllWindows()

    language = 'en'
    if string == ' ':
        return
    myobj = gTTS(text=string, lang=language, slow=False)

    myobj.save("audio.mp3")

    # Play the audio file
    mixer.init()
    mixer.music.load("audio.mp3")
    mixer.music.play()


def func3():
    arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    user_input = input("Enter a string: ").lower()
    for a in user_input:
        a = a.lower()

        for c in string.punctuation:
            a = a.replace(c, "")

        if a in arr:
            ImageAddress = 'assets/letters/' + a + '.jpg'
            ImageItself = Image.open(ImageAddress)
            ImageNumpyFormat = np.asarray(ImageItself)
            plt.imshow(ImageNumpyFormat)
            plt.draw()
            plt.pause(0.8)
        else:
            continue

    plt.close()



while 1:
    image = "Welcome_Screen.jpg"
    msg = "3-Way American Sign Language Translator (ASL)  \n By Harshit Maheshwari"
    choices = ["Audio to Text/Sign", "Sign to Audio/Text", "Text to Sign", "Quit"]
    reply = buttonbox(msg, image=image, choices=choices)
    if reply == choices[0]:
        func1()
    if reply == choices[1]:
        func2()
    if reply == choices[2]:
        func3()
    if reply == choices[3]:
        quit()    
