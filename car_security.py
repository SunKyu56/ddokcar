import cv2
import numpy as np
from os import makedirs, listdir
import os
from os.path import isdir, isfile, join
import pygame

import RPi.GPIO as GPIO # 라즈베리파이 GPIO 핀을 쓰기위해 임포트
import time # 시간 간격으로 제어하기 위해 임포트

# 얼굴 저장 함수
face_dirs = 'faces/' #얼굴 저장하는 파일 경로
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(-1)

# 얼굴 검출 함수
def face_extractor(img):
    color = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    faces = face_classifier.detectMultiScale(color,1.3,5)
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face

# 얼굴만 저장하는 함수
def take_pictures(name):
    # 해당 이름의 폴더가 없다면 생성

    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)

    # 카메라 ON
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()

        
        
        
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 
        if face_extractor(frame) is not None:
            
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame),(200,200))
            # 컬러로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)


            # 200x200  사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = face_dirs + name + '/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        
        # 얼굴 사진을 다 얻었거나 esc키 누르면 종료
        if cv2.waitKey(1)==27 or count==50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')
    
# 사용자 얼굴 학습    
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬         
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 리턴
    return model

# 여러 사용자 학습
def trains():
    #faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models    
#얼굴 검출
def face_detector(img, size = 0.5):

        color = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        faces = face_classifier.detectMultiScale(color,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


# 인식 시작
def run(models):    
    #카메라 열기
    cap = cv2.VideoCapture(0)
    start = time.time()
    end = 5
    
    
 
    while True:
        #카메라로 부터 사진 한장 읽기 
        #ret, frame = cap.read()
        #image, face = face_detector(frame)
        #frame = cv2.flip(frame, 1)
        # 얼굴 검출 시도
        detect = False
        temp = True
        while ((time.time() - start) <  5 and detect) or temp:
            temp = False
            ret, frame = cap.read()
            image, face = face_detector(frame)
            if face is not None and not detect:
                start = time.time()
                detect = True
            cv2.imshow('Face X', image)
            cv2.waitKey(1)
            print(time.time())

        
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            #검출된 사진을 컬러으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    print(min_score)
                    
            #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.         
            if min_score < 80:
                #????? 어쨋든 0~100표시하려고 한듯 
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
                cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            #75 보다 크면 동일 인물로 간주해 UnLocked!
                confidence = int(100*(1-(min_score)/300))
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
                cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
                if confidence > 75:
                    cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', image)
                    return min_score_name
                else:
                    print("yjyj")
                    return False
            else:
                print("yjyj")
                return False
        except:
            #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face X', image)
            pass
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
def display(user_name):

    current_path = os.path.dirname(__file__)
    face_path = '/home/pi/Desktop/fdfd/faces/'+ str(user_name)
    image_path = '/home/pi/Desktop/fdfd/images/'
    size = [800,600] # diplay size
    display_width = size[0]
    display_height = size[1] 
    pygame.init()
    pygame.display.set_caption("Car User Interface")
    font = pygame.font.SysFont(None,50)
    font_small = pygame.font.SysFont(None,30)
    text_color = (0,0,0)
    
    path = '/home/pi/Desktop/fdfd/faces/'+ user_name + '/setting.txt'
    
    n=0
    with open(path, 'r') as f:
        example = f.readlines()
        for line in example:
            line = line.strip()
            line = int(line)
            if n==0:
                hd_degree = line
                n += 1
            elif n==1:
                seat_degree = line
                n += 1
            else:
                 pass
            
    running = True
    while running:
        
        screen=pygame.display.set_mode(size)
        start = pygame.transform.scale(pygame.image.load("start.png"),(display_width,display_height))
        car = pygame.transform.scale(pygame.image.load("car.png"),(display_width,display_height))
        screen.blit(start,(0,0))
        screen.blit(car,(90,240))
        user_face = pygame.image.load(os.path.join(face_path, "user31.jpg")) #new 사진중 아무거나 로드
        screen.blit(user_face,(10,10))
        profile_text = font.render("Hello, It's Your car profile" ,True,text_color)
        name_text = font_small.render(' - Name : ' + user_name,True,text_color)
        handle_text = font_small.render(' - Handle : ' + str(hd_degree)+' Degree',True,text_color)
        seat_text = font_small.render(' - Seat : ' + str(seat_degree)+' Degree',True,text_color)

        screen.blit(profile_text,(240,30))
        screen.blit(name_text,(300,80))
        screen.blit(handle_text,(300,120))
        screen.blit(seat_text,(300,160))
        pygame.display.update()


        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
                if event.type == pygame.QUIT :   #종료처리
                    running=False
                    pygame.quit()
    
def user_setting(name):
    
    path = '/home/pi/Desktop/fdfd/faces/'+ name + '/setting.txt'
    pin = 36
    with open(path, 'r') as f:
        example = f.readlines()
        # example = example.strip()
        for line in example:
            line = line.strip()
            line = int(line)
            servoMotor(pin, line, 1) # 신호선을 16번 핀에 연결, 8의 각도로 1초동안 실행
            pin = pin + 1
    
    
  
def servoMotor(pin, degree, t):
    print(pin)
    print(degree)
    SERVO_MAX_DUTY = 12   # 서보의 최대(180도) 위치의 주기
    SERVO_MIN_DUTY= 3    # 서보의 최소(0도) 위치의 주기
    GPIO.setmode(GPIO.BOARD) # 핀의 번호를 보드 기준으로 설정, BCM은 GPIO 번호로 호출함
    GPIO.setup(pin, GPIO.OUT) # GPIO 통신할 핀 설정
    servo = GPIO.PWM(pin, 50)  # 서보핀을 PWM 모드 50Hz로 사용하기 (50Hz > 20ms)
    servo.start(0)  # 서보 PWM 시작 duty = 0, duty가 0이면 서보는 동작하지 않는다.

    if pin == 36:
        duty = SERVO_MIN_DUTY+(170*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
        servo.ChangeDutyCycle(duty)
        time.sleep(t) # 서보모터가 이동할만큼의 충분한 시간을 입력. 너무 작은 값을 입력하면 이동하다가 멈춤
        duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
        servo.ChangeDutyCycle(duty)
        time.sleep(t)
    else:
        duty = SERVO_MIN_DUTY+(0*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
        servo.ChangeDutyCycle(duty)
        time.sleep(t) # 서보모터가 이동할만큼의 충분한 시간을 입력. 너무 작은 값을 입력하면 이동하다가 멈춤
        duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
        servo.ChangeDutyCycle(duty)
        time.sleep(t)
        # 아래 두줄로 깨끗하게 정리해줘야 다음번 실행할때 런타임 에러가 안남
    servo.stop() 
    GPIO.cleanup(pin)

def new(name):
    take_pictures(name)
    train(name)
    print("name ok")
    f = open("/home/pi/Desktop/fdfd/faces/"+name+"/setting.txt", 'w')
    f.close()
    print("file ok")
    f = open("/home/pi/Desktop/fdfd/faces/"+name+"/setting.txt", 'a')
    h = input("handle : ")
    c = input("char : ")
    f.write(h)
    f.write("\n")
    f.write(c)
    f.close()
    print("setting file ok")
    display(name)
    user_setting(name)
    
    



if __name__ == "__main__":
    # 학습 시작
    models = trains()
    result = run(models)
    
    if result == False:
        print("new") 
        new_user = input("이름을 입력해 주세요 : ")
        new(new_user)
    else:
        print("setting")
        print(result)
        user_setting(result)
        display(result)
        