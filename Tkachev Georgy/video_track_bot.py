import cv2
from ultralytics import YOLO
import os
import datetime
import shutil
import numpy as np
import glob
import time
import telebot

startTime = datetime.datetime.now()

TOKEN = '6505487059:AAHV7XrTHhIHp7Z6lyY7pqoaMvnERL_RA5E'
chat_id = '-1002101119268' #'420997240'
bot = telebot.TeleBot(TOKEN)

# Папки архива и общая папка со всеми изображениями
fin = r'C:\Users\user\Desktop\thonny-4.1.3-windows-portable\user_data\RUDN\yolo_testing\finish_rudn\archive'
fin1 = r'C:\Users\user\Desktop\thonny-4.1.3-windows-portable\user_data\RUDN\yolo_testing\finish_rudn\all_frame'
if os.path.exists(fin):
    shutil.rmtree(fin)
if os.path.exists(fin1):
    shutil.rmtree(fin1)

os.mkdir(fin)
os.mkdir(fin1)

# Загрузите модель YOLOv8
model = YOLO('best.pt')

# Откройте видеофайл
video_path = r'C:\Users\user\Desktop\thonny-4.1.3-windows-portable\user_data\RUDN\yolo_testing\test_videos\01.mp4'
cap = cv2.VideoCapture(video_path)
num_cap = cap.get(7)
num = 1
num1 = 159
lst_id = set()

# Цикл по кадрам видео
while cap.isOpened():
    # Чтение кадра из видео
    ret, frame = cap.read()
    file_name = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + ".jpg"
    name_save = os.path.join(fin, file_name)
    name_save1 = os.path.join(fin1, f"{num:05}.jpg")

    if ret and num < num1:
        print(f"Кадр {num} из {num_cap}")

        # Выполните отслеживание YOLOv8 для кадра, сохраняя следы между кадрами
        results = model.track(frame,
                              classes=[1, 2],
                              verbose=False,
                              conf=0.5,
                              persist=True,
                              stream=True
        )

        for box in results:
            # print(box.boxes.data.cpu().numpy().astype(int).tolist())
            # lst_res = results[0].boxes.data.type('torch.ShortTensor').tolist()
            lst_res = box.boxes.data.cpu().numpy().astype(int).tolist()
            print(box.boxes.data)
            
            # if 1.0 in results[0].boxes.cls.tolist():
            for n, i in enumerate(lst_res):
                if i[-1] == 1:
                    x1, y1, x11, y11 = (
                        lst_res[n][0],
                        lst_res[n][1],
                        lst_res[n][2],
                        lst_res[n][3],
                    )
                if i[-1] == 2:
                    x2, y2, x21, y21 = (
                        lst_res[n][0],
                        lst_res[n][1],
                        lst_res[n][2],
                        lst_res[n][3],
                    )
                if (x2 - x1) < 20:
                    cv2.rectangle(frame, (x2, y2), (x21, y21), (0, 255, 255), 1)
                    '''
                    if y1-y > 100:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 255), 1)
                    '''
                    if i[-3] not in lst_id:
                        lst_id.add(i[-3])
                        resized = cv2.resize(frame, (640, 480))
                        cv2.imwrite(name_save, resized)
                        print("file_name", file_name)
                        #if os.stat(name_save).st_size != 0:
                            #bot.send_photo(chat_id, open(name_save, 'rb'))                        

        cv2.imwrite(name_save1, frame)
        # print(name_save1)
        # frame = results[0].plot()
        # cv2.imshow("Отслеживание", frame)
        num += 1
        print('local', len(lst_id))
        print("=======================")

        # Прервать цикл, если нажата клавиша 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Прервать цикл, если достигнут конец видео
        break

print("last", len(lst_id))
# Освободите объект захвата видео и закройте окно отображения
cap.release()
cv2.destroyAllWindows()


print(f"Сохраняли изображения - ", datetime.datetime.now() - startTime)


