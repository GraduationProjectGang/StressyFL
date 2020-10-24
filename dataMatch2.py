import json
import time
import pprint
from ast import literal_eval
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import time
import pprint
import csv   
    
jsonPath = './data.json'
statspath = './appstats.json'
dataAll={}
stressArr = []

client_num = 1

with open(jsonPath, encoding= 'UTF-8') as json_file:
    data = json.load(json_file)
   
with open(statspath, encoding= 'UTF-8') as file:
    statsData = json.load(file)

    for userKey in data:

        client_training = {}
        client_stress = []

        for item in data[userKey]: #jsonItem - user,timestamp,ifMoving,posture,posture_accuracy,std_posture,orientation
            for user in statsData:
                if userKey == user: # 여기서 유저 일치 여부 확인하고 데이터를 쓴다
                    for coroutine in statsData[userKey]:
                        index = 0

                        

                        if item['timestamp'] == statsData[userKey][coroutine]['timestamp']:
                            dataAll[coroutine] = []

                            client_training[coroutine] = []

                            stressCount = int(item['stressCount'])
                            if stressCount >=0 and stressCount <= 3:
                                stressLabel = 0
                            if stressCount >=4 and stressCount <= 7 :
                                stressLabel = 1
                            if stressCount >=8 and stressCount <= 11:
                                stressLabel = 2
                            if stressCount >=12 and stressCount <= 16:
                                stressLabel = 3
                            addFlag = True
                            for apps in statsData[userKey][coroutine]:
                                
                                if len(apps) == 1:#timestamp가 아니라 app이면
                                    temp = statsData[userKey][coroutine][apps]

                                    if temp == 0:
                                        dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],0,0])
                                        # dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],0,0])
                                        # print(dataAll[coroutine][len(dataAll[coroutine]-1)])
                                        client_training[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],0,0])


                                    elif 'category' in temp:
                                        dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],temp['category'],temp['totalTimeInForeground']])
                                        # dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],temp['category'],temp['totalTimeInForeground']])
                                        # print(dataAll[coroutine][len(dataAll[coroutine]-1)])
                                        # stressArr.append(stressLabel)
                                        client_training[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],temp['category'],temp['totalTimeInForeground']])
                                                                      
                                    else:
                                        addFlag = False
                                    index += 1

                            if addFlag is True:
                                stressArr.append(stressLabel)

                                client_stress.append(stressLabel)

                            else:
                                # print(dataAll[coroutine])
                                del dataAll[coroutine]
                                del client_training[coroutine]
                                
            
        print(userKey)
        if len(client_training.values()) == len(client_stress):
            for idx, row in enumerate(client_training.values()):
                writeFilePath_t = './data/trainingData_' + str(client_num) + '.csv'
                writeFilePath_s = './data/stressData_' + str(client_num) + '.csv'
                rowlist = list(row)
                with open(writeFilePath_t,'w',newline='\n') as file:
                    for each_row in rowlist:
                        cw = csv.writer(file)
                        cw.writerow(each_row)

                with open(writeFilePath_s,'w',newline='') as file:
                    cw = csv.writer(file)
                    for i in range(5):
                        cw.writerow([client_stress[idx]])
                client_num = client_num + 1
                
        else:
            print("userKey %s, %f, %f"%(userKey, len(client_training.values()) ,len(client_stress)))
            client_training = {}
            client_stress = []


print(client_num)