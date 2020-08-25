import json
import time
import pprint
from openpyxl import Workbook
from ast import literal_eval
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import time
from rotate import getRotateVec
import pprint 
import csv   
    
jsonPath = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\data.json'
statspath = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\appstats.json'
dataAll={}
stressArr = []

with open(jsonPath, encoding= 'UTF-8') as json_file:
    data = json.load(json_file)
   
with open(statspath, encoding= 'UTF-8') as file:
    statsData = json.load(file)

    for userKey in data:
        for item in data[userKey]: #jsonItem - user,timestamp,ifMoving,posture,posture_accuracy,std_posture,orientation
            for user in statsData:
                if userKey == user:
                    for coroutine in statsData[userKey]:
                        
                        index = 0 
                        if item['timestamp'] == statsData[userKey][coroutine]['timestamp']:
                            dataAll[coroutine] = []
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
                                                             
                                    elif 'category' in temp:
                                        dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],temp['category'],temp['totalTimeInForeground']])
                                        # dataAll[coroutine].append([item['ifMoving'],item['orientation'],item['posture'],item['std_posture'],temp['category'],temp['totalTimeInForeground']])
                                        # print(dataAll[coroutine][len(dataAll[coroutine]-1)])
                                        # stressArr.append(stressLabel)
                                                                      
                                    else:
                                        addFlag = False
                                    index += 1

                            if addFlag is True:
                                stressArr.append(stressLabel)
                            else:
                                print(dataAll[coroutine])
                                del dataAll[coroutine]
                                
                                    

print(len(dataAll))
print(len(stressArr))
with open('trainingData2.csv','w',newline='') as file:
    for i in list(dataAll.values()):
        cw = csv.writer(file)
        cw.writerow(i)

        
with open('stressData2.csv','w',newline='') as file:
    cw = csv.writer(file)
    cw.writerow(stressArr)