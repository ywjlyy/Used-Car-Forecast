# -*- coding: UTF-8 -*-
__author__ = 'Jason.Fan'

import csv
import os,time,datetime,re
import queue
import requests
from lxml import etree
from bs4 import BeautifulSoup
import time
import random
from threading import Lock
import threading
from multiprocessing.pool import ThreadPool

#自定义的功能导入 附加功能函数可以在utils/utils.py中定义
from utils.utils import get_header
# from setting.configs import *
from setting.configs import *
import json


class Spider(object):
    '''
    爬虫类
    '''
    def __init__(self ):
        self.base_url = "https://proconsumer.taocheche.com/c-car-consumer/carsource/gettyjucarlocallist"
        #定义你的表头在这
        self.csv_header = ['标题', '品牌','城市', '价格', '新车价格','首付', '里程', '年份','排放标准']  #暂时没用到
        if not os.path.exists('./resultdata/data.csv'):
            with open('./resultdata/data.csv','w',newline='',encoding='utf-8-sig') as f:
                csv.DictWriter(f,self.csv_header).writeheader()
            f.close()
        self.lock = Lock()
        self.myQueue = queue.Queue()
        self.thread_list=[]
        self.MaxThreadNumb = 1    #最大线程数
        self.TotalPages = 463   #总页数
        self.SetQueue()


    def SetQueue(self): #设置加入规则所需要的元素到队列
        for item in range(self.TotalPages): self.myQueue.put(item)


    def SaveData(self,carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission):
        self.lock.acquire()
        with open('./resultdata/data.csv', 'a+',encoding='utf-8-sig') as f:  # utf-8-sig  防止写入csv中文后，excel打开乱码
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission))
            f.close()
        self.lock.release()


    def getCarLink(self,url):  #获取二级链接网页内容，返回 新车价格和排放标准
            html = requests.get(url)
            # \"referprice\":\"新车含税价29.37万\",\"vendorname\
            # "emission\":\"国5\",\"mapinfo

            selector = etree.HTML(html.text)
            relis=(selector.xpath('/html/body/script[16]/text()'))
            re_result = str(relis[0])

            # patten = re.compile('新车含税价(\d+.\d+)万')
            patten = re.compile('新车含税价(.*?)万')
            result = patten.findall(re_result)
            newCarPrice = result[0]

            patten = re.compile('emission(.*?)mapinfo')
            result = patten.findall(re_result)

            To_List = [':',',','"','\\\\']
            ss = re.sub('|'.join(To_List),'',result[0]) #更换字符串中To_List中出现的内容为''空
            emission = ss  #排放标准 "国5"

            return newCarPrice,emission



    def SpiderWork(self):    #爬虫主体代码
        while not self.myQueue.empty():
            pageIdx = self.myQueue.get()
            time.sleep(random.uniform(1.2, 3.1))
            payload = json.dumps({"pageIndex":pageIdx })

            try:
                response = requests.request("POST", self.base_url, headers=Headers, data=payload)

            except Exception as e:
                for i in range(9):
                    time.sleep(random.uniform(1.2, 3.1))
                    response = requests.request("POST", self.base_url, headers=Headers, data=payload,verify=False)
                    if response.status_code == 200:
                        break
                    if i == 8:
                        print('Missing Loss ~~~~~~~~~~~~~~~~~~~!!')
                        return
            result = json.loads(response.text)
            res = result['data']['uCarBasicInfoList']['dataList']
            try:
                for i in res:
                    newCarPrice,emission = self.getCarLink(i['carLink'])
                    carName = i['carName']
                    mainBrandName = i['mainBrandName']
                    cityName = i['cityName']
                    activityPrice = i['activityPrice']
                    loanFirstPayText = i['loanFirstPayText']
                    drivingMileageText = i['drivingMileageText']
                    buyCarYear = i['buyCarYear']
                    print( str(pageIdx),'====================>',carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission)
                    self.SaveData(carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission)

            except Exception as e:
                print('Skip ================> Page No {}'.format(pageIdx))
                pass






    def run(self):
        for i in range(self.MaxThreadNumb):
            t = threading.Thread(target=self.SpiderWork,args=())
            self.thread_list.append(t)
        for t in self.thread_list:
            t.start()
        for t in self.thread_list:
            t.join()

        print('Finished Spider Work!!!!')



        # if os.path.exists(self.path):
        #     self.path = os.path.join(self.path, 'save-data')
        #     data = self.Spider()
        #     print(data)
        #     with open(os.path.join(self.path, 'Boss直聘_关键词_{}_城市_{}.csv'.format(self.keyword, self.city)), 'w',
        #               newline='', encoding='gb18030') as f:
        #         f_csv = csv.DictWriter(f, self.csv_header)
        #         f_csv.writeheader()
        #         f_csv.writerows(data)




if __name__ == '__main__':
    Spider().run()
