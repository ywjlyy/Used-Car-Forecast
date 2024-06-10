# -*- coding: UTF-8 -*-
__author__ = 'Jason.Fan'
'''
在线转化curl commands为python代码
https://curlconverter.com/
'''

# Cookies = {
#     'ASP.NET_SessionId': 'eq55j0uypictcbhgltb3wtrh',
#     '__AntiXsrfToken': 'f2bc0bbde852449e85ee58c58991455e',
# }
#
# Headers = {
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
#     'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
#     'Cache-Control': 'max-age=0',
#     'Connection': 'keep-alive',
#     # 'Cookie': 'ASP.NET_SessionId=eq55j0uypictcbhgltb3wtrh; __AntiXsrfToken=f2bc0bbde852449e85ee58c58991455e',
#     'Upgrade-Insecure-Requests': '1',
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54',
# }
#
# Params = {
#     'ItemNo': '292711',
#     'year': '2018',
#     'type': 'student',
#     'IsLXItem': '1',
# }

import json


payload = json.dumps({
  "pageIndex": 0
})
Headers = {
  'Accept': '*/*',
  'Accept-Language': 'zh-CN,zh;q=0.9',
  'Connection': 'keep-alive',
  'Content-Type': 'application/json',
  'Cookie': 'auto_id=87a224b873003e71b4a2138c009b8b2f; uuid=b5a80ec4-4c91-4ab9-8972-a641aaceddcd; _utrace=b5a80ec4-4c91-4ab9-8972-a641aaceddcd; ipCity={%22cityId%22:2103%2C%22cityName%22:%22%E7%83%9F%E5%8F%B0%22%2C%22citySpell%22:%22yantai%22%2C%22cityCode%22:%22370600%22}; t_city=201; city=%7B%22cityId%22%3A201%2C%22cityName%22%3A%22%E5%8C%97%E4%BA%AC%22%2C%22cityCode%22%3A110100%2C%22citySpell%22%3A%22beijing%22%2C%22storeId%22%3A321484%7D; storeId=321484',
  'Origin': 'https://m.taocheche.com',
  'Referer': 'https://m.taocheche.com/cars?city=beijing&storeId=321484',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.160 Safari/537.36',
  'cityId': '201',
  'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sign': '85b0d904421204cb3158b719059b183c0ead22a40bce6e538934301ea7fd283325ff0cbab3a5a014f21b33fa11be38cb3d9d047ac6dcd721ecc52db7281c1bdc',
  'storeId': '321484',
  'timestamp': '1709015763539'
}




#only debug ......
if __name__ == '__main__':
    import requests
    from lxml import etree
    import re

    url = "https://proconsumer.taocheche.com/c-car-consumer/carsource/gettyjucarlocallist"
    response = requests.request("POST", url, headers=Headers, data=payload)

    # print(response.text)

    result = json.loads(response.text)
    res = result['data']['uCarBasicInfoList']['dataList']

    for i in res:
        # print(i)
        #i['cityId']
        # ['标题', '品牌','城市', '价格', '首付', '里程', '年份']

        def getCarLink(url):
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

        nCP,eMI = getCarLink(i['carLink'])
        print('{:^50}'.format(i['carName']),
              '{:>10}'.format(i['mainBrandName']),
              '{:>10}'.format(i['cityName']),
              '{:>10}'.format(i['activityPrice']),
              '{:>10}'.format(i['loanFirstPayText']),
              '{:>10}'.format(i['drivingMileageText']),
              '{:>5}'.format(i['buyCarYear']),
              '{:>5}万'.format(nCP),
              '{:>5}'.format(eMI))




        # selectors = parsel.Selector(html.text)
        # lis = selectors.css('/html/body/div[1]/div[1]/div[6]/div[1]/div[1]/div[1]/div[2]/div[1]')
        # for li in lis:
        #     print(li)  #每个li都是一个selectors对象
        #     # href = li.css('.name a::attr(href)').get()
        #     # print(href,'-----',title,'*****',comment,'----',recomment)

        # break