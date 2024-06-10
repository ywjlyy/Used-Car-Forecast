# -*- coding: utf-8 -*-

import requests

url_dcd = "https://www.dongchedi.com/motor/pc/sh/sh_sku_list?aid=1839&app_name=auto_web_pc"

pageidx = 0

payload = 'sh_city_name=%E5%85%A8%E5%9B%BD&page={}&limit=20'.format(pageidx)
Headers_dcd = {
    'authority': 'www.dongchedi.com',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'content-type': 'application/x-www-form-urlencoded',
    'cookie': 'ttwid=1%7CYr-D16e5JOxZP6qwnBF-HFZMtrhbQyjVzCnP0-noQT4%7C1709198197%7Cb958d94c345ebaa0c9fba4699792ce3077b64199731d8a47cbce3e2ba314efb9; tt_webid=7340950303053530633; tt_web_version=new; is_dev=false; is_boe=false; Hm_lvt_3e79ab9e4da287b5752d8048743b95e6=1709198181; Hm_lpvt_3e79ab9e4da287b5752d8048743b95e6=1709198181; _gid=GA1.2.16402841.1709198185; _gat_gtag_UA_138671306_1=1; _ga=GA1.1.1532363292.1709198185; s_v_web_id=verify_lt70gxtj_hB1tRfUf_Sv4o_4Lua_BqM9_QzKIm57HNRjz; city_name=%E7%83%9F%E5%8F%B0; _ga_YB3EWSDTGF=GS1.1.1709198184.1.1.1709198192.52.0.0',
    'origin': 'https://www.dongchedi.com',
    'referer': 'https://www.dongchedi.com/usedcar/x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-110000-2-x-x-x-x-x',
    'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.160 Safari/537.36',
    'x-forwarded-for': ''
}

Headers_dcd_sub = {
        'authority': 'vcs.zijieapi.com',
        'accept': '*/*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.dongchedi.com',
        'referer': 'https://www.dongchedi.com/usedcar/13875820',
        'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.160 Safari/537.36',
        'x-setting-flag': '1'
    }


# only debug ......
if __name__ == '__main__':
    # response = requests.request("POST", url_dcd, headers=Headers_dcd, data=payload)
    # print(response.text.encode('gbk','ignore').decode('gbk'))
    # res_new = json.loads(response.text.encode('gbk','ignore').decode('gbk'))
    #         # carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission
    # carList = (res_new['data']['search_sh_sku_info_list'])
    # for car in carList:
    #     print(car['car_name'],car['brand_name'],car['car_source_city_name'],car['car_year'],car['sh_price'],car['official_price'],car['sub_title'])

    from lxml import etree

    myurl = 'https://www.dongchedi.com/usedcar/13406367'
    headers = {
        'authority': 'vcs.zijieapi.com',
        'accept': '*/*',
        'accept-language': 'zh-CN,zh;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.dongchedi.com',
        'referer': 'https://www.dongchedi.com/usedcar/13406367',
        'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.160 Safari/537.36',
        'x-setting-flag': '1'
    }
    html = requests.get(myurl, headers=headers)
    print(html.text.encode('gbk', 'ignore').decode('gbk'))
    selector = etree.HTML(html.text)
    firstpay = selector.xpath('//*[@id="__next"]/div/div[2]/div/div[2]/div[2]/div[4]/div[2]/div/text()')[0]
    print(firstpay)
    import re
    pattern = re.compile('\d+.\d+|\d+')
    firstpay = re.findall(pattern,firstpay)
    print(firstpay)

    emission = selector.xpath('//*[@id="__next"]/div/div[2]/div/div[2]/div[2]/div[5]/div/div[3]/p[1]/text()')[0]
    print(emission)
    emission = (emission.encode('utf-8').decode('utf-8'))
    print(emission)

