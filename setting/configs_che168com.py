import requests
import json
from lxml import etree
from urllib.parse import parse_qs, urlparse, urlencode, unquote, unquote_plus, quote, quote_plus

# url = 'https://m.che168.com/beijing/list/'
numblis = list(range(1,100,3))

payload = {}
headers = {
  'authority': 'apiuscdt.che168.com',
  'accept': '*/*',
  'accept-language': 'zh-CN,zh;q=0.9',
  'cookie': 'fvlid=1709175886109AvTMm1YOLaCz; Hm_lvt_d381ec2f88158113b9b76f14c497ed48=1709175886; sessionid=c1a59cb4-ef95-471f-bf3a-fd1c2f67822b; sessionip=60.212.41.5; area=370611; che_sessionid=F91014E0-FEE1-49C6-9210-4C0A9E80661D%7C%7C2024-02-29+11%3A05%3A04.520%7C%7Cwww.baidu.com; userarea=0; listuserarea=0; Hm_lpvt_d381ec2f88158113b9b76f14c497ed48=1709176191; showNum=4; sessionvisit=1b9d5d45-8786-463b-bbc7-f1b043ebaad3; sessionvisitInfo=c1a59cb4-ef95-471f-bf3a-fd1c2f67822b|pcm.che168.com|105564; che_sessionvid=A6F0D762-AEB0-4C7B-AE17-5364A2965B7C; _home_tofu_=5%2C11%2C12%2C6%2C10; uarea=110000%7Cbeijing; v_no=63; visit_info_ad=F91014E0-FEE1-49C6-9210-4C0A9E80661D||A6F0D762-AEB0-4C7B-AE17-5364A2965B7C||-1||-1||63; che_ref=www.baidu.com%7C0%7C0%7C0%7C2024-02-29+14%3A38%3A43.068%7C2024-02-29+11%3A05%3A04.520; ahpvno=73; ahuuid=1B2D0297-4369-4EEA-A6B0-20E2102469F0; sessionuid=c1a59cb4-ef95-471f-bf3a-fd1c2f67822b',
  'origin': 'https://pcm.che168.com',
  'referer': 'https://pcm.che168.com/2023/cardetail_rn/index?infoid=50122303&pvareaid=108223',
  'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'same-site',
  'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.160 Safari/537.36'
}



def parse_url_params(url):
    """将url ？后的参数转成字典"""
    # 提取url参数
    query = urlparse(url).query
    # 将字符串转换为字典
    params = parse_qs(query)
    # 所得的字典的value都是以列表的形式存在，若列表中都只有一个值
    result = {key: params[key][0] for key in params}
    return result

def savedate(carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission):
    with open('./resultdata/data2.csv', 'a+',encoding='utf-8-sig') as f:  # utf-8-sig  防止写入csv中文后，excel打开乱码
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(carName, mainBrandName, cityName, activityPrice,newCarPrice, loanFirstPayText, drivingMileageText,buyCarYear,emission))
        f.close()



for n in numblis:
    url = 'https://m.che168.com/beijing/a0_0msdgscncgpi1ltocsp{}exa0/'.format(n)
    resp = requests.get(url)
    selector = etree.HTML(resp.text)

    relis = (selector.xpath('//*[@id="CarList"]/a'))
    loopnumb = (len(relis))

    for i in range(1,loopnumb):

        carName = selector.xpath('//*[@id="CarList"]/a[{}]/div[2]/h3/text()'.format(i))[0]
        url_2nd = 'https:'+selector.xpath('//*[@id="CarList"]/a[{}]/@href'.format(i))[0]
        print(carName)
        print(url_2nd)
# 奔驰E级 2021款 E 300 L 豪华型
# http://pcm.che168.com/2023/cardetail_rn/index?infoid=50200989&pvareaid=108223



    break
