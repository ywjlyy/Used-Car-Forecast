# -*- coding: UTF-8 -*-
__author__ = 'Jason.Fan'
import time
from faker import Faker   #用它来生成各种各样的伪数据
user_agent = Faker('zh-CN').user_agent()  #伪代理
# print(user_agent)


def get_user_agent():
    return user_agent

def get_header():
    return {
        'User-Agent': user_agent
    }

def get_time():
    return int(time.time())


def get_name():
    return Faker(locale='zh_CN').name()    #locale='zh_TW' 繁体

def get_address():
    return Faker(locale='zh_CN').address()

def get_phonenumber():
    return Faker().phone_number()



if __name__ == '__main__':
    print(get_name())
    print(get_address())
    print(get_phonenumber())

    for i in range(20):
        print(Faker('zh-CN').user_agent()) 