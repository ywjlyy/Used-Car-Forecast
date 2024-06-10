# -*- coding: utf-8 -*-
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')  #改变标准输出的默认编码 ,如果报错请加入此行
# 网页bai的数据应该是'utf-8'编码duzhi,这个可以在网页的head上面看得到,
# zhuan网页的时候会把它转化shu成Unicode,出问题的是在print()这儿,
# 对于print()这个函数,他需要把内容转化为'gbk'编码才能显示出来. 然后解决办法是这样,
# 你在转化后的Unicode编码的string后面,加上 .encode('gbk','ignore').decode('gbk') 也就是先用gbk编码,忽略掉非法字符,然后再译码,现在解决了
__author__ = 'Jason.Fan'
'''


'''
from multiprocessing import Pool
from core.Spider import Spider
from core.Spider_dongchedi import Spider_dongchedi


def main():
    pool = Pool(processes=4)  # cpu数量
    for i in [Spider().run(), ]:    #可以加多个爬虫任务threading在列表中
    # for i in [Spider_dongchedi().run(), ]:    #可以加多个爬虫任务threading在列表中
        pool.apply_async(i)  # 非阻塞的方式
    pool.close()  # 关闭进程池，不在接受新的任务
    pool.join()  # 主进程阻塞等待子进程退出，join方法要在close , terminate之后用


if __name__ == '__main__':
    main()
