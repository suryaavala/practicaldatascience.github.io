# -*- coding: utf-8 -*-
import scrapy


class QuoraSpider(scrapy.Spider):
    name = 'quora'
    #allowed_domains = ['www.quora.com']
    start_urls = ['http://www.zhihu.com']

    def parse(self, response):
        pass
