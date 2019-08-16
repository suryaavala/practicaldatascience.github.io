# -*- coding: utf-8 -*-
import scrapy


class GuntenbergCrawlerSpider(scrapy.Spider):
    name = 'guntenberg_crawler'
    allowed_domains = ['www.gutenberg.org']
    start_urls = ['http://www.gutenberg.org/']

    def parse(self, response):
        pass
