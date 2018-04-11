# -*- coding: utf-8 -*-


import scrapy


class WebCrawler(scrapy.Spider):
    name = "web_crawler"
    #allowed_domains = ['http://quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com']
    
    def parse(self, response):
        # record the scraped urls
        filename = 'web_crawler_log.txt'
        with open(filename, 'a+') as f:
            f.writeln(response.url)
        #extract all urls and crawl them later
        next_pages = response.xpath('//a/@href').extract()
        for next_page in next_pages:
            if next_page is not None:
                yield response.follow(next_page, callback=self.parse)