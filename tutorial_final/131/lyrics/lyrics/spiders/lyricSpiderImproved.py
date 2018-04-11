import scrapy
from lyrics.items import LyricsItem

class LyricSpiderImproved(scrapy.Spider):
    name = 'lyricSpiderImproved' 
    allowed_domains = ["www.metrolyrics.com"]
    start_urls = ["http://www.metrolyrics.com/top100.html"]
    
    def parse(self, response):
        slc2 = scrapy.Selector(response)
        sites = slc2.xpath('//*[@id="main-content"]/div[1]/div/div/ul/li')
    
        for site in sites:
            item = LyricsItem()
            item['song'] = site.xpath('span[3]/a/text()').extract_first().strip()[:-7] 
            item['artist'] = site.xpath('span[3]/span/a/text()').extract_first().strip()
            item['url'] = site.xpath('span[3]/a/@href').extract_first()

            item['thisWeekRank'] = site.xpath('span[1]/text()').extract_first()
            item['up'] = site.xpath('div[@class="last-week up"]/text()').extract_first()
            item['same'] = site.xpath('div[@class="last-week same"]/text()').extract_first()
            item['down'] = site.xpath('div[@class="last-week down"]/text()').extract_first()
            
            item['images_urls']= site.xpath('span[2]/a/img/@src').extract_first()

            yield scrapy.Request(url = item['url'], meta = {'item':item}, callback = self.parse_lyric)
            
            yield scrapy.Request(url = item['url'], meta = {'item':item}, callback = self.parse_lyric)

    
    def parse_lyric(self, response):
        slc3 = scrapy.Selector(response)
        textLines= slc3.xpath('//*[@id="lyrics-body-text"]/p/text()').extract()
        #print(textLines)
        
        item = response.meta['item']
        item['lyric'] = " ".join(textLines)
        yield item