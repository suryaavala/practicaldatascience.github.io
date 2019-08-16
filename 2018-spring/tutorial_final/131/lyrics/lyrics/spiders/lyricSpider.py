import scrapy
from lyrics.items import LyricsItem

class LyricSpider(scrapy.Spider):
    name = 'lyricSpider'  #name for each spider should be unique
    allowed_domains = ["www.metrolyrics.com"]
    start_urls = ["http://www.metrolyrics.com/top100.html"] #where to start scraping
    
    
    #the parse() method will be called by defualt
    def parse(self, response):
        #use Scrapy's Selector to select certain parts of the HTML source:
        slc = scrapy.Selector(response)
        sites = slc.xpath('//div[@id="top100"]/div[@class="row"]/div/div[@class="grid_8"]/div/div/ul/li/span[3]')
        
        items = []        
        for site in sites:
            item = LyricsItem()
            item['song'] = site.xpath('a/text()').extract_first().strip()[:-7] #get rid of the 'Lyric' word of the text.
            item['artist'] = site.xpath('span/a/text()').extract()[0].strip() #extract()[0] and extract_first() does the same thing.
            item['url'] = site.xpath('a/@href').extract()[0]
            items.append(item)            
        
        return items   