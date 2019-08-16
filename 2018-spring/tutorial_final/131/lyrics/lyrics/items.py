import scrapy

class LyricsItem(scrapy.Item):
    song = scrapy.Field()
    artist = scrapy.Field()
    url = scrapy.Field()
    lyric = scrapy.Field()
    thisWeekRank = scrapy.Field()
    lastWeekRank = scrapy.Field()
    up = scrapy.Field()
    same = scrapy.Field()
    down = scrapy.Field()
    images_urls = scrapy.Field()
    images = scrapy.Field()