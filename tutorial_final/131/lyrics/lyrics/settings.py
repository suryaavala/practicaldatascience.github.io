
BOT_NAME = 'lyrics'

SPIDER_MODULES = ['lyrics.spiders']
NEWSPIDER_MODULE = 'lyrics.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True


# Configure item pipelines

ITEM_PIPELINES = {
   'scrapy.pipelines.images.ImagesPipeline': 100,
   'lyrics.pipelines.LyricsPipeline': 300,
   'lyrics.pipelines.JsonWriterPipeline': 800,
}

IMAGES_STORE = 'pics_scraped'