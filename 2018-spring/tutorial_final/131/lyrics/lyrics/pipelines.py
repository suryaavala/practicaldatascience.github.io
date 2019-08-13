import json
from scrapy.exceptions import DropItem

class LyricsPipeline(object):
    
    def process_item(self, item, spider):
        if item['up']:
            item['lastWeekRank'] = str(int(item['thisWeekRank'])-int(item['up'][1:]))
            return item
        else:
            raise DropItem("Song %s does not rise in ranking" % item)

            
class JsonWriterPipeline(object):
    
    def open_spider(self, spider):
        self.file = open('processed_data.json','w')
        
    def close_spider(self,spider):
        self.file.close()
        
    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item


from scrapy.pipelines.images import ImagesPipeline

class MyImagesPipeline(ImagesPipeline): #Be careful about what to be inherited here!!!
    
    def get_media_requests(self,item,info):
        for url in item['images_urls']:
            yield scrapy.Request(url)
    
    def item_completed(self,results,item,info):
        images_paths = [x['images'] for ok, x in results if ok]
        print(images_paths)
        if not image_paths:
            raise DropItem("Song has no cover")
        item['images'] = image_paths
        return item