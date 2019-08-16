import queue
import requests
import BeautifulSoup as bs4
import time

class NaiveCrawler:
    #@init_pg:   list of initial pages for crawling
    #@max_qsize: the max size of url_queue
    def __init__(self, init_pg, max_qsize=10000):
        self.url_queue = queue.Queue()
        self.visited_url = set(init_pg)
        for url in init_pg:
            self.url_queue.put(url)
        self.pg_cng = 0
        self.saved_pg = {}
        self.max_qsize = max_qsize
    
    #process the downloaded web page, including store/filter content 
    #and add new url to url_queue
    #@html: the html doc downloaded
    #@url:  the url of the downloaded web page 
    def process_page(self,html,url):
        soup = bs4(html, 'html.parser')
        #simply store the whole html doc
        self.saved_pg[url] = html
        self.saved_pg += 1
        for new_url in soup.find_all('a', href=True):
            if self.url_queue.qsize < self.max_qsize:
                if new_url not in self.visited_url:
                    self.url_queue.put(new_url)
                    self.visited_url.add(new_url)

    #crawling function
    #@pg_limit:  max number of pages to crawl, 0 means no upper bound
    #@max_qsize: max number of pages stored in the url_queue
    #@headers:   the http GET request headers
    #@params:    the http GET request params
    def crawl(self, pg_limit=100, 
        headers={}, params={},
        interval=0.1):
        #only download pg_limit number of web pages
        while(self.pg_cng < pg_limit):
            time.sleep(interval)
            if self.url_queue.qsize()>0:
                #get the first url in the queue
                current_url = self.url_queue.get() 
                #request for web pages  
                try:
                    response = requests.get(current_url, headers=headers, params=params)
                    html = response.text
                    self.process_page(html,current_url)
                except Exception as e:
                    print(e)
                    pass
            else:
                break