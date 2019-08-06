---
layout: page
title: Data collection and scraping
img: table.png
---

[Download notes as Jupyter notebook](data_collection.tar.gz)

## Introduction

The first step of most data science pipelines, as you may imagine, is to get some data.  Data that you typically use comes from many different sources.  If you're lucky, someone may hand directly had you a file, such as a CSV (comma separated value) file or something similar, which they want you to analyze directly.  Or sometimes you'll need to issue a database query to collect the relevant data (we'll discuss relational databases in a later lecture).  But in this lecture, we'll talk about collecting data from two main sources: 1) querying an API (the majority of which are web-based, these days); and 2) scraping data from a web page.  The latter case is a common method to extract data in and of itself, but it also serves as a general example for parsing data from (relatively) _unstructured_ sources.  Data that you get "in the wild" typically needs substantial preprocessing before you actually use it for data science tasks (this applies even to seemingly structured data sources like CSV files or API results), and some of the techniques we will cover in this lecture apply equally well to processing any sort of unstructured data.

## Collecting data from web-based sources

With those general caveats in mind, let's dive a bit more deeply into the specific case of gather data from an web-based source, which is one of the more common forms of querying data.  It will also serve as an introduction to the type of Python coding that you'll do in this class.

The first step of collecting web-based data is to issue a request for this data via some protocol: HTTP (HyperText Transfer Protocol) or HTTPS (the secure version).  And while I know that one of the principles of this course is to teach you how things work "under the hood" as well the common tools for doing so, we won't be concerned at all with the actual HTTP protocol or how these methods work in any detail; for our purposes, we're going to use the [requests](http://docs.python-requests.org/en/master/) library in Python.

Let's see how this works with some code.  The following code will load data from the course webpage:


```python
import requests
response = requests.get("http://www.cmu.edu")

print("Status Code:", response.status_code)
print("Headers:", response.headers)
```

```
Status Code: 200
Headers: {'Date': 'Mon, 22 Jan 2018 02:39:55 GMT', 'Server': 'Apache', 'x-xss-protection': '1; mode=block', 'x-content-type-options': 'nosniff', 'x-frame-options': 'SAMEORIGIN', 'Vary': 'Referer', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=7200, must-revalidate', 'Expires': 'Mon, 22 Jan 2018 04:39:55 GMT', 'Keep-Alive': 'timeout=5, max=500', 'Connection': 'Keep-Alive', 'Transfer-Encoding': 'chunked', 'Content-Type': 'text/html'}
```

This code issues an "HTTP GET" request to load the content of the paper, and returns it in the `response` object.  The `status_code` field contains the "200" code, which indicates a successful query, and the `headers` field contains meta-information about the page (in this case, you could see, for instance, that despite the URL, we're actually hosting this page on github).  If you want to see the actual content of the page, you can use the `response.content` or `response.text` fields, as below.


```python
print(response.text[:480])
```

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="IE=edge" http-equiv="X-UA-Compatible"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Homepage -     CMU - Carnegie Mellon University</title>    
    <meta content="CMU is a global research university known for its world-class, interdisciplinary programs: arts, business, computing, engineering, humanities, policy and science." name="description"/>
  
```

**Important note:** There's one very important point here, which may be obvious to you if you've spend substantial time doing any kind of software development, but if most of your experience with programming is via class exercises, it may not be completely apparent, so I emphasize it here.  You will see code samples like this throughout the course, in the slides and in these notes. It's important _not_ to take this to mean that you should memorize these precise function calls, or even do anything other than just scan over them briefly.  As a data scientist, you'll be dealing with hundreds of different libraries and APIs, and trying to commit them all to memory is not useful.  Instead, what you need to develop is _the ability to quickly find a library and function call that you need to accomplish some task_.  For example, even if you know nothing about the in this case, you want to download the content of some URL.  You can type into Google something like ["Python download url content"](https://www.google.com/search?q=python+download+url+content) (I just picked this precise phrasing randomly, feel free to try some variants on this).  The first result for my search is a Stack Overflow page: [How do I download a file over HTTP using Python?](https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python).  While the first response actually lists the `urllib2` package (this was the more common library at one point, but the `requests` library provides a simpler interface that does things like automatically encode parameters to urls and other niceties), the `requests` library [home page](http://docs.python-requests.org/en/master/) is a few responses down.  And once you find the home page for that library, the very first example on the page shows how to use it for simple calls like the one above.  You can look through documentation here, but like above, if you have a question about the `requests` library, you can likely use good for a direct answer there too.  For instance, if you want to learn to use the POST command, you can Google something like "python requests library post command" and the searches will either bring you straight to the relevant requests documentation or to a Stack Overflow page.

With all that in mind, let's look at a few more calls we can issue with the library.  You probably have seen URLS like this before

    https://www.google.com/search?q=python+download+url+content&source=chrome
    
The `https://www.google.com/search` string is the URL, and everything after the ? are _parameters_; each parameter is of the form "parameter=value" and are separated by ampersands &.  If you've seen URLS before you've noticed that a lot of content needs to be encoded in these parameters, such as spaces replaces with the code "%20" (the Google url above can also handle the "+" character, but "%20" is the actual encoding of a space).  Fortunately, `requests` handles all of this for you.  You can simply pass all the parameters as a Python dictionary.


```python
params = {"query": "python download url content", "source":"chrome"}
response = requests.get("http://www.google.com/search", params=params)
print(response.status_code)
```

```
200
```

Besides the HTTP GET command, there are other common HTTP commands (POST, PUT, DELETE) which can also be called by the corresponding function in the library.

### RESTful APIs

While parsing data in HTML (the format returned by these web queries) is sometimes a necessity, and we'll discuss it further before, HTML is meant as a format for displaying pages visually, not as the most efficient manner for encoding data.  Fortunately, a fair number of web-based data services you will use in practice employ something called REST (Representational State Transfer, but no one uses this term) APIs.  We won't go into detail about REST APIs, but there are a few main feature that are important for our purposes:

1. You call REST APIs using standard HTTP commands: GET, POST, DELETE, PUT.  You will probably see GET and POST used most frequently.
2. REST servers don't store state.  This means that each time you issue a request, you need to include all relevant information like your account key, etc.
3. REST calls will usually return information in a nice format, typically JSON (more on this later).  The `requests` library will automatically parse it to return a Python dictionary with the relevant data.

Let's see how to issue a REST request using the same method as before.  We'll here query my GitHub account to get information.  More info about GitHub's REST API is available at their [Developer Site](https://developer.github.com/v3/).


```python
# Get your own at https://github.com/settings/tokens/new
token = "3125e4430a58c5259a14ddd48157061cdb7055c0" 
response = requests.get("https://api.github.com/user", params={"access_token":token})

print(response.status_code)
print(response.headers["Content-Type"])
print(response.json().keys())
```

```
200
application/json; charset=utf-8
dict_keys(['login', 'id', 'avatar_url', 'gravatar_id', 'url', 'html_url', 'followers_url', 'following_url', 'gists_url', 'starred_url', 'subscriptions_url', 'organizations_url', 'repos_url', 'events_url', 'received_events_url', 'type', 'site_admin', 'name', 'company', 'blog', 'location', 'email', 'hireable', 'bio', 'public_repos', 'public_gists', 'followers', 'following', 'created_at', 'updated_at', 'private_gists', 'total_private_repos', 'owned_private_repos', 'disk_usage', 'collaborators', 'two_factor_authentication', 'plan'])
```

The token element there (that is an example that was linked to my account, which I have since deleted, you can get your own token for your account at https://github.com/settings/tokens/new) identifies your account, and because this is a REST API there is no "login" procedure, you just simply include this token with each call to identify yourself.  The call here is just a standard HTTP request: it requests the URL `https://api.github.com/user` passing our token as the parameter `access_token`.  The response looks similar to our above response, except if we look closer we see that the "Content-Type" in the HTTP header is "application/json".  In these cases, the `requests` library has a nice function, `response.json()`, which will convert the returned data into a Python dictionary (I'm just showing the keys of the dictionary here).

### Authentication

Most APIs will use an authentication procedure that is more involved than this example above.  The standard here for a while was called "Basic Authentication", and can be used via the `requests` library by simply passing the login and password as the `auth` argument to the relevant calls, as below. 


```python
response = requests.get("https://api.github.com/user", auth=("zkolter", "github_password"))
print(response.status_code)
```

```
200
```

As seen above, GitHub does support Basic Authentication, though it's becoming less common in a majority of APIs.  Instead, most APIs use something called OAuth, which you'll use a little bit in the first homework.

## Common data formats and handling

Now that you've obtained some data (either by requesting it from a web source, or just getting a file sent to you), you'll need to know how to handle that data format.  Obviously, data comes in many different formats, but some of the more common ones that you'll deal with as a data scientist are:

- CSV (comma separated value) files
- JSON (Javascript object notation) files and string
- HTML/XML (hypertext markup language / extensible markup language) files and string


### CSV files

The "CSV" name is really a misnomer: CSV doesn't only refer to comma separated values, but really refers to any delimited text file (for instance, fields could be delimited by spaces or tabs, or any other character, specific to the file).  For example, let's take a look at the following data file describing weather data near at Pittsburg airport:

It can be surprisingly hard to find historical weather data in CSV format (most large weather sites charge for API access, and the official National Weather Service historical data is in a custom, hard-to-parse format).  So as a shameless plug I'll note that I downloaded this data from [http://wefacts.org](http://wefacts.org) which is a site created by [Xiao Zhang](https://shawxiaozhang.github.io), a former PhD student of mine, that gives an easy interface for querying relatively large amounts of historical data in CSV form.  Description of the meaning of each data column above is [here](https://shawxiaozhang.github.io/wefacts/), but the important points are that the first two columns are time (UTC and local), and for example the third column is degrees Celsius scaled by 10.

To parse CSV files in Python, the most common library to use is [Pandas](https://pandas.pydata.org/), which we will cover a lot more later in this course.  For the purposes of this lecture, though, we'll just note that we can load the data using the following code:


```python
import pandas as pd
dataframe = pd.read_csv("kpit_weather.csv", delimiter=",", quotechar='"')
dataframe.head()
```


<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
        font-size: 10pt;
        padding: 4px;
    }

    .dataframe thead th td {
        text-align: left;
        font-size: 10pt;
        padding: 4px;
    }

    .dataframe tbody tr th {
        vertical-align: top;
        font-size: 10pt;
        padding: 4px;
    }

    .dataframe td {
        vertical-align: top;
        font-size: 10pt;
        padding: 4px;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ZTime</th>
      <th>Time</th>
      <th>OAT</th>
      <th>DT</th>
      <th>SLP</th>
      <th>WD</th>
      <th>WS</th>
      <th>SKY</th>
      <th>PPT</th>
      <th>PPT6</th>
      <th>Plsr.Event</th>
      <th>Plsr.Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170820040000</td>
      <td>20170820000000</td>
      <td>178</td>
      <td>172</td>
      <td>10171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-9999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170820050000</td>
      <td>20170820010000</td>
      <td>178</td>
      <td>172</td>
      <td>10177</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-9999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170820060000</td>
      <td>20170820020000</td>
      <td>167</td>
      <td>161</td>
      <td>10181</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-9999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170820070000</td>
      <td>20170820030000</td>
      <td>161</td>
      <td>161</td>
      <td>10182</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>-9999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170820080000</td>
      <td>20170820040000</td>
      <td>156</td>
      <td>156</td>
      <td>10186</td>
      <td>180</td>
      <td>15</td>
      <td>-9999</td>
      <td>0</td>
      <td>-9999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div></small></div>
We don't actually need the `delimiter` or `quotechar` arguments here, because the default argument for delimiter is indeed a comma (which is what this CSV file is using), but you can pass an additional argument to this function to use a different delimiter.  One issue that can come up is if any of the values you want to include contain this delimiter; to get around this, you can surround the value with the `quotechar` character.  Several CSV files will just include quotes around any entry, by default.  Again, our file here doesn't contain quotes, so it is not an issue, but its it a common occurrence when handling CSV files.  One final thing to note is that by default, the first row of the file a header row that lists the name of each column in the file.  If this is not in the file, then you can load the data with the additional `header=None` argument.

### JSON data

Although originally built as a data format specific to the Javascript language, JSON (Javascript Object Notation) is another extremely common way to share data.  We've already seen in it with the GitHub API example above, but very briefly, JSON allows for storing a few different data types:

- Numbers: e.g. `1.0`, either integers or floating point, but typically always parsed as floating point
- Booleans: `true` or `false` (or `null`)
- Strings: `"string"` characters enclosed in double quotes (the `"` character then needs to be escaped as `\"`)
- Arrays (lists): `[item1, item2, item3]` list of items, where item is any of the described data types
- Objects (dictionaries): `{"key1":item1, "key2":item2}`, where the keys are strings and item is again any data type

Note that lists and dictionaries can be nested within each other, so that, for instance

    {"key1":[1.0, 2.0, {"key2":"test"}], "key3":false}

would be a valid JSON object.

Let's look at the full JSON returned by the GitHub API above:


```python
print(response.content)
```

```
b'{"login":"zkolter","id":2465474,"avatar_url":"https://avatars1.githubusercontent.com/u/2465474?v=4","gravatar_id":"","url":"https://api.github.com/users/zkolter","html_url":"https://github.com/zkolter","followers_url":"https://api.github.com/users/zkolter/followers","following_url":"https://api.github.com/users/zkolter/following{/other_user}","gists_url":"https://api.github.com/users/zkolter/gists{/gist_id}","starred_url":"https://api.github.com/users/zkolter/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/zkolter/subscriptions","organizations_url":"https://api.github.com/users/zkolter/orgs","repos_url":"https://api.github.com/users/zkolter/repos","events_url":"https://api.github.com/users/zkolter/events{/privacy}","received_events_url":"https://api.github.com/users/zkolter/received_events","type":"User","site_admin":false,"name":"Zico Kolter","company":"Carnegie Mellon","blog":"","location":null,"email":"zkolter@cs.cmu.edu","hireable":null,"bio":null,"public_repos":1,"public_gists":0,"followers":5,"following":0,"created_at":"2012-10-01T17:22:55Z","updated_at":"2017-12-12T16:06:58Z","private_gists":0,"total_private_repos":0,"owned_private_repos":0,"disk_usage":0,"collaborators":0,"two_factor_authentication":false,"plan":{"name":"developer","space":976562499,"collaborators":0,"private_repos":9999}}'
```

We have already seen that we can use the `response.json()` call to convert this to a Python dictionary, but more common is to use the `json` library in the Python standard library: documentation page [here](https://docs.python.org/3/library/json.html).  To convert our GitHub response to a Python dictionary manually, we can use the `json.loads()` (load string) function like the following.


```python
import json
print(json.loads(response.content))
```

```
{'login': 'zkolter', 'id': 2465474, 'avatar_url': 'https://avatars1.githubusercontent.com/u/2465474?v=4', 'gravatar_id': '', 'url': 'https://api.github.com/users/zkolter', 'html_url': 'https://github.com/zkolter', 'followers_url': 'https://api.github.com/users/zkolter/followers', 'following_url': 'https://api.github.com/users/zkolter/following{/other_user}', 'gists_url': 'https://api.github.com/users/zkolter/gists{/gist_id}', 'starred_url': 'https://api.github.com/users/zkolter/starred{/owner}{/repo}', 'subscriptions_url': 'https://api.github.com/users/zkolter/subscriptions', 'organizations_url': 'https://api.github.com/users/zkolter/orgs', 'repos_url': 'https://api.github.com/users/zkolter/repos', 'events_url': 'https://api.github.com/users/zkolter/events{/privacy}', 'received_events_url': 'https://api.github.com/users/zkolter/received_events', 'type': 'User', 'site_admin': False, 'name': 'Zico Kolter', 'company': 'Carnegie Mellon', 'blog': '', 'location': None, 'email': 'zkolter@cs.cmu.edu', 'hireable': None, 'bio': None, 'public_repos': 1, 'public_gists': 0, 'followers': 5, 'following': 0, 'created_at': '2012-10-01T17:22:55Z', 'updated_at': '2017-12-12T16:06:58Z', 'private_gists': 0, 'total_private_repos': 0, 'owned_private_repos': 0, 'disk_usage': 0, 'collaborators': 0, 'two_factor_authentication': False, 'plan': {'name': 'developer', 'space': 976562499, 'collaborators': 0, 'private_repos': 9999}}
```

If you have the data as a file (i.e., as a file descriptor opened with the Python `open()` command), you can use the `json.load()` function instead.  To convert a Python dictionary to a JSON object, you'll use the `json.dumps()` command, such as the following.


```python
data = {"a":[1,2,3,{"b":2.1}], 'c':4}
json.dumps(data)
```

```
'{"a": [1, 2, 3, {"b": 2.1}], "c": 4}'
```

Notice that Python code, unlike JSON, can include single quotes to denote strings, but converting it to JSON will replace it with double quotes.  Finally, if you try to dump an object that includes types not representable by JSON, it will throw an error.


```python
json.dumps(response)
```

```
TypeError: Object of type 'Response' is not JSON serializable
```

### XML/HTML

Last, another format you will likely encoder are XML/HTML documents, though my assessment XML seems to be loosing out to JSON as a generic format for APIs and data files, at least for cases where JSON will suffice, mainly because JSON is substantially easier to parse.  XML files contain hierarchical content delineated by tags, like the following:

XML contains "open" tags denoted by brackets, like `<tag>`, which are then closed by a corresponding "close" tag `</tag>`.  The tags can be nested, and have optional attributes, of the form `attribute_name="attribute_value"`.  Finally, there are "open/close" tags that don't have any included content (except perhaps attributes), denoted by `<openclosetag/>`.

HTML, the standard for describing web pages, may seem syntactically similar to XML, but it is difficult to parse properly (open tags may not have closed tags, etc).  Generally speaking, HTML was developed to display content for the web, not to organize data, so a lot of invalid structure (like the aforementioned open without close) became standard simply because people frequently did this in practice, and so the data format evolved.  In the homework (the 688 version), you will actually write a simple XML parser, to understand how such parsing works, but for the most part you will use a library.  There are many such libraries for Python, but a particularly nice one is [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/).  Beautiful soup was actually written for parsing HTML (it is a common tool for scraping web pages), but it works just as well for the more-structured XML format.  You will also use BeautifulSoup as a precursor to writing your own XML parser on the homework.


```python
from bs4 import BeautifulSoup

root = BeautifulSoup("""
<tag attribute="value">
    <subtag>
        Some content for the subtag
    </subtag>
    <openclosetag attribute="value2"/>
    <subtag>
        Second one
    </subtag>
</tag>
""", "lxml-xml")

print(root, "\n")
print(root.tag.subtag, "\n")
print(root.tag.openclosetag.attrs)
```

```
<?xml version="1.0" encoding="utf-8"?>
<tag attribute="value">
<subtag>
        Some content for the subtag
    </subtag>
<openclosetag attribute="value2"/>
<subtag>
        Second one
    </subtag>
</tag> 

<subtag>
        Some content for the subtag
    </subtag> 

{'attribute': 'value2'}
```

The `BeautifulSoup()` call creates the object to parse, where the second argument specifies the parser ("lxml-xml" indicates that it is actually XML data, whereas "lxml" is the more common parser for parsing HTML files).  As illustrated above, when the hierarchical layout of the data is fairly simple, here a "tag" followed by a "subtag" (by default this will return the first such tag), or an "openclosetag", you can access the various parts of the hierarchy simply by a structure-like layout of the BeautifulSoup object.  Where this gets trickier is when there are multiple tags with the same name as the hierarchy level, as there is with the two "subtag" tags.  Above.  In this case, you can use the `find_all` function, which returns a list of all the subtags.


```python
print(root.tag.find_all("subtag"))
```

```
[<subtag>
        Some content for the subtag
    </subtag>, <subtag>
        Second one
    </subtag>]
```

The nice thing about the `find_all` function is that you can call it at previous levels in the tree, and it will recurse down the whole document.  So we could have just as easily done.


```python
print(root.find_all("subtag"))
```

```
[<subtag>
        Some content for the subtag
    </subtag>, <subtag>
        Second one
    </subtag>]
```

Let's end with a slightly more complex example, that looks through the CMU homepage to get a list of upcoming events.  This isn't perfect (the parser will keep all the whitespace from the source HTML, and so the results aren't always pretty), but it does the job.  If we examine the source of the CMU homepage, we'll see that the events are listed within `<div class="events">` tags, then within `<li>` tags.  The following illustrates how we can get the text information of each event (the `.text` attribute returns just the text content that doesn't occur within any tag).  Again, the details aren't important here, but by playing around with these calls you'll get a sense of how to extract information from web pages or from XML documents in general.


```python
response = requests.get("http://www.cmu.edu")
root = BeautifulSoup(response.content, "lxml")
for div in root.find_all("div", class_="events"):
    for li in div.find_all("li"):
        print(li.text.strip())
```

```
Jan 23
                                MLK Day of Service
Jan 25
                                Performance: Il Matrimonio Segreto
Jan 25
                                Crafting a Compelling 3MT Presentation
Jan 25
                                A Conversation with Damon Young: Race, Culture, and Politics in the Age of New Media
Jan 26
                                International Privacy Day
Feb 22 - Mar 3
                                Performance: The Drowsy Chaperone
```

## Regular expressions

The last tool we're going to consider in these notes are regular expressions.  Regular expressions are invaluable when parsing any type of unstructured data, if you're trying to quickly find or extract some text from a long string, and even if you're writing a more complex parser.  In general, regular expressions let us find and match portions of text using a simple syntax (by some definition).  

### Finding 

Let's start with the most basic example, that simply searches text for some sting.  In this case, the text we are searching is "This course will introduce the basics of data science", and the string we are searching for is "data science".  This is done with the following code:


```python
import re
text = "This course will introduce the basics of data science"
match = re.search(r"data science", text)
print(match.start())
```

```
41
```

The important element here is the `re.search(r"data science", text)` call.  It searches `text` for the string "data science" and returns a regular expression "match" object that contains information about where this match was found: for instance, we can find the character index (in `text`) where the match is found, using the `match.start()` call.  In addition to the search call, there are two or three more regular expression matching commands you may find useful: 
- `re.match()`: Match the regular expression starting at the _beginning_ of the text string
- `re.finditer()`: Find all matches in the text, returning a iterator over match objects
- `re.findall()`: Find all matches in the text, returning a list of the matched text only (not a match object)

For example, the following code would return `None`, since there is no match to "data science" at the beginning of the string:


```python
match = re.match(r"data science", text)
print(match)
```

```
None
```

Similarly, we could use `re.finditer()` to list the location of all the 'i' characters in the string:


```python
for match in re.finditer(r"i", text):
    print(match.start())
```

```
2
13
17
34
48
```

On the other hand, `re.findall()` just returns a list of the matched strings, with no additional info such as where they occurred:


```python
re.findall(r"i", text)
```

```
['i', 'i', 'i', 'i', 'i']
```

This last call may not seem particularly useful, but especially when you use more complex matching expressions, this last call can still be of some use.  Finally, you can also "compile" a regular expression and then make all the same calls on this compiled object, as in the following:


```python
regex = re.compile(r"data science")
regex.search(text)
```

```
<_sre.SRE_Match object; span=(41, 53), match='data science'>
```

However, given that Python will actually compile expressions on the fly anyways, don't expect a big speed benefit from using the above; whether to use `re.compile()` separately is more of a personal preference than anything else.

### Matching multiple potential strings

While using regular expressions to search for a string within a long piece of text may be a handy tools, the real power of regular expressions comes from the ability to match multiple potential strings with a single regular expression.  This is where the syntax of regular expressions gets nasty, but here are some of the more common rules:

- Any character (except special characters, `".$*+?{}\[]|()` ), just matches itself.  I.e., the character `a` just matches the character `a`.  This is actually what we used previously, where each character in the `r"data science"` regular expression was just looking to match that exact character.
- Putting a group of characters within brackets `[abc]` will match any of the characters `a`, `b`, or `c`. You can also use ranges within these brackets, so that `[a-z]` matches any lower case letter.
- Putting a caret within the bracket matches anything _but_ these characters, i.e., `[^abc]` matches any character _except_ `a`, `b`, or `c`.
- The special character `\d` will match any digit, i.e. `[0-9]`
- The special character `\w` will match any alphanumeric character plus the underscore; i.e., it is equivalent to `[a-zA-Z0-9_]`.
- The special character `\s` will match whitespace, any of `[ \t\n\r\f\v]` (a space, tab, and various newline characters).
- The special character `.` (the period) matches _any_ character.  In their original versions, regular expressions were often applies line-by-line to a file, so by default `.` will _not_ match the newline character.  If you want it to match newlines, you pass `re.DOTALL` to the "flags" argument of the various regular expression calls.

As an example, the following regular expression will match "data science" regardless of the capitalization, and with any type of space between the two words.


```python
print(re.search(r"[Dd]ata\s[Ss]cience", text))
```

```
<_sre.SRE_Match object; span=(41, 53), match='data science'>
```

Note that now the match objects also now include what was the particular text that matched the expression (which could be one of any number of possibilities now).  This is why calls like `re.findall` are still useful even if they only return the matched expression.

The second important feature of regular expressions it the ability to match multiple _occurences_ of a character.  The most important rules here are:
- To match `a` exactly once, use `a`.
- To match `a` zero or one times, use `a?`.
- To match `a` zero or more times, use `a*`.
- To match `a` one or more times, use `a+`.
- And finally, to match `a` exactly n times, use `a{n}`.

These rules can of course be combined with the rules to match potentially very complicated expressions.  For instance, if we want to match the text "*something* science" where *something* is any alphanumeric character, and there can be any number of spaces of any kind between *something* and the word "science", we could use the expression `r"\w+\s+science"`.


```python
print(re.match("\w+\s+science", "data science"))
print(re.match("\w+\s+science", "life science"))
print(re.match("\w+\s+science", "0123_abcd science"))
```

```
<_sre.SRE_Match object; span=(0, 12), match='data science'>
<_sre.SRE_Match object; span=(0, 12), match='life science'>
<_sre.SRE_Match object; span=(0, 17), match='0123_abcd science'>
```

These kinds of matching, even with relatively simple starting points, can lead to some interesting applications:

{% include image.html img="http://imgs.xkcd.com/comics/regex_golf.png" caption="Regex Golf, from https://xkcd.com/1313/.  Yes, this is a real thing, see e.g. https://alf.nu/RegexGolf"%}

**Aside:** One thing you may notice is the `r""` format of the regular expressions (quotes with an 'r' preceding them).  You can actually use any string as a regular expression, but the `r` expressions are quite handy for the following reason.  In a typical Python string, backslash characters denote escaped characters, so for instance `"\\"` really just encodes a single backslash.  But backslashes are _also_ used within regular expressions.  So if we want the regular expression `\\` represented as a string (that is, match a single backslash), we'd need to use the string `"\\\\"`.  This gets really tedious quickly.  So the `r""` notation just _ignores_ any handling of handling of backslashes, and thus makes inputing regular expressions much simpler.


```python
print("\\")
print(r"\\")
```

```
\
\\
```

### Grouping

Beyond the ability to just match strings, regular expressions also let you easily find specific sub-elements of the matched strings.  The basic syntax is the following: if we want to "remember" different portions of the matched expression, we just surround those portions of the regular expression in parentheses.  For example, the regular expression `r"(\w+)\s([Ss]cience)"` would store whatever element is matched to the `\w+` and `[Ss]cience` portions in the `groups()` object in the returned match.


```python
match = re.search(r"(\w+)\s([Ss]cience)", text)
print(match.groups())
```

```
('data', 'science')
```

The `.group(i)` notation also lets you easily find just individual groups, `.group(0)` being the entire text.


```python
match = re.search(r"(\w+)\s([Ss]cience)", text)
print(match.group(0))
print(match.group(1))
print(match.group(2))
```

```
data science
data
science
```

### Substitutions



Regular expression can also be used to automatically substitute one expression for another within the string.  This is done using the `re.sub()` call.  This returns a string with (all) the instances of the first regular expression replaced with the second one.  For example, to replace all the occurrences of "data science" with "data schmience", we could use the following code:


```python
print(re.sub(r"data science", r"data schmience", text))
```

```
This course will introduce the basics of data schmience
```

Where this gets really powerful is when we use groups in the first regular expression.  These groups can then be backreferenced using the `\1`, `\2`, etc notation in the second one (more generally, you can actually use these backreferencs within a single regular expression too, outside the context of substitutions, but we won't cover that here).  So if we have the regular expression `r"(\w+) ([Ss])cience"` to match "*something* science" (where science can be capitalized or not), we would replace it with the string "*something* schmience", keeping the *something* the same, and keeping the capitalization of science the same, using the code:


```python
print(re.sub(r"(\w+) ([Ss])cience", r"\1 \2chmience", text))
print(re.sub(r"(\w+) ([Ss])cience", r"\1 \2chmience", "Life Science"))
```

```
This course will introduce the basics of data schmience
Life Schmience
```

As you can imagine, this allows for very powerful processing with very short expressions.  For example, to create the notes for this Data Science Course, I actually do post-processing of Jupyter Notebooks, and use regular expressions to covert (along with some other Python code) to convert various syntax to Markdown-friendly syntax for the course web page.  For tasks like this, regular expressions are an incredibly useful tool.

### Miscellaneous items



There are a few last points that we'll discuss here, mainly because they can be of some use for the homework assignments in the course.  There are, of course, many more particulars to regular expressions, and we will later highlight a few references for further reading.

**Order of operations.** The first point comes in regard to the order of operations for regular expressions.  The `|` character in regular expressions is like an "or" clause, the regular expression should can match the regular expression to the left or to the right of the character.  For example, the regular expression `r"abc|def"` would match the string "abc" or "def".


```python
print(re.match(r"abc|def", "abc"))
print(re.match(r"abc|def", "def"))
```

```
<_sre.SRE_Match object; span=(0, 3), match='abc'>
<_sre.SRE_Match object; span=(0, 3), match='def'>
```

But what if we want to match the string "ab*(c or d)*ef"?  We can capture this in a regular expression by parentheses around the portion we want to give a higher order of operations.


```python
print(re.match(r"abc|def", "abdef"))
print(re.match(r"ab(c|d)ef", "abdef"))
```

```
None
<_sre.SRE_Match object; span=(0, 5), match='abdef'>
```

But, of course, since we also use the parentheses for specifying groups, this means that we will be creating a group for the *c or d* character here.  While it's probably not the end of the world to create a few groups you don't need, in the case that you _don't_ want to create this group, you can use the notation `r"ab(?:c|d)ef"`.  There is no real logic to this notation that I can see, it just happens to be shorthand for "use these parentheses to manage order of operations but don't create a group."  Regular expressions are full of fun things like this, and while you likely don't need to commit this to memory, it's handy to remember that there are situations like this.


```python
print(re.match(r"ab(c|d)ef", "abdef").groups())
print(re.match(r"ab(?:c|d)ef", "abdef").groups())
```

```
('d',)
()
```

**Greedy and lazy matching** The second point is something that you likely _will_ need to know, at least in the context of the homework.  This is the fact that, by default, regular expressions will always match _the longest possible string_.  So suppose we have the regular expression `r"<.*>"` (a less-than, followed by any number of any character, followed by a greater-than).  As you might imagine, this sort of expression will come up if (and for this class, when) you're writing a parser for a format like XML.  If we try to match the string "&lt;tag>hello&lt;/tag>", then the regular expression will match the _entire_ string (since the entire string is a less-than, followed by some number of characters, followed by a greater-than); it will not just match the "&lt;tag>" part of the string.  This is known as "greedy" matching.


```python
print(re.match(r"<.*>", "<tag>hello</tag>"))
```

```
<_sre.SRE_Match object; span=(0, 16), match='<tag>hello</tag>'>
```

In the case where you don't want this, but would instead want you match the _smallest_ possible string, you can use the alternative regular expression `r"<.*?>"`.  The `*?` notation indicates "lazy" matching, that you want to match any number of characters possible, but would prefer the _smallest_ possible match.  This will then match just the "&lt;tag>" string.


```python
print(re.match(r"<.*?>", "<tag>hello</tag>"))
```

```
<_sre.SRE_Match object; span=(0, 5), match='<tag>'>
```

### Final note

As one last note, while it is good to run through all of these aspects of regular expressions, they are likely something that you will not remember unless you use regular expressions quite often; the notation of regular expressions is dense and not well-suited to effortless memorization.  I had to look up a few things myself when preparing this lecture and notes.  And there are some completely crazy constructs, like the famous regular expression `r".?|(..+?)\\1+"` that [matches only prime numbers of characters](https://iluxonchik.github.io/regular-expression-check-if-number-is-prime/).

The point is, for anything remotely complex that you'd do with regular expression, you will have to look up the documentation, which for the Python libraries is available at: [https://docs.python.org/3/howto/regex.html](https://docs.python.org/3/howto/regex.html) and [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html).  These sources will be the best way to remember specifics about any particular syntax you want to use.


## References

- [requests library](http://docs.python-requests.org/en/master/)
- Fielding, Roy. [Representational State Transfer (REST)](http://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm) (thesis chapter where REST was first proposed)
- [WeFacts](http://wefacts.org) (historical weather data)
- [Pandas library](https://pandas.pydata.org)
- [json library](https://docs.python.org/3/library/json.html)
- [BeautifulSoup library](https://www.crummy.com/software/BeautifulSoup/)
- [Python Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)
- [Python regular expression library](https://docs.python.org/3/library/re.html)

