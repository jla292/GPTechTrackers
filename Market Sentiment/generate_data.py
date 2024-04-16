# Package used to execute HTTP POST request to the API
import json
import urllib.request
import pandas as pd
from dateutil import rrule
from datetime import date, datetime, timedelta
import time

# curl -XPOST -H "Content-type: application/json" -d '{
#     "type": "filterArticles",
#     "queryString": "title:WTI OR description:WTI",
#     "from": 0,
#     "size": 50
# }' 'https://api.newsfilter.io/public/actions?token=8a9ea1e0669e43cc8e3bf11bfde7883b83d6242d3ae34deaa6805dcb3242537f'

# API endpoint
API_KEY = '8a9ea1e0669e43cc8e3bf11bfde7883b83d6242d3ae34deaa6805dcb3242537f'
API_ENDPOINT = "https://api.newsfilter.io/public/actions?token={}".format(API_KEY)

def articles_perweek (queryString, titles, descriptions):
  # Define the filter parameters
  # queryString = "title:WTI OR title:oil OR description:WTI OR symbols:WTI AND publishedAt:[2021-01-01 TO 2023-11-11]"
  # "symbols:NFLX AND publishedAt:[2020-02-01 TO 2020-05-20]"

  payload = {
  "type": "filterArticles",
  "queryString": queryString,
  "from": 0,
  "size": 8
  # max size for one api call is 50
  }

  # Format your payload to JSON bytes
  jsondata = json.dumps(payload)
  jsondataasbytes = jsondata.encode('utf-8')

  # Instantiate the request
  req = urllib.request.Request(API_ENDPOINT)

  # Set the correct HTTP header: Content-Type = application/json
  req.add_header('Content-Type', 'application/json; charset=utf-8')
  # Set the correct length of your request
  req.add_header('Content-Length', len(jsondataasbytes))

  # Send the request to the API
  response = urllib.request.urlopen(req, jsondataasbytes)

  # Read the response
  res_body = response.read()
  # Transform the response into JSON
  articles = json.loads(res_body.decode("utf-8"))
  # Get titles only
  a = articles['articles']
  i = 0
  while i < len(a):
    article = a[i]
    title = article['title']
    description = article['description']
    titles.append(title)
    descriptions.append(description)
    i = i + 1
  
  print(articles)

  return titles, descriptions

# Run the function
titles_train = []
descriptions_train = []
titles_test = []
descriptions_test = []

now = datetime.now()
start = date(2023, 1, 1)
train_end = now - timedelta(days=112)

# train
for dt in rrule.rrule(rrule.WEEKLY, dtstart=start, until=train_end):
  dt_plusweek = dt + timedelta(days=7)
  queryString = "title:ChatGPT AND description:ChatGPT AND publishedAt:[" + dt.strftime('%Y-%m-%d') + " TO " + dt_plusweek.strftime('%Y-%m-%d') + "]"
  titles_train, descriptions_train = articles_perweek(queryString, titles_train, descriptions_train)
  time.sleep(1)
  queryString1 = "title:OpenAI AND description:OpenAI AND publishedAt:[" + dt.strftime('%Y-%m-%d') + " TO " + dt_plusweek.strftime('%Y-%m-%d') + "]"
  titles_train, descriptions_train = articles_perweek(queryString1, titles_train, descriptions_train)
  time.sleep(1)

# test
for dt in rrule.rrule(rrule.WEEKLY, dtstart=train_end, until=now):
  dt_plusweek = dt + timedelta(days=7)
  queryString = "title:ChatGPT AND description:ChatGPT AND publishedAt:[" + dt.strftime('%Y-%m-%d') + " TO " + dt_plusweek.strftime('%Y-%m-%d') + "]"
  titles_test, descriptions_test = articles_perweek(queryString, titles_test, descriptions_test)
  time.sleep(1)
  queryString1 = "title:OpenAI AND description:OpenAI AND publishedAt:[" + dt.strftime('%Y-%m-%d') + " TO " + dt_plusweek.strftime('%Y-%m-%d') + "]"
  titles_test, descriptions_test = articles_perweek(queryString1, titles_test, descriptions_test)
  time.sleep(1)
#print(titles)

# Export train to csv for annotations
df1 = pd.DataFrame(descriptions_train, titles_train)
df1.to_csv('stock_sentiment_train.csv')
# article_titles_descriptions

# Export test to csv for annotations
df2 = pd.DataFrame(descriptions_test, titles_test)
df2.to_csv('stock_sentiment_test.csv')