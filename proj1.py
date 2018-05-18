
# coding: utf-8

# Before you turn this assignment in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All). Lastly, hit **Validate**.
# 
# If you worked locally, and then uploaded your work to the hub, make sure to follow these steps:
# - open your uploaded notebook **on the hub**
# - hit the validate button right above this cell, from inside the notebook
# 
# These  steps should solve any issue related to submitting the notebook on the hub.
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Brandon Griffin"
COLLABORATORS = ""


# ---

# # Project 1: Trump, Twitter, and Text
# 
# Welcome to the first project of Data 100! In this project, we will work with the Twitter API in order to analyze Donald Trump's tweets.
# 
# **The project is due 11:59pm Tuesday, Feb 27, California Time.**
# 
# *Fair warning:* This project involves significantly more challenging pandas operations than the previous homeworks. We strongly suggest you start early.
# 
# **Fun:**
# 
# We intended this project to be fun! You will analyze actual data from the Twitter API. You will also draw conclusions about the current (and often controversial) US President's tweet behavior. If you find yourself getting frustrated or stuck on one problem for too long, we suggest coming into office hours and working with friends in the class.
# 
# *If you find yourself getting frustrated with the data we suggest you vote and/or encourage others to vote.*
# 
# With that in mind, let's get started!

# In[2]:


# Run this cell to set up your notebook
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()
sns.set_context("talk")
import re


# # Downloading Recent Tweets
# 
# ---
# 
# Since we'll be looking at Twitter data, we need to download the data from Twitter!
# 
# Twitter provides an API for downloading tweet data in large batches.  The `tweepy` package makes it fairly easy to use.

# In[3]:


## Make sure you are in your data100 conda environment if you are working locally.
# The following should run:
import tweepy


# There are instructions on using `tweepy` [here](http://tweepy.readthedocs.io/en/v3.5.0/getting_started.html), but we will give you example code.
# 
# Twitter requires you to have authentication keys to access their API.  To get your keys, you'll have to sign up as a Twitter developer.  The next question will walk you through this process.

# ## Question 1
# 
# Follow the instructions below to get your Twitter API keys.  **Read the instructions completely before starting.**
# 
# 1. [Create a Twitter account](https://twitter.com).  You can use an existing account if you have one; if you prefer to not do this assignment under your regular account, feel free to create a throw-away account.
# 2. Under account settings, add your phone number to the account.
# 3. [Create a Twitter developer account](https://dev.twitter.com/resources/signup).  Attach it to your Twitter account.
# 4. Once you're logged into your developer account, [create an application for this assignment](https://apps.twitter.com/app/new).  You can call it whatever you want, and you can write any URL when it asks for a web site.  You don't need to provide a callback URL.
# 5. On the page for that application, find your Consumer Key and Consumer Secret.
# 6. On the same page, create an Access Token.  Record the resulting Access Token and Access Token Secret.
# 7. Edit the file [keys.json](keys.json) and replace the placeholders with your keys.  

# 
# ## WARNING (Please Read) !!!!
# 
# 
# ### Protect your Twitter Keys
# <span style="color:red">
# If someone has your authentication keys, they can access your Twitter account and post as you!  So don't give them to anyone, and **don't write them down in this notebook**. 
# </span>
# The usual way to store sensitive information like this is to put it in a separate file and read it programmatically.  That way, you can share the rest of your code without sharing your keys.  That's why we're asking you to put your keys in `keys.json` for this assignment.
# 
# 
# ### Avoid making too many API calls.
# 
# <span style="color:red">
# Twitter limits developers to a certain rate of requests for data.  If you make too many requests in a short period of time, you'll have to wait awhile (around 15 minutes) before you can make more.  </span> 
# So carefully follow the code examples you see and don't rerun cells without thinking.  Instead, always save the data you've collected to a file.  We've provided templates to help you do that.
# 
# 
# ### Be careful about which functions you call!
# 
# <span style="color:red">
# This API can retweet tweets, follow and unfollow people, and modify your twitter settings.  Be careful which functions you invoke! </span> One of your instructors accidentally re-tweeted some tweets because that instructor typed `retweet` instead of `retweet_count`. 
# </span>
# 

# In[4]:


import json
key_file = 'keys.json'
# Loading your keys from keys.json (which you should have filled
# in in question 1):
with open(key_file) as f:
    keys = json.load(f)
# if you print or view the contents of keys be sure to delete the cell!


# This cell tests the Twitter authentication. It should run without errors or warnings and display your Twitter username.

# In[5]:


import tweepy
from tweepy import TweepError
import logging

try:
    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
    auth.set_access_token(keys["access_token"], keys["access_token_secret"])
    api = tweepy.API(auth)
    print("Your username is:", api.auth.get_username())
except TweepError as e:
    logging.warning("There was a Tweepy error. Double check your API keys and try again.")
    logging.warning(e)


# ## Question 2
# 
# In the example below, we have loaded some tweets by @BerkeleyData.  Run it and read the code.

# In[6]:


from pathlib import Path
import json

ds_tweets_save_path = "BerkeleyData_recent_tweets.json"
# Guarding against attempts to download the data multiple
# times:
if not Path(ds_tweets_save_path).is_file():
    # Getting as many recent tweets by @BerkeleyData as Twitter will let us have.
    # We use tweet_mode='extended' so that Twitter gives us full 280 character tweets.
    # This was a change introduced in September 2017.
    
    # The tweepy Cursor API actually returns "sophisticated" Status objects but we 
    # will use the basic Python dictionaries stored in the _json field. 
    example_tweets = [t._json for t in tweepy.Cursor(api.user_timeline, id="BerkeleyData", 
                                             tweet_mode='extended').items()]
    
    # Saving the tweets to a json file on disk for future analysis
    with open(ds_tweets_save_path, "w") as f:        
        json.dump(example_tweets, f)

# Re-loading the json file:
with open(ds_tweets_save_path, "r") as f:
    example_tweets = json.load(f)


# Assuming everything ran correctly you should be able to look at the first tweet by running the cell below.
# 
# <span style="color:red">
# **Warning** Do not attempt to view all the tweets in a notebook.  It will likely freeze your browser.  The following would be a **bad idea**:
# ```python
#     pprint(example_tweets)
# ```
# 
# </span> 

# In[7]:


# Looking at one tweet object, which has type Status: 
from pprint import pprint # ...to get a more easily-readable view.
pprint(example_tweets[0])


# ## Question 2a
# 
# ### What you need to do. 
# 
# Re-factor the above code fragment into reusable snippets below.  You should not need to make major modifications; this is mostly an exercise in understanding the above code block. 

# In[8]:


def load_keys(path):
    """Loads your Twitter authentication keys from a file on disk.
    
    Args:
        path (str): The path to your key file.  The file should
          be in JSON format and look like this (but filled in):
            {
                "consumer_key": "<your Consumer Key here>",
                "consumer_secret":  "<your Consumer Secret here>",
                "access_token": "<your Access Token here>",
                "access_token_secret": "<your Access Token Secret here>"
            }
    
    Returns:
        dict: A dictionary mapping key names (like "consumer_key") to
          key values."""
    
    # YOUR CODE HERE
    import json
    key_file = 'keys.json'
    # Loading your keys from keys.json (which you should have filled
    # in in question 1):
    with open(key_file) as f:
        keys = json.load(f)
    # if you print or view the contents of keys be sure to delete the cell!
    return keys


# In[9]:


def download_recent_tweets_by_user(user_account_name, keys):
    """Downloads tweets by one Twitter user.

    Args:
        user_account_name (str): The name of the Twitter account
          whose tweets will be downloaded.
        keys (dict): A Python dictionary with Twitter authentication
          keys (strings), like this (but filled in):
            {
                "consumer_key": "<your Consumer Key here>",
                "consumer_secret":  "<your Consumer Secret here>",
                "access_token": "<your Access Token here>",
                "access_token_secret": "<your Access Token Secret here>"
            }

    Returns:
        list: A list of Dictonary objects, each representing one tweet."""
    import tweepy
    # We use tweet_mode='extended' so that Twitter gives us full 280 character tweets.
    # The tweepy Cursor API actually returns "sophisticated" Status objects but we 
    # will use the basic Python dictionaries stored in the _json field. 
    tweets = [t._json for t in tweepy.Cursor(api.user_timeline, id=user_account_name, 
                                             tweet_mode='extended').items()]

    return tweets


# In[10]:


def save_tweets(tweets, path):
    """Saves a list of tweets to a file in the local filesystem.
    
    This function makes no guarantee about the format of the saved
    tweets, **except** that calling load_tweets(path) after
    save_tweets(tweets, path) will produce the same list of tweets
    and that only the file at the given path is used to store the
    tweets.  (That means you can implement this function however
    you want, as long as saving and loading works!)

    Args:
        tweets (list): A list of tweet objects (of type Dictionary) to
          be saved.
        path (str): The place where the tweets will be saved.

    Returns:
        None"""
    # YOUR CODE HERE
    from pathlib import Path
    import json

    save_path = path + ".json"
    # Guarding against attempts to download the data multiple
    # times:
    if not Path(save_path).is_file():
        # Saving the tweets to a json file on disk for future analysis
        with open(ds_tweets_save_path, "w") as f:        
            json.dump(tweets, f)
        print('Tweets saved!')


# In[11]:


def load_tweets(path):
    """Loads tweets that have previously been saved.
    
    Calling load_tweets(path) after save_tweets(tweets, path)
    will produce the same list of tweets.
    
    Args:
        path (str): The place where the tweets were be saved.

    Returns:
        list: A list of Dictionary objects, each representing one tweet."""
    
    if Path(path).is_file():
        #Loading the json file:
        with open(path, "r") as f:
            tweets = json.load(f)
        print('Tweets Loaded!')
        return tweets
    return None


# In[12]:


def get_tweets_with_cache(user_account_name, keys_path):
    """Get recent tweets from one user, loading from a disk cache if available.
    
    The first time you call this function, it will download tweets by
    a user.  Subsequent calls will not re-download the tweets; instead
    they'll load the tweets from a save file in your local filesystem.
    All this is done using the functions you defined in the previous cell.
    This has benefits and drawbacks that often appear when you cache data:
    
    +: Using this function will prevent extraneous usage of the Twitter API.
    +: You will get your data much faster after the first time it's called.
    -: If you really want to re-download the tweets (say, to get newer ones,
       or because you screwed up something in the previous cell and your
       tweets aren't what you wanted), you'll have to find the save file
       (which will look like <something>_recent_tweets.pkl) and delete it.
    
    Args:
        user_account_name (str): The Twitter handle of a user, without the @.
        keys_path (str): The path to a JSON keys file in your filesystem.
    """
    
    # YOUR CODE HERE
    twits = load_tweets(user_account_name)
    if twits is None:
        keys = load_keys(keys_path)
        twits = download_recent_tweets_by_user(user_account_name, keys)
        save_tweets(twits, user_account_name)
        return twits    
    return twits


# If everything was implemented correctly you should be able to obtain roughly the last 3000 tweets by the `realdonaldtrump`.  (This may take a few minutes)

# In[13]:


# When you are done, run this cell to load @realdonaldtrump's tweets.
# Note the function get_tweets_with_cache.  You may find it useful
# later.
trump_tweets = get_tweets_with_cache("realdonaldtrump", key_file)
print("Number of tweets downloaded:", len(trump_tweets))


# In[14]:


assert 2000 <= len(trump_tweets) <= 4000


# ### Question 2b
# 
# We are limited to how many tweets we can download.  In what month is the oldest tweet from Trump?

# In[15]:


import time
# Enter the number of the month of the oldest tweet (e.g. 1 for January)
lasttime = time.strptime(trump_tweets[-1]['created_at'], '%a %b %d %X %z %Y')
oldest_month = lasttime.tm_mon
print(oldest_month,lasttime.tm_year)


# 
# ## Question 3
# 
# 
# **IMPORTANT! PLEASE READ**
# 
# Unfortunately, Twitter prevent us from going further back in time using the public APIs.  Fortunately, we have a snapshot of earlier tweets that we can combine with our new data.  
# 
# We will again use the `fetch_and_cache` utility to download the dataset. 

# In[16]:


# Download the dataset
from utils import fetch_and_cache
data_url = 'http://www.ds100.org/sp18/assets/datasets/old_trump_tweets.json.zip'
file_name = 'old_trump_tweets.json.zip'

dest_path = fetch_and_cache(data_url=data_url, file=file_name)
print(f'Located at {dest_path}')


# Finally, we we will load the tweets directly from the compressed file without decompressing it first.

# In[17]:


my_zip = zipfile.ZipFile(dest_path, 'r')
with my_zip.open("old_trump_tweets.json", "r") as f:
    old_trump_tweets = json.load(f)


# This data is formatted identically to the recent tweets we just downloaded:

# In[18]:


pprint(old_trump_tweets[0])


# As a dictionary we can also list the keys:

# In[19]:


old_trump_tweets[0].keys()


# ### Question 3a
# 
# Merge the `old_trump_tweets` and the `trump_tweets` we downloaded from twitter into one giant list of tweets. 
# 
# **Important:** There may be some overlap so be sure to eliminate duplicate tweets.  
# **Hint:** the `id` of a tweet is always unique.

# In[20]:


print(len(trump_tweets))
all_tweets = trump_tweets.copy()

tweetids = [x['id'] for x in trump_tweets[:]]
for i in old_trump_tweets:
    if i['id'] not in tweetids:
        all_tweets.append(i)
print(all_tweets[0])


# In[21]:


assert len(all_tweets) > len(trump_tweets)
assert len(all_tweets) > len(old_trump_tweets)


# ### Question 3b
# 
# Construct a DataFrame called `trump` containing all the tweets stored in `all_tweets`. The index of the dataframe should be the ID of each tweet (looks something like `907698529606541312`). It should have these columns:
# 
# - `time`: The time the tweet was created encoded as a datetime object. (Use `pd.to_datetime` to encode the timestamp.)
# - `source`: The source device of the tweet.
# - `text`: The text of the tweet.
# - `retweet_count`: The retweet count of the tweet. 
# 
# Finally, **the resulting dataframe should be sorted by the index.**
# 
# **Warning:** *Some tweets will store the text in the `text` field and other will use the `full_text` field.*

# In[22]:


allids = [x['id'] for x in all_tweets[:]]
texts = []
times = []
sources = []
rt = []
for y in all_tweets:
    sources.append(y['source'])
    rt.append(y['retweet_count'])
    times.append(pd.to_datetime(y['created_at']))
    if 'full_text' in y.keys():
        texts.append(y['full_text'])
    else:
        texts.append(y['text'])
        
trump = pd.DataFrame({
    'time': times,
    'source': sources,
    'text': texts,
    'retweet_count': rt
})
trump.index = allids
#pd.DataFrame(all_tweets, list(allids), ['time', 'source', 'text', 'retweet_count'])


print(trump.head)


# In[23]:


assert isinstance(trump, pd.DataFrame)
assert trump.shape[0] < 8000
assert trump.shape[1] >= 4
assert 831846101179314177 in trump.index
assert 753063644578144260 in trump.index
assert all(col in trump.columns for col in ['time', 'source', 'text', 'retweet_count'])
# If you fail these tests, you probably tried to use __dict__ or _json to read in the tweets
assert np.sometrue([('Twitter for iPhone' in s) for s in trump['source'].unique()])
assert trump['time'].dtype == np.dtype('<M8[ns]')
assert trump['text'].dtype == np.dtype('O')
assert trump['retweet_count'].dtype == np.dtype('int64')


# ## Question 4: Tweet Source Analysis
# 
# In the following questions, we are going to find out the charateristics of Trump tweets and the devices used for the tweets.
# 
# First let's examine the source field:

# In[24]:


trump['source'].unique()


# ## Question 4a
# 
# Remove the HTML tags from the source field. 
# 
# **Hint:** Use `trump['source'].str.replace` and your favorite regular expression.

# In[25]:


## Uncomment and complete
import re
trump['source'] = trump['source'].str.replace(re.compile('<.*?>'), '')
#trump = trump[trump['source'][:7] == 'Twitter']
trump['source'].unique()


# In[26]:


from datetime import datetime
ELEC_DATE = datetime(2016, 11, 8)
INAUG_DATE = datetime(2017, 1, 20)
assert set(trump[(trump['time'] > ELEC_DATE) & (trump['time'] < INAUG_DATE) ]['source'].unique()) == set(['Twitter Ads',
 'Twitter Web Client',
 'Twitter for Android',
 'Twitter for iPhone'])


# We can see in the following plot that there are two device types that are more commonly used

# In[27]:


trump['source'].value_counts().plot(kind="bar")
plt.ylabel("Number of Tweets")


# ## Question 4b
# 
# 
# Is there a difference between his Tweet behavior across these devices? We will attempt to answer this question in our subsequent analysis.
# 
# First, we'll take a look at whether Trump's tweets from an Android come at different times than his tweets from an iPhone. Note that Twitter gives us his tweets in the [UTC timezone](https://www.wikiwand.com/en/List_of_UTC_time_offsets) (notice the `+0000` in the first few tweets)

# In[28]:


for t in trump_tweets[0:3]:
    print(t['created_at'])


# We'll convert the tweet times to US Eastern Time, the timezone of New York and Washington D.C., since those are the places we would expect the most tweet activity from Trump.

# In[29]:


trump['est_time'] = (
    trump['time'].dt.tz_localize("UTC") # Set initial timezone to UTC
                 .dt.tz_convert("EST") # Convert to Eastern Time
)
trump.head()


# **What you need to do:**
# 
# Add a column called `hour` to the `trump` table which contains the hour of the day as floating point number computed by:
# 
# $$
# \text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{60^2}
# $$
# 

# In[30]:


hours = []
for t in trump['est_time']:
    hours.append(float(float(t.hour)+float(float(t.minute)/60.0)+(float(t.second)/float(60*60))))

trump['hour'] = hours
print(trump.loc[690171032150237184]['hour'])




# In[31]:


assert np.isclose(trump.loc[690171032150237184]['hour'], 8.93639)


# ## Question 4c
# 
# Use this data along with the seaborn `distplot` function to examine the distribution over hours of the day in `easter time` that trump tweets on each device for the 2 most commonly used devices.  Your plot should look similar to the following. 
# 
# <img src="images/device_hour2.png" width="600px" />

# In[32]:


### make your plot here
# YOUR CODE HERE
trump.sort_values('hour', inplace=True)
#print(trump[trump['source']== 'Twitter for iPhone'].groupby(['hour'])['source'].value_counts())
iphone = pd.DataFrame(trump[trump['source'] == 'Twitter for iPhone'])

android = pd.DataFrame(trump[trump['source'] == 'Twitter for Android'])
#mostpop.index = trump['hour'].value_counts()

#for col in ['Twitter for iPhone', 'Twitter for Android']:
sns.distplot(iphone['hour'], label='Twitter for iPhone')
sns.distplot(android['hour'], label='Twitter for Android')
plt.legend()
plt.ylabel('fraction')

#iphone = mostpop.loc[mostpop['source']=='Twitter for iPhone']
#sns.distplot(android)


# 
# ## Question 4d
# 
# Are there any striking differences between these curves.  If someone told you that Trump tends to tweet early in the morning and then later in the evening, which device might you conclude is most likely his?

# They both seem to peak early in the day and then again later in the day, however the peaks are much more apparent in the Android plot due to a steeper dropoff midday. This makes me believe Trump's device is the Android, which exhibits more extreme tweeting volume mins and maxes.

# ## Question 5
# 
# Let's now look at which device he has used over the entire time period of this dataset.
# 
# To examine the distribution of dates we will convert the date to a fractional year that can be plotted as a distribution.
# 
# (Code borrowed from https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years)

# In[38]:


import datetime
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


trump['year'] = trump['time'].apply(year_fraction)


# ### Question 5a
# 
# Use the `sns.distplot` to overlay the distributions of the 2 most frequently used web technologies over the years.  Your final plot should look like:
# 
# <img src="images/source_years.png" width="600px" />
# 

# In[39]:


trump.sort_values('year', inplace=True)
iphone = pd.DataFrame(trump[trump['source'] == 'Twitter for iPhone'])
android = pd.DataFrame(trump[trump['source'] == 'Twitter for Android'])
sns.distplot(iphone['year'], label='Twitter for iPhone')
sns.distplot(android['year'], label='Twitter for Android')
plt.legend()
plt.xlabel('year')


# ### Question 5b
# According to the plot, Trump's tweets come from many different sources. It turns out that many of his tweets were not from Trump himself but from his staff. [Take a look at this Verge article.](https://www.theverge.com/2017/3/29/15103504/donald-trump-iphone-using-switched-android)
# 
# Does the data support the information in the article? What else do you find out about changes in Trump's tweets sources from the plot?

# The data does support the info in the article, namely that Trump switched from an iPhone to an android somewhere shortly after being inaugurated. It seems when he switched devices there was a slight drop in tweeting overall, and also there were still tweets coming from an iPhone while he supposedly only had an Android, meaning he either was using both or his staff were tweeting from an iPhone before the switch.

# ## Question 6: Sentiment Analysis
# 
# It turns out that we can use the words in Trump's tweets to calculate a measure of the sentiment of the tweet. For example, the sentence "I love America!" has positive sentiment, whereas the sentence "I hate taxes!" has a negative sentiment. In addition, some words have stronger positive / negative sentiment than others: "I love America." is more positive than "I like America."
# 
# We will use the [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment) lexicon to analyze the sentiment of Trump's tweets. VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media which is great for our usage.
# 
# The VADER lexicon gives the sentiment of individual words. Run the following cell to show the first few rows of the lexicon:

# In[33]:


print(''.join(open("vader_lexicon.txt").readlines()[:10]))


# ## Question 6a
# 
# As you can see, the lexicon contains emojis too! The first column of the lexicon is the *token*, or the word itself. The second column is the *polarity* of the word, or how positive / negative it is.
# 
# (How did they decide the polarities of these words? What are the other two columns in the lexicon? See the link above.)
# 
#  Read in the lexicon into a DataFrame called `sent`. The index of the DF should be the tokens in the lexicon. `sent` should have one column: `polarity`: The polarity of each token.

# In[34]:


vader = pd.read_table("vader_lexicon.txt", header=None,
                      names=['token', 'polarity', 'weight', 'array'], index_col=0)

sent = pd.DataFrame(vader['polarity'])


# In[35]:


assert isinstance(sent, pd.DataFrame)
assert sent.shape == (7517, 1)
assert list(sent.index[5000:5005]) == ['paranoids', 'pardon', 'pardoned', 'pardoning', 'pardons']
assert np.allclose(sent['polarity'].head(), [-1.5, -0.4, -1.5, -0.4, -0.7])


# ## Question 6b
# 
# Now, let's use this lexicon to calculate the overall sentiment for each of Trump's tweets. Here's the basic idea:
# 
# 1. For each tweet, find the sentiment of each word.
# 2. Calculate the sentiment of each tweet by taking the sum of the sentiments of its words.
# 
# First, let's lowercase the text in the tweets since the lexicon is also lowercase. Set the `text` column of the `trump` DF to be the lowercased text of each tweet.

# In[36]:


trump['text']=trump['text'].str.lower()


# In[37]:


assert trump['text'].loc[884740553040175104] == 'working hard to get the olympics for the united states (l.a.). stay tuned!'


# ## Question 6c
# 
# Now, let's get rid of punctuation since it'll cause us to fail to match words. Create a new column called `no_punc` in the `trump` DF to be the lowercased text of each tweet with all punctuation replaced by a single space. We consider punctuation characters to be any character that isn't a Unicode word character or a whitespace character. You may want to consult the Python documentation on regexes for this problem.
# 
# (Why don't we simply remove punctuation instead of replacing with a space? See if you can figure this out by looking at the tweet data.)

# In[38]:


import string
# Save your regex in punct_re
punct_re = r'[^\s\w\d]'
trump['no_punc'] = trump['text'].str.replace(re.compile(punct_re), ' ')



# In[39]:


assert isinstance(punct_re, str)
assert re.search(punct_re, 'this') is None
assert re.search(punct_re, 'this is ok') is None
assert re.search(punct_re, 'this is\nok') is None
assert re.search(punct_re, 'this is not ok.') is not None
assert re.search(punct_re, 'this#is#ok') is not None
assert re.search(punct_re, 'this^is ok') is not None
assert trump['no_punc'].loc[800329364986626048] == 'i watched parts of  nbcsnl saturday night live last night  it is a totally one sided  biased show   nothing funny at all  equal time for us '
assert trump['no_punc'].loc[894620077634592769] == 'on  purpleheartday i thank all the brave men and women who have sacrificed in battle for this great nation   usa   https   t co qmfdlslp6p'
# If you fail these tests, you accidentally changed the text column
assert trump['text'].loc[884740553040175104] == 'working hard to get the olympics for the united states (l.a.). stay tuned!'


# ## Question 6d:
# 
# 
# Now, let's convert the tweets into what's called a [*tidy format*](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) to make the sentiments easier to calculate. Use the `no_punc` column of `trump` to create a table called `tidy_format`. The index of the table should be the IDs of the tweets, repeated once for every word in the tweet. It has two columns:
# 
# 1. `num`: The location of the word in the tweet. For example, if the tweet was "i love america", then the location of the word "i" is 0, "love" is 1, and "america" is 2.
# 2. `word`: The individual words of each tweet.
# 
# The first few rows of our `tidy_format` table look like:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>num</th>
#       <th>word</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>894661651760377856</th>
#       <td>0</td>
#       <td>i</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>1</td>
#       <td>think</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>2</td>
#       <td>senator</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>3</td>
#       <td>blumenthal</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>4</td>
#       <td>should</td>
#     </tr>
#   </tbody>
# </table>
# 
# **Note that you'll get different results depending on when you pulled in the tweets.** However, you can double check that your tweet with ID `894661651760377856` has the same rows as ours. Our tests don't check whether your table looks exactly like ours.
# 
# This will require some rather advanced Pandas hacking, but our solution uses a chain of 5 methods on the `trump` DF.
# 
# * **Hint 1:** Try looking at the `expand` argument to pandas' `str.split`.
# 
# * **Hint 2:** Try looking at the `stack()` method.
# 

# In[40]:



tidy_format = pd.DataFrame(trump['no_punc'].str.split(' ', expand=True).stack())
tidy_format.reset_index(inplace=True)
tidy_format.columns = ['id','num', 'word']
tidy_format.index = tidy_format['id']
tidy_format = tidy_format.drop(columns=['id'])
tidy_format['word'].replace('', np.nan, inplace=True)
tidy_format.dropna(subset=['word'], inplace=True)


# In[41]:


assert tidy_format.loc[894661651760377856].shape == (27, 2)
assert ' '.join(list(tidy_format.loc[894661651760377856]['word'])) == 'i think senator blumenthal should take a nice long vacation in vietnam where he lied about his service so he can at least say he was there'


# ## Question 6e:
# 
# Now that we have this table in the tidy format, it becomes much easier to find the sentiment of each tweet: we can join the table with the lexicon table. 
# 
# Add a `polarity` column to the `trump` table.  The `polarity` column should contain the sum of the sentiment polarity of each word in the text of the tweet.
# 
# **Hint** you will need to merge the `tidy_format` and `sent` tables and group the final answer.
# 

# In[42]:


#tidy_format = tidy_format.groupby(tidy_format.index.get_level_values(0))
#print(tidy_format.head)
fixey = tidy_format.merge(sent, how='left', right_index=True, left_on='word').groupby(['id']).sum()
trump['polarity'] = fixey['polarity']


# In[43]:


assert np.allclose(trump.loc[744701872456536064, 'polarity'], 8.4)
assert np.allclose(trump.loc[745304731346702336, 'polarity'], 2.5)
assert np.allclose(trump.loc[744519497764184064, 'polarity'], 1.7)
assert np.allclose(trump.loc[894661651760377856, 'polarity'], 0.2)
assert np.allclose(trump.loc[894620077634592769, 'polarity'], 5.4)
# If you fail this test, you dropped tweets with 0 polarity
assert np.allclose(trump.loc[744355251365511169, 'polarity'], 0.0)


# Now we have a measure of the sentiment of each of his tweets! Note that this calculation is rather basic; you can read over the VADER readme to understand a more robust sentiment analysis.
# 
# Now, run the cells below to see the most positive and most negative tweets from Trump in your dataset:

# In[44]:


print('Most negative tweets:')
for t in trump.sort_values('polarity').head()['text']:
    print('\n  ', t)


# In[52]:


print('Most positive tweets:')
for t in trump.sort_values('polarity', ascending=False).head()['text']:
    print('\n  ', t)


# ## Question 6g
# 
# Plot the distribution of tweet sentiments broken down by whether the text of the tweet contains `nyt` or `fox`.  Then in the box below comment on what we observe?

# In[45]:


nyt = pd.DataFrame(trump[trump['text'].str.contains('nyt')])
fox = pd.DataFrame(trump[trump['text'].str.contains('fox')])
sns.distplot(nyt['polarity'], label='nyt')
sns.distplot(fox['polarity'], label='fox')
plt.legend()
plt.xlabel('polarity')


# #### Comment on what you observe:

# Trump loves fox news much more than nyt, as it is skewed right in polarity (mostly positive) and has a much higher frequency of occurences.

# ## Question 7: Engagement
# 
# ## Question 7a
# 
# Which of Trump's tweets had the most retweets? Were there certain words that often led to more retweets?
# 
# We can find this out by using our `tidy_format` DataFrame. For each word in the `tidy_format` DF, find out the number of retweets that its tweet got. Filter out words that didn't appear in at least 25 tweets, find out the median number of retweets each word got, and save the top 20 most retweeted words into a DataFrame called `top_20`. Your `top_20` table should have this format:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>retweet_count</th>
#     </tr>
#     <tr>
#       <th>word</th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>fake</th>
#       <td>22963.0</td>
#     </tr>
#     <tr>
#       <th>news</th>
#       <td>20463.0</td>
#     </tr>
#     <tr>
#       <th>ds100</th>
#       <td>20432.0</td>
#     </tr>
#     <tr>
#       <th>great</th>
#       <td>20159.0</td>
#     </tr>
#     <tr>
#       <th>class</th>
#       <td>20121.0</td>
#     </tr>
#   </tbody>
# </table>

# In[46]:


#print(tidy_format.head)

tidyup = trump.merge(tidy_format, left_index=True, right_index=True).reset_index()
tidy2 = tidyup.groupby('word').count()
words = tidy2[tidy2['retweet_count'] >=25].reset_index()


#trump[trump['text'].str.contains(tidy_format['word'])]['retweet_count'].median
#print(tidy_format['word'].value_counts())


top_20 = words.merge(tidyup, on='word').groupby('word').median().sort_values('retweet_count_y', ascending=False)[['retweet_count_y']].head(20)
top_20.columns = ['retweet_count']


# In[47]:


#### NOTE This Test is kind of iffy (very variable) - needs review before publishing

# Although it can't be guaranteed, it's very likely that the top 7 words will still be
# in the top 20 words in the next month.
assert 'daca'     in top_20.index
assert 'nfl'     in top_20.index
assert 'anthem' in top_20.index
assert 'fbi'    in top_20.index
assert 'russia'    in top_20.index


# Here's a bar chart of your results:

# In[48]:


top_20['retweet_count'].sort_values().plot.barh(figsize=(10, 8));


# ## Question 7b
# 
# The phrase "fake news" is apparently really popular! We can conclude that Trump's tweets containing "fake" and/or "news" result in the most retweets relative to words his other tweets. Or can we?
# 
# Consider each of the statements about possible confounding factors below. State whether each statement is true or false and explain. If the statement is true, state whether the confounding factor could have made "fake" and/or "news" higher on our list than they should be.
# 
# 1. We didn't restrict our word list to nouns, so we have unhelpful words like "let" and "any" in our result.
# 1. We didn't remove hashtags in our text, so we have duplicate words (eg. #great and great).
# 1. We didn't account for the fact that Trump's follower count has increased over time.

# 1. False, while it is true we did not restrict our word list to nouns, we restricted our search by using median retweet values, to find only the words that show up in tweets with a lot of retweets and not those that show up very often.
# 
# 2. False, I believe hashtags were removed in the punctuation step, at least upon my searching i found no words with remaining hashtags.
# 
# 3. True, Trump likely gained a lot of followers after becoming president, and has only increased in following since then, meaning his newer tweets on average all receive more retweets.

# ## Question 8
# 
# Using the `trump` tweets construct an interesting plot describing a property of the data and discuss what you found below.
# 
# **Ideas:**
# 
# 1. How has the sentiment changed with length of the tweets?
# 1. Does sentiment affect retweet count?
# 1. Are retweets more negative than regular tweets?
# 1. Are there any spikes in the number of retweets and do the correspond to world events? 
# 1. *Bonus:* How many Russian twitter bots follow Trump? 
# 
# You can look at other data sources and even tweets.
# 

# ### Plot:

# In[57]:


pos = pd.DataFrame(trump[trump['polarity'] > 0])
neg = pd.DataFrame(trump[trump['polarity'] < 0])
sns.distplot(pos['hour'], label='positive tweets')
sns.distplot(neg['hour'], label='negative tweets')
plt.legend()
plt.xlabel('hour')


# ### Discussion:

# Trump seems to tweet much more negatively in his morning tweets, where tweets are most frequent in general from the president, and slightly more positive later in the day, with the switch coming around lunchtime. Perhaps the president enjoys a delicious lunch and begins to feel better about himself.
# 

# ## Submission
# 
# Congrats, you just finished Project 1!
