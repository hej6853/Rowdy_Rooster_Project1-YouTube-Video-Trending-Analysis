# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
import itertools
import nltk
import chardet
import string
import pprint
import itertools
import seaborn as sns
import scipy.stats as st
from scipy.stats import linregress
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from PIL import Image

# import file
US = pd.read_csv('Dataset/US_youtube_trending_data.csv') 
GB = pd.read_csv('Dataset/GB_youtube_trending_data.csv')
CA = pd.read_csv('Dataset/CA_youtube_trending_data.csv')

US['country'] = 'US'
GB['country'] = 'GB'
CA['country'] = 'CA'
frames = [US, GB, CA]

#merge
df = pd.concat(frames).drop_duplicates()

# Drop unnecessary rows 
df.drop(['channelId', 'thumbnail_link', 'comments_disabled', 'ratings_disabled'], inplace=True, axis=1)

df.head()

# Data preprocessing
# Lowercase title and tags columns 
df['title'] = df['title'].str.lower()
df['tags'] = df['tags'].str.lower()
df['description'] = df['description'].str.lower()

# Splitting tag and title contents for easier parsing
df['title content'] = df['title'].str.split()
df['tag content'] = df['tags'].str.split("|")
df['description content'] = df['description'].str.split()

# Getting the total word count of video title (title length)
df['total count title'] = df['title'].str.split().str.len()

# Getting the total tag count of video tags (tag length)
df['total count tag'] = df['tags'].str.split("|").str.len()
df.head()

#Divide into 3 countries
df_us = df[df['country'] == 'US']
df_gb = df[df['country'] == 'GB']
df_ca = df[df['country'] == 'CA']

#convert categoriesID to name
df_us['categoryId'] = df_us['categoryId'].replace([24,10,20,17,22,23,28,26,25,1,27,2,19,15,29],
                                               ["Entertainment","Music","Gaming","Sports","People & Blogs","Comedy","Science & Technology",
                                               "Howto & Style","News & Politics","Film & Animation","Education","Autos & Vehicles","Travel & Events",
                                               "Pets & Animals","Nonprofits & Activism"])
df_ca['categoryId'] = df_ca['categoryId'].replace([24,10,20,17,22,23,28,26,25,1,27,2,19,15,29],
                                               ["Entertainment","Music","Gaming","Sports","People & Blogs","Comedy","Science & Technology",
                                               "Howto & Style","News & Politics","Film & Animation","Education","Autos & Vehicles","Travel & Events",
                                               "Pets & Animals","Nonprofits & Activism"])
df_gb['categoryId'] = df_gb['categoryId'].replace([24,10,20,17,22,23,28,26,25,1,27,2,19,15,29],
                                               ["Entertainment","Music","Gaming","Sports","People & Blogs","Comedy","Science & Technology",
                                               "Howto & Style","News & Politics","Film & Animation","Education","Autos & Vehicles","Travel & Events",
                                               "Pets & Animals","Nonprofits & Activism"])

# i) Do title, tags and description word count affect viewership count ?
# title and tag count
# Getting the total word count of video title (title length)
df['total count title'] = df['title'].str.split().str.len()

# Getting the total tag count of video tags (tag length)
df['total count tag'] = df['tags'].str.split("|").str.len()

#tag word count
#Create total frequency count of individual tags 
df_us['tag content'].to_list()
us_tag_counts = dict(Counter(itertools.chain.from_iterable(df_us['tag content'].to_list())))

df_gb['tag content'].to_list()
gb_tag_counts = dict(Counter(itertools.chain.from_iterable(df_gb['tag content'].to_list())))

df_ca['tag content'].to_list()
ca_tag_counts = dict(Counter(itertools.chain.from_iterable(df_ca['tag content'].to_list())))

#Convert to dataframe and sort
us_tags = pd.DataFrame(list(us_tag_counts.items()),columns = ['tag','count']) 
us_tags = us_tags.sort_values(by='count', ascending=False)

gb_tags = pd.DataFrame(list(gb_tag_counts.items()),columns = ['tag','count']) 
gb_tags = gb_tags.sort_values(by='count', ascending=False)

ca_tags = pd.DataFrame(list(ca_tag_counts.items()),columns = ['tag','count']) 
ca_tags = ca_tags.sort_values(by='count', ascending=False)

# Title Count Plot Box
frames = [df_us, df_gb, df_ca]

# Merge all three dataframes
df_merge = pd.concat(frames)
my_pal = {"US": "b", "GB": "g", "CA":"r"}
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(15,10))
plt.suptitle("")
ax.set_title("Title Length Across Countries", fontdict={'fontsize':24})
sns.boxplot(x=df_merge["country"], y=df_merge["total count title"], palette=my_pal)

ax.set_xlabel("Countries", fontdict={'fontsize':24})
ax.set_ylabel("Title Length", fontdict={'fontsize':24})

plt.savefig("TitleCountBoxplot.png")
plt.show()

usa = df_merge[df_merge["country"] == 'US']
title_usa = usa['total count title']

gb = df_merge[df_merge["country"] == 'GB']
title_gb = gb['total count title']

ca = df_merge[df_merge["country"] == 'CA']
title_ca = ca['total count title']

#Quartile calculations for title length US
# Quartile calculations for title length US
us_quartiles = title_usa.quantile([.25,.5,.75])
lowerq = us_quartiles[0.25]
upperq = us_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of US title length is: {lowerq}")
print(f"The upper quartile of US title length is: {upperq}")
print(f"The interquartile range of US title length is: {iqr}")
print(f"The the median of US title length is: {us_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")

#Quartile calculations for title length GB
# Quartile calculations for title length GB
gb_quartiles = title_gb.quantile([.25,.5,.75])
gb_lowerq = gb_quartiles[0.25]
gb_upperq = gb_quartiles[0.75]
gb_iqr = gb_upperq-gb_lowerq

print(f"The lower quartile of GB title length is: {gb_lowerq}")
print(f"The upper quartile of GB title length is: {gb_upperq}")
print(f"The interquartile range of GB title length is: {gb_iqr}")
print(f"The the median of GB title length is: {gb_quartiles[0.5]} ")

gb_lower_bound = gb_lowerq - (1.5*gb_iqr)
gb_upper_bound = gb_upperq + (1.5*gb_iqr)
print(f"Values below {gb_lower_bound} could be outliers.")
print(f"Values above {gb_upper_bound} could be outliers.")

# Quartile calculations for title length CA
ca_quartiles = title_ca.quantile([.25,.5,.75])
ca_lowerq = ca_quartiles[0.25]
ca_upperq = ca_quartiles[0.75]
ca_iqr = ca_upperq-ca_lowerq

print(f"The lower quartile of CA title length is: {ca_lowerq}")
print(f"The upper quartile of CA title length is: {ca_upperq}")
print(f"The interquartile range of CA title length is: {ca_iqr}")
print(f"The the median of CA title length is: {ca_quartiles[0.5]} ")

ca_lower_bound = ca_lowerq - (1.5*ca_iqr)
ca_upper_bound = ca_upperq + (1.5*ca_iqr)
print(f"Values below {ca_lower_bound} could be outliers.")
print(f"Values above {ca_upper_bound} could be outliers.")

#Tag Count Boxplots
# Boxplot of tag count
my_pal = {"US": "b", "GB": "g", "CA":"r"}
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(15,10))
plt.suptitle("")
ax.set_title("Tag Count Across Countries", fontdict={'fontsize':24})
sns.boxplot(x=df_merge["country"], y=df_merge["total count tag"], palette=my_pal)

ax.set_xlabel("Countries", fontdict={'fontsize':24})
ax.set_ylabel("Tag Count", fontdict={'fontsize':24})

plt.savefig("TagCountBoxplot.png")
plt.show()

usa = df_merge[df_merge["country"] == 'US']
tag_usa = usa['total count tag']

gb = df_merge[df_merge["country"] == 'GB']
tag_gb = gb['total count tag']

ca = df_merge[df_merge["country"] == 'CA']
tag_ca = ca['total count tag']

# Quartile calculations for title length US
us_tag_quartiles = tag_usa.quantile([.25,.5,.75])
us_tag_lowerq = us_tag_quartiles[0.25]
us_tag_upperq = us_tag_quartiles[0.75]
us_tag_iqr = us_tag_upperq-us_tag_lowerq

print(f"The lower quartile of US title length is: {us_tag_lowerq}")
print(f"The upper quartile of US title length is: {us_tag_upperq}")
print(f"The interquartile range of US title length is: {us_tag_iqr}")
print(f"The the median of US title length is: {us_tag_quartiles[0.5]} ")

us_tag_lower_bound = us_tag_lowerq - (1.5*us_tag_iqr)
us_tag_upper_bound = us_tag_upperq + (1.5*us_tag_iqr)
print(f"Values below {us_tag_lower_bound} could be outliers.")
print(f"Values above {us_tag_upper_bound} could be outliers.")

# Quartile calculations for title length GB
gb_tag_quartiles = tag_gb.quantile([.25,.5,.75])
gb_tag_lowerq = gb_tag_quartiles[0.25]
gb_tag_upperq = gb_tag_quartiles[0.75]
gb_tag_iqr = gb_tag_upperq-gb_tag_lowerq

print(f"The lower quartile of US title length is: {gb_tag_lowerq}")
print(f"The upper quartile of US title length is: {gb_tag_upperq}")
print(f"The interquartile range of US title length is: {gb_tag_iqr}")
print(f"The the median of US title length is: {gb_tag_quartiles[0.5]} ")

gb_tag_lower_bound = gb_tag_lowerq - (1.5*gb_tag_iqr)
gb_tag_upper_bound = gb_tag_upperq + (1.5*gb_tag_iqr)
print(f"Values below {gb_tag_lower_bound} could be outliers.")
print(f"Values above {gb_tag_upper_bound} could be outliers.")

# Quartile calculations for title length CA
ca_quartiles = title_ca.quantile([.25,.5,.75])
ca_lowerq = ca_quartiles[0.25]
ca_upperq = ca_quartiles[0.75]
ca_iqr = ca_upperq-ca_lowerq

print(f"The lower quartile of CA title length is: {ca_lowerq}")
print(f"The upper quartile of CA title length is: {ca_upperq}")
print(f"The interquartile range of CA title length is: {ca_iqr}")
print(f"The the median of CA title length is: {ca_quartiles[0.5]} ")

ca_lower_bound = ca_lowerq - (1.5*ca_iqr)
ca_upper_bound = ca_upperq + (1.5*ca_iqr)
print(f"Values below {ca_lower_bound} could be outliers.")
print(f"Values above {ca_upper_bound} could be outliers.")

#ii) Do title, tags content by categories affect viewership count?
#1) Categories Count Bar Chart
# Create a dataframe with categoryId counts
category_counts_us = df_us['categoryId'].value_counts().to_dict()
category_counts_gb = df_gb['categoryId'].value_counts().to_dict()
category_counts_ca = df_ca['categoryId'].value_counts().to_dict()

df_ca_cat = pd.DataFrame(list(category_counts_ca.items()),columns = ['category','count']) 
df_gb_cat = pd.DataFrame(list(category_counts_gb.items()),columns = ['category','count'])  
df_us_cat = pd.DataFrame(list(category_counts_us.items()),columns = ['category','count'])

# Define the sorter
sorter = ['Music', 'Entertainment', 'Gaming', 'Sports', 'People & Blogs', 'Comedy', 'Science & Technology', 'News & Politics',
          'Howto & Style','Film & Animation','Education', 'Autos & Vehicles', 'Pets & Animals', 
          'Travel & Events', 'Nonprofits & Activism']

# Create the dictionary that defines the order for sorting
sorterIndex = dict(zip(sorter, range(len(sorter))))

# Generate a rank column that will be used to sort
# the dataframe numerically
df_ca_cat['Tm_Rank'] = df_ca_cat['category'].map(sorterIndex)
df_us_cat['Tm_Rank'] = df_us_cat['category'].map(sorterIndex)
df_gb_cat['Tm_Rank'] = df_gb_cat['category'].map(sorterIndex)

# Here is the result asked with the lexicographic sort
# Result may be hard to analyze, so a second sorting is
# proposed next
## NOTE: 
## Newer versions of pandas use 'sort_values' instead of 'sort'
df_ca_cat.sort_values(['Tm_Rank'], ascending=True, inplace = True)
df_ca_cat.drop('Tm_Rank', 1, inplace = True)

df_us_cat.sort_values(['Tm_Rank'], ascending=True,inplace = True)
df_us_cat.drop('Tm_Rank', 1, inplace = True)

df_gb_cat.sort_values(['Tm_Rank'], ascending=True,inplace = True)
df_gb_cat.drop('Tm_Rank', 1, inplace = True)

x_ticks = []

x_axis4 = df_us_cat['category']
y_axis4 = df_us_cat['count']

x_axis5 = df_gb_cat['category']
y_axis5 = df_gb_cat['count']

x_axis6 = df_ca_cat['category']
y_axis6 = df_ca_cat['count']

# create data

ind = np.arange(15) 
width = 0.25

 
# Make the plot
plt.figure(figsize=(12, 10))

plt.barh(ind, y_axis4, color='blue', height=width)
plt.barh(ind+width, y_axis5, color='green', height=width)
plt.barh(ind+width*2, y_axis6, color='red', height=width)
plt.gca().invert_yaxis()

plt.xlabel("Upload Count")
plt.ylabel("Category")
plt.title("Popular Categories Across Countries")
plt.yticks(np.arange(15),['Music', 'Entertainment', 'Gaming', 'Sports', 'People & Blogs', 'Comedy', 'Science & Technology', 'News & Politics',
          'Howto & Style','Film & Animation','Education', 'Autos & Vehicles', 'Pets & Animals', 
          'Travel & Events', 'Nonprofits & Activism'])

plt.legend(handles=[us_avg, gb_avg, ca_avg], loc="lower right")
plt.show()

plt.figure(figsize=(12, 6))


us_avg, = plt.plot(x_axis4, y_axis4, color="blue", label="United States" )
gb_avg, = plt.plot(x_axis4, y_axis5, color="green", label="Great Britain" )
ca_avg, = plt.plot(x_axis4, y_axis6, color="red", label="Canada" )



plt.xlabel("Category")
plt.ylabel("Upload Count")
plt.title("Popular Categories Across Countries")

plt.legend(handles=[us_avg, gb_avg, ca_avg], loc="best")

#2) What are the top N hot topics for each category of videos?
#Top categories for each country¶
#df_gb category video count vs top views : 10: music, 24:Entertainment, 20: gaming
df_us_topcategory = pd.DataFrame(df_us.groupby('categoryId')['view_count'].sum()).sort_values(by = 'view_count',ascending=False).reset_index()
video_count = pd.DataFrame(df_us['categoryId'].value_counts()).reset_index().rename(columns={'index' : 'categoryId',
                                                                                             'categoryId':'video_count'})
df_us_topcategory = df_us_topcategory.merge(video_count, how = 'inner', on = 'categoryId')

#Create total frequency count of individual words in title 
us_title_list = df_us['title content'].to_list()
us_all_title_counts = dict(Counter(itertools.chain.from_iterable(df_us['title content'].to_list())))

gb_title_list = df_gb['title content'].to_list()
gb_all_title_counts = dict(Counter(itertools.chain.from_iterable(df_gb['title content'].to_list())))

ca_title_list = df_ca['title content'].to_list()
ca_all_title_counts = dict(Counter(itertools.chain.from_iterable(df_ca['title content'].to_list())))

#Convert to dataframe and sort
df_title_us = pd.DataFrame(list(us_all_title_counts.items()),columns = ['word','count']) 
df_title_us.sort_values(by='count', ascending=False)

df_title_gb = pd.DataFrame(list(gb_all_title_counts.items()),columns = ['word','count']) 
df_title_gb.sort_values(by='count', ascending=False)

df_title_ca = pd.DataFrame(list(ca_all_title_counts.items()),columns = ['word','count']) 
df_title_ca.sort_values(by='count', ascending=False)

#Extracting hot topics with NLTK
#split categories
df_us_10 = df_us[df_us['categoryId'] == 10]
df_us_24 = df_us[df_us['categoryId'] == 24]
df_us_20 =  df_us[df_us['categoryId'] == 20]
df_us_25 =  df_us[df_us['categoryId'] == 25]

#Textual Analysis
##Extracting hot topics with NLTK
text = df_us_10['title'].str.lower().replace('|', ' ').str.cat(sep=' ')


stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
        
# Stemming with NLTK
Stem_words = []
ps =PorterStemmer()
for w in filtered_sentence:
    rootWord=ps.stem(w)
    Stem_words.append(rootWord)
    
# Lemmatization with NLTK
filtered_sentence = list(filter(lambda token: token not in string.punctuation, filtered_sentence))
filtered_sentence

# remove unnecessay words
stopwords = ["'s", "’", "..." , "ft." , "2" ,"x" , "1", "n't", "–", "3", "5", "4",
             "2021","2020","trailer", "de", "official", "season", "video", "official", "season", "episode","la", "le", "je",
             "part", "je", "des","world","day", "10","e", "avec", "‘", "à", "music", "none", "new","lil", "like", "songs", "song",
            "thee","love","bad","g","tv", "voice","game", "news","live","watch", "full", "today", "uk" ]
for word in list(filtered_sentence):  # iterating on a copy since removing will mess things up
    if word in stopwords:
        filtered_sentence.remove(word)

#Generate World Cloud
#wordcloud
word_could_dict= Counter(filtered_sentence)

wordcloud = WordCloud(width = 1000, height = 500, background_color ='black',
                      stopwords = stopwords,
                      min_font_size = 10).generate_from_frequencies(word_could_dict)


plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# plt.savefig('us20_tags_wordcloud.png', bbox_inches='tight')
plt.close()

#Hot Topic words count
filtered_sentence = pd.DataFrame(filtered_sentence)
filtered_sentence_unique = pd.DataFrame(filtered_sentence.value_counts())
filtered_sentence_unique = filtered_sentence_unique.rename(columns={'0':'count'})
filtered_sentence_unique.head(50)

# Convert list to string
# using list comprehension
listToStr = ' '.join([str(elem) for elem in filtered_sentence_unique])
  
listToStr

#iii) Does published time affect viewership count?
#Published Times Analysis
# Remove the dates, mins, and seconds in 'publshedAt' column 
df_us['publishedAt'] = df_us['publishedAt'].str[10:]
df_us['publishedAt'] = df_us['publishedAt'].str[:3]

df_gb['publishedAt'] = df_gb['publishedAt'].str[10:]
df_gb['publishedAt'] = df_gb['publishedAt'].str[:3]

df_ca['publishedAt'] = df_ca['publishedAt'].str[10:]
df_ca['publishedAt'] = df_ca['publishedAt'].str[:3]


# Create a dataframe with published time counts
time_counts_us = df_us['publishedAt'].value_counts().to_dict()
time_counts_gb = df_gb['publishedAt'].value_counts().to_dict()
time_counts_ca = df_ca['publishedAt'].value_counts().to_dict()

#Published Time line plot
df_ca_time = pd.DataFrame(list(time_counts_ca.items()),columns = ['time','count']).sort_values(by=['time']) 
df_gb_time = pd.DataFrame(list(time_counts_gb.items()),columns = ['time','count']).sort_values(by=['time'])  
df_us_time = pd.DataFrame(list(time_counts_us.items()),columns = ['time','count']).sort_values(by=['time'])

x_ticks = []

x_axis = df_us_time['time']
y_axis = df_us_time['count']

x_axis2 = df_ca_time['time']
y_axis2 = df_ca_time['count']

x_axis3 = df_gb_time['time']
y_axis3 = df_gb_time['count']

plt.figure(figsize=(12, 6))

us_avg, = plt.plot(x_axis, y_axis, color="blue", label="United States" )
gb_avg, = plt.plot(x_axis3, y_axis3, color="green", label="Great Britain" )
ca_avg, = plt.plot(x_axis2, y_axis2, color="red", label="Canada" )

plt.xlabel("Published Time")
plt.ylabel("Upload Count")
plt.title("Published Times Across Countries")
plt.xticks([0, 2, 4,6,8,10,12,14,16,18,20,22], 
           ['12AM', '2AM','4AM','6AM','8AM','10AM','12PM','2PM','4PM','6PM','8PM', '10PM'])


plt.legend(handles=[us_avg, gb_avg, ca_avg], loc="best")