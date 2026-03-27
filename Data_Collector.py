from googleapiclient.discovery import build
import pandas as pd
import os
import re


#API key for YouTube Data API v3
API_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_key)

#Function to get the channel ID from the username
def get_channel_id(username):
    req = youtube.channels().list(
        part='id', forUsername=username)
    return req.execute()['items'][0]['id']

#Function to get the video IDs from the channel ID
def get_video_ids(channel_id):
    ids = []
    request = youtube.search().list(
        part='id', channelId=channel_id,
        maxResults=50, type='video')
    response = request.execute()
    for item in response['items']:
        ids.append(item['id']['videoId'])
    return ids
    
#Function to get the video stats from the video IDs
def get_video_stats(video_ids):
    stats = []
    for video_id in video_ids:
        res = youtube.videos().list(
            part='statistics,snippet,contentDetails', 
            id=video_id).execute()
        item = res['items'][0]
        stats.append({
            'video_id': video_id,
            'views': int(item['statistics']['viewCount']),
            'likes': int(item['statistics'].get('likeCount', 0)),
            'title': item['snippet']['title'],
            'published': item['snippet']['publishedAt'],
            'duration': item['contentDetails']['duration'],
        })
    return pd.DataFrame(stats)
    
#Main function to get the video stats from the channel ID
channel_id = get_channel_id('youtube')
video_ids = get_video_ids(channel_id)
df = get_video_stats(video_ids)
df.to_csv('data/youtube_shorts_stats.csv', index=False)
print(f"Saved {len(df)} videos!")


#Data Cleaning and Feature Engineering
df = pd.read_csv('data/youtube_data.csv')
#Calculate the like ratio
df['like_ratio'] = df['likes'] / df['views']
#Parse the duration of the video
def parse_duration(duration):
    match = re.search(r'PT(\d+)S', str(duration))
    return int(match.group(1)) if match else 0
#Convert the duration of the video to seconds
df['duration_secs'] = df['duration'].apply(parse_duration)
#Convert the published date of the video to datetime
df['published'] = pd.to_datetime(df['published'])
df['day_of_week'] = df['published'].dt.dayofweek
df['hour_posted'] = df['published'].dt.hour
df['days_live'] = (pd.Timestamp.now() - df ['published']).dt.days
#Calculate the length of the title
df['title_length'] = df['title'].str.len()
df = df[df['views'] > 0]

