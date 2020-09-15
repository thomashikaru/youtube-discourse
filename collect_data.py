import os
import time
import pandas as pd
import googleapiclient.discovery
from googleapiclient.errors import HttpError
import datasets
import consts


def get_videos_from_channel(channel_id, n):
    """
    Call YT API and get list of n videos for a given channel ID.
    :param channel_id: unique identifier for channel to scrape
    :param n: number of videos to return
    :return: list of video ids
    """
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        consts.API_SERVICE_NAME, consts.API_VERSION, developerKey=consts.DEVELOPER_KEY)

    # first need to get the identifier for the uploads playlist
    request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )
    response = request.execute()
    uploads = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # then get the list of videos in the uploads playlist
    request = youtube.playlistItems().list(
        part="snippet, contentDetails",
        maxResults=n,
        playlistId=uploads
    )
    response = request.execute()
    video_id_list = []
    for item in response["items"]:
        print("Title: {}, Date: {}, Video ID: {}".format(item['snippet']['title'], item['snippet']['publishedAt'],
                                                         item['snippet']['resourceId']['videoId']))
        video_id_list.append(item['snippet']['resourceId']['videoId'])
    return video_id_list


def extract_comments_from_response(response):
    """
    Get list of comments from a JSON response.
    :param response: JSON response from a call to YouTube Data API's CommentThreads.list()
    :return: A list of lists, where each list stores comment_id, video_id, original text,
             author's display name, author's channel ID, like count, and timestamp
    """
    comments = []
    for item in response["items"]:
        try:
            comment_id = item['snippet']['topLevelComment']['id']
            video_id = item['snippet']['videoId']
            textOriginal = item['snippet']['topLevelComment']['snippet']['textOriginal']
            authorDisplayName = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            authorChannelId = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
            likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']
            publishedAt = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments.append([comment_id, video_id, textOriginal, authorDisplayName,
                         authorChannelId, likeCount, publishedAt])
        except KeyError as e:
            print(e)
            continue
    return comments


def get_comments_from_video(videoId, n):
    """
    Get list of comments from a video identified by its unique videoId
    :param video_id:
    :param n:
    :return:
    """
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        consts.API_SERVICE_NAME, consts.API_VERSION, developerKey=consts.DEVELOPER_KEY)

    # first request is made without specifying a page token
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=videoId,
        maxResults=100,
        order="time"
    )

    try:
        response = request.execute()
    except HttpError as e:
        print(e)
        return

    next_page_token = None
    if "nextPageToken" in response:
        next_page_token = response["nextPageToken"]

    comments = []
    tokenLL = extract_comments_from_response(response)
    for tokenL in tokenLL:
        print(tokenL)
    comments += tokenLL

    # continue retrieving comments while comments exist and count hasn't reached desired count
    count = 1
    while next_page_token and count < n:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=videoId,
            pageToken=next_page_token,
            maxResults=100,
            order="time"
        )

        try:
            response = request.execute()
        except HttpError as e:
            print(e)
            return

        next_page_token = None
        if "nextPageToken" in response:
            next_page_token = response["nextPageToken"]
        comments += extract_comments_from_response(response)
        count += 1
        time.sleep(0.5)

    # read the existing database in order to append, or create new one if database doesn't exist yet
    if os.path.exists(consts.COMMENTS_CSV):
        df = pd.read_csv(consts.COMMENTS_CSV, usecols=consts.COMMENT_COLS, lineterminator="\n")
        df = df.append(pd.DataFrame(comments, columns=consts.COMMENT_COLS))
    else:
        df = pd.DataFrame(comments, columns=consts.COMMENT_COLS)

    # remove duplicates and n/a values, save
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.to_csv(consts.COMMENTS_CSV)


if __name__ == "__main__":

    # load and sync videos database
    vdb = datasets.VideoDatabase(consts.VIDEOS_CSV)
    vdb.load_from_csv()
    vdb.sync_comments_file(consts.COMMENTS_CSV)

    # specify channels to scrape
    channel_ids = ["UCXIJgqnII2ZOINSWNOGFThA"] # Fox News channel ID
    
    # once a channel is in the database, you can look it up by name
    # channel_ids = ["Fox News", "MSNBC"]
    # channel_ids = map(lambda x: vdb.channelName_to_channelId(x), channel_ids)

    # specify these parameters
    videos_per_channel = 5
    comment_pages_per_video = 10

    for channel in channel_ids:
        videos = get_videos_from_channel(channel, videos_per_channel)
        for video in videos:
            get_comments_from_video(video, comment_pages_per_video)

    # sync the videos database
    vdb.sync_comments_file(consts.COMMENTS_CSV)
