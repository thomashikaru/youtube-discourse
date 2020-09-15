import pandas as pd
import os
import consts
import googleapiclient.discovery


class VideoDatabase:
    """
    Represents database of videos, enables lookup of channel names, channel ids, video ids, etc.
    """
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.vid2cname = {}

    def load_from_csv(self):
        self.df = pd.read_csv(self.filename, usecols=consts.VIDEO_COLS, lineterminator="\n")
        self.df.drop_duplicates(inplace=True)
        pairs = zip(list(self.df["videoId"]), list(self.df["channelName"]))
        self.vid2cname = dict(set(pairs))

    def append_data(self, new_df):
        self.df = self.df.append(new_df)

    def save_to_csv(self):
        self.df.to_csv(self.filename)

    def videoId_to_channelName(self, videoId):
        assert videoId in self.vid2cname
        return self.vid2cname[videoId]

    def channelName_to_channelId(self, channelName):
        assert channelName in self.df["channelName"].values
        return self.df[self.df["channelName"] == channelName]["channelId"].values[0]

    def sync_comments_file(self, comments_file):
        comm_df = pd.read_csv(comments_file, usecols=consts.COMMENT_COLS, lineterminator="\n")
        videos = set(comm_df["videoId"])
        existing_videos = set(self.df["videoId"])
        new_videos = videos - existing_videos  # set difference -> get videos in comments db not in videos db

        if new_videos:
            os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
            youtube = googleapiclient.discovery.build(
                consts.API_SERVICE_NAME, consts.API_VERSION, developerKey=consts.DEVELOPER_KEY)

            # get information for new videos
            video_data = []
            for v in new_videos:
                request = youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=v
                )
                response = request.execute()
                try:
                    title = response["items"][0]['snippet']['title']
                    publishedAt = response["items"][0]['snippet']['publishedAt']
                    channelId = response["items"][0]['snippet']['channelId']
                    channelName = response["items"][0]['snippet']['channelTitle']
                    views = response["items"][0]['statistics']['viewCount']
                    likes = response["items"][0]['statistics']['likeCount']
                    dislikes = response["items"][0]['statistics']['dislikeCount']
                    num_comments = response["items"][0]['statistics']['commentCount']
                    video_data.append([v, title, publishedAt, channelId,
                                       channelName, views, likes, dislikes, num_comments])
                except IndexError:
                    print("Index Error")
                    continue

            self.df = self.df.append(pd.DataFrame(video_data, columns=consts.VIDEO_COLS))
            self.save_to_csv()
            self.load_from_csv()
