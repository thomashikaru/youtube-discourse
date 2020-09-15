DEVELOPER_KEY = ""  # insert your API Key here 
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
VIDEO_COLS = ["videoId", "title", "publishedAt",
              "channelId", "channelName", "views", "likes", "dislikes", "num_comments"]
COMMENT_COLS = ["comment_id", "videoId", "textOriginal", "authorDisplayName",
                "authorChannelId", "likeCount", "publishedAt"]
VIDEOS_CSV = "videos.csv"
COMMENTS_CSV = "comments.csv"
W2V_SIZE = 64
