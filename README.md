# youtube-discourse
Analyzing political discourse and finding troll bots on YouTube using the YT Data API

## collect_data.py
Specify the channels that you wish to collect from, the number of videos to scrape, and the number of pages of comments per video to scrape. 
If a channel does not exist in the video database yet, you need to collect it by providing the channel ID, e.g. "UCXIJgqnII2ZOINSWNOGFThA" for Fox News. Once a channel exists in the database, you can use code like the following to convert from channel name to channel id:

```channel_ids = ["Fox News", "MSNBC"]
   channel_ids = map(lambda x: vdb.channelName_to_channelId(x), channel_ids)
```

## analysis.py
Data analysis and exploration code here. Broken into utility functions which do not need to be modified, and experiment functions, which should be modified and added to in order to set up new experiments. The main section runs the desired experiments. 

## datasets.py
Stores the VideoDatabase class. Include future database implementations here (e.g. if we add a ChannelDatabase). 

## consts.py 
Stores constants, API key, etc. which are shared between program files.  
