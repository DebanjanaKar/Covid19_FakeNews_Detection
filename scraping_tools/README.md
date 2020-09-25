We have created two notebooks one for extracting twitter features and one for annotating them.

-->The twitter feature extraction notebook produces two csv files. One with all the raw tweet features such as total number of user mentions, links, images etc., and one csv with all the user features such as followers count, friends count and many hand-crafted user features such as account life, that is the number of days since the account was created, posting frequency which tells us the number of days required by the user to post a new tweet, and so on.

--> In the feature extraction notebook, we have created a function which will extract all the user features as mentioned in the attached file. It takes 4 parameters as input. The parameters are the desired language of the tweet, the list of keywords that we need to be present in the tweet, the maximum number of tweets that we want with the limit of 1000 tweets as per twitter upi access, and the last parameter being how many request should be sent to twitter API per query.

--> In the annotation notebook, we just need to run the notebook and it will automatically load the user_features_extracted csv that was saved from the former notebook. The user can decide how many tweets he/she wants to label at the given time, for e.g. 10 tweets, then "starting_index" should be set to 11, for the next time we will run this notebook. We also give the link to the currently annotated tweet so that the user can very easily look at the entire tweet and decide the corresponding label for the tweet.

--> We designed these tools with major focus on Indic languages (Hindu and Bengali at present) as the data was not readily available and was difficult to annotate as well.
