# CS 4641 Project

Spotify is a media streaming platform that allows users to listen to upwards of 60 million songs. Spotify provides users with information about songs like music genre, song length, artist, album. By tallying the number of individual streams of a song (or any piece of media), Spotify also provides users (and artists) with a metric of how popular a song is. Behind the scenes, Spotify keeps track of features of songs including modality, “danceability, energy, and tone. We have reason to believe that these features created by spotify are capable of predicting some level of popularity amongst other song titles with similar features. We have examples of analysing these sorts of features to support us [1,4]. 


# Problem Statement

Spotify has been an ever improving platform for a while now and currently holds the top spot among audio streaming platforms. Many artists recognize the importance of such a platform and have been utilizing the platform even more to attract listeners and build fanbases.
Given that Spotify measures a large variety of information about the songs on its platform, we intend to investigate how the relationships between these features, both front-facing and not, interweave to affect the popularity of music upon release by predicting the popularity of a song given certain features. These predictive capabilities would be useful for artists and labels looking to optimize their music outputs in an effective way [1]. Though some other sources exist that have intended to achieve a similar goal, we hope to use a combination of unsupervised and supervised learning to achieve a more optimal and accurate analysis.

# Methods

Spotify’s metrics give us unique insight into how songs are distributed with respect to traits like “danceability” and “acousticness” which are difficult to directly quantify [2]. Our first task will be to make sense of how these traits relate to each other and how they inform a song’s popularity and genre classification. We intend to plot the dataset using a variety of methods, from simple covariance metrics to see which attributes are related to each other to unsupervised clustering based on algorithms like DBSCAN to see what natural groups the songs fall into [3]. With this information we can hopefully get an idea of which features are more relevant to our task and how they relate to each other. 
	In order to build a classifier to predict whether a song will be popular or we will create a cutoff for the “popularity” dataset and divide the training examples into “popular” and “not popular” classes. We will then train multiple models to see whether we can create a reliable predictor. As a starting point we will train a basic dense neural network with Keras as well as a decision tree model. This should give us a good idea of how different types of algorithms respond to the data. We can pick the more promising approach and tune the hyperparameters to produce an optimized model for popularity prediction. 

# Potential Results

Measure the chance of song becoming popular based on attributes of the song such as danceability, energy, and tempo.
Also be able to figure out which features of songs play the largest role in determining a song’s fate. Whether it’ll be remembered as a bop or gather dust in the cloud. 

With our model, we hope to measure the chance of a song becoming popular based on attributes of the song such as its danceability, energy, and tempo. By determining which attributes are the most important when determining the fate of a song as well as which songs have the perfect combination of said features, our model will determine whether a song will likely be remembered as a bop or gather dust in the cloud.

# Discussion
Our chosen datasets are not bound to cover the same range of songs so it is highly likely that we will have to acquire additional data from the respective APIs to have a consistent set of songs that we can evaluate metrics on. Sanitizing the datasets will be important given the ratio of popular to unpopular songs is not equal. Expanding on the issue of popularity, the datasets will likely contain songs that do not equally have a balance of metrics (we expect more songs from the “pop” genre etc.) and in the process of scraping more songs, we need to balance our dataset. 


# References
[1] http://cs229.stanford.edu/proj2015/140_report.pdf 
[2] https://github.com/MattD82/Predicting-Spotify-Song-Popularity
[3] https://dl.acm.org/doi/pdf/10.1145/3068335 
[4] https://arxiv.org/pdf/1908.08609.pdf 


