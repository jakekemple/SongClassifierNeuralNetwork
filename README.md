# SongClassifierNeuralNetwork
UW CSS 581 Machine Learning Project: Personalized Song Recommendation & Audio Feature Identification

## Presentation Materials

[Presentation](https://drive.google.com/file/d/1cm5X2AOGS0e0fRbbqie0bdl7gMoMV8Cu/view?usp=sharing)

[Report](https://drive.google.com/file/d/1PmU0BHjCdAFGKBG76EJkc1dfi5WnWD0R/view?usp=sharing)

## About

In large-scale commercial music recommendation systems, songs are typically recommended 
based on insight about the songs that other users are listening to who also listened to the
songs you’ve listened to. More recently, machine learning algorithms are now being
developed with the sophistication of comparing audio features within a song to
classify/cluster similar songs. Although these methods are effective in helping determine
songs that may be enjoyable to a given listener, song recommenders typically fall short in
identifying specifically why a listener may enjoy a given song, and instead rely on similarity
detection between songs themselves. Recommendation systems which rely on similar user
behavior fall short in that they rarely recommend songs that are less popular and difficult to
discover, defeating a core purpose of song recommendation systems.

To solve the problem described above, I developed an MLP Neural Network classification
model trained on labeled song audio feature data as a starting point for improved song
recommendation tailoring to a given user. As the target user, the binary prediction labels will
represent whether I personally like or dislike songs represented by the song audio feature
data.
