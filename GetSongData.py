"""
Jake Kemple (2022) UW-Bothell
CSS581 Machine Learning Project
Get Song Data for Classifier Neural Network
"""

from configparser import RawConfigParser
import csv
import time
import tekore as tk
from secrets import *
import numpy as np

config = RawConfigParser()
config.read('account.ini')
cid = config.get('Spotify', 'CLIENT_ID')
secret = config.get('Spotify', 'CLIENT_SECRET')

token = tk.request_client_token(cid, secret)
spotify = tk.Spotify(token)

def retrieve_song_data(playlist, like):
    """
    Retrieve metadata & audio features from spotify for
    a given playlist.

    :param str playlist: The spotify playlist to receive data for
    :param int like: val correlating to desired 'like' col value ->
                     0 for no label, 1 for like, 2 for dislike
    """
    sp_items = []
    first_items = spotify.playlist_items(playlist)
    for item in spotify.all_items(first_items):
        # Append each track object to sp_items list
        sp_items.append(item)
        # sp_items = sp_items[-10:]

    ctr = 1
    for sp_item in sp_items:
        if ctr % 50 == 0:
            time.sleep(15)
        ctr += 1
        print(ctr)

        song = {}

        #URI
        # song['track_uri'] = sp_item.track.uri
        
        #Track name
        song['track_name'] = sp_item.track.name
        
        #Main Artist
        artist_id = sp_item.track.artists[0].id
        song_artist_info = spotify.artist(artist_id)
        
        #Artist Name, Popularity, Genre
        song['artist_name'] = song_artist_info.name
        song['artist_pop'] = song_artist_info.popularity
        song['artist_genres'] = np.array(song_artist_info.genres)
        
        # Album Name
        song['album'] = sp_item.track.album.name
        
        # Popularity of the track
        song['track_pop'] = sp_item.track.popularity

        # Audio Features of the track
        audio_feats = spotify.track_audio_features(sp_item.track.id)
        song['danceability'] = audio_feats.danceability
        song['energy'] = audio_feats.energy
        song['key'] = audio_feats.key
        song['loudness'] = audio_feats.loudness
        song['mode'] = audio_feats.mode
        song['speechiness'] = audio_feats.speechiness
        song['acousticness'] = audio_feats.acousticness
        song['instrumentalness'] = audio_feats.instrumentalness
        song['liveness'] = audio_feats.liveness
        song['valence'] = audio_feats.valence
        song['tempo'] = audio_feats.tempo
        song['id'] = audio_feats.id
        song['track_href'] = audio_feats.track_href
        song['analysis_url'] = audio_feats.analysis_url
        song['time_signature'] = audio_feats.time_signature

        # Like/Dislike Label
        if like == 1:
            song['like'] = 1
        elif like == 2:
            song['like'] = 0
        else:
            song['like'] = None

        song_data.append(song)


if __name__ == '__main__':

    song_data = []

    likes_playlist = "https://open.spotify.com/playlist/2kLQgqz4oDlYk0wRuIJYde?si=e4c5dab8d6624bb5"
    likes_playlist_URI = likes_playlist.split("/")[-1].split("?")[0]

    dislikes_playlist = "https://open.spotify.com/playlist/5EbV4OSgNBqBl8IRinrxOs?si=6dbf6c9490764208"
    dislikes_playlist_URI = dislikes_playlist.split("/")[-1].split("?")[0]

    unlabeled_playlist = "https://open.spotify.com/playlist/2y8nsWwLYF4tldbQ6N4zLr?si=6a32b8f103f645e8"
    unlabeled_playlist_URI = unlabeled_playlist.split("/")[-1].split("?")[0]

    sample_playlist = "https://open.spotify.com/playlist/6ONlEMBWFyjj8L0UndJ47X?si=ca8903937b3f4bfc"
    sample_playlist_URI = sample_playlist.split("/")[-1].split("?")[0]

    retrieve_song_data(likes_playlist_URI, 1)
    time.sleep(30)
    retrieve_song_data(dislikes_playlist_URI, 2)
    # retrieve_song_data(sample_playlist_URI, False)
    
    if song_data:
        with open('All_Songs.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(song_data[0].keys())
            for song in song_data:
                writer.writerow(song.values())

    # reset song data list and retrieve unlabeled
    # songs to be predicted on
    song_data = []
    retrieve_song_data(unlabeled_playlist_URI, 0)

    if song_data:
        with open('Unlabeled_Songs.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(song_data[0].keys())
            for song in song_data:
                writer.writerow(song.values())