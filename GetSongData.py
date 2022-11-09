"""
Jake Kemple (2022) UW-Bothell
CSS581 Machine Learning Project
Get Song Data for Classifier Neural Network
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from configparser import RawConfigParser
import csv

if __name__ == '__main__':
    config = RawConfigParser()
    config.read('account.ini')
    cid = config.get('Spotify', 'CLIENT_ID')
    secret = config.get('Spotify', 'CLIENT_SECRET')

    song_data = []

    # Authentication - without user
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    likes_playlist = "https://open.spotify.com/playlist/2kLQgqz4oDlYk0wRuIJYde?si=e4c5dab8d6624bb5"
    # (TEST) likes_playlist = "https://open.spotify.com/playlist/36tmxCwDIGmNY2hF4CsUJw?si=674cb9d61e4e4790"

    likes_playlist_URI = likes_playlist.split("/")[-1].split("?")[0]

    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(likes_playlist_URI)["items"]]
    for track in sp.playlist_tracks(likes_playlist_URI)["items"]:
        song = {}

        #URI
        song['track_uri'] = track["track"]["uri"]
        
        #Track name
        song['track_name'] = track["track"]["name"]
        
        #Main Artist
        song_artist_uri = track["track"]["artists"][0]["uri"]
        song_artist_info = sp.artist(song_artist_uri)
        
        #Artist Name, Popularity, Genre
        song['artist_name'] = track["track"]["artists"][0]["name"]
        song['artist_pop'] = song_artist_info["popularity"]
        song['artist_genres'] = song_artist_info["genres"]
        
        # Album Name
        song['album'] = track["track"]["album"]["name"]
        
        # Popularity of the track
        song['track_pop'] = track["track"]["popularity"]

        # Audio Features of the track
        audio_feats = sp.audio_features(song['track_uri'])[0]
        song['danceability'] = audio_feats['danceability']
        song['energy'] = audio_feats['energy']
        song['key'] = audio_feats['key']
        song['loudness'] = audio_feats['loudness']
        song['mode'] = audio_feats['mode']
        song['speechiness'] = audio_feats['speechiness']
        song['acousticness'] = audio_feats['acousticness']
        song['instrumentalness'] = audio_feats['instrumentalness']
        song['liveness'] = audio_feats['liveness']
        song['valence'] = audio_feats['valence']
        song['tempo'] = audio_feats['tempo']
        song['type'] = audio_feats['type']
        song['id'] = audio_feats['id']
        song['uri'] = audio_feats['uri']
        song['track_href'] = audio_feats['track_href']
        song['analysis_url'] = audio_feats['analysis_url']
        song['time_signature'] = audio_feats['time_signature']

        # Like/Dislike Label
        song['like'] = 1

        song_data.append(song)

    with open('Liked_Songs.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(song_data[0].keys())
        for song in song_data:
            writer.writerow(song.values())
