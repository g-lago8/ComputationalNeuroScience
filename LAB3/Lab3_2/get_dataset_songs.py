# scripts for getting all song lyrics from a given artist

import lyricsgenius as lg
import os
from dotenv import load_dotenv

def get_lyrics(artist, out_file):
    load_dotenv()
    api_key = os.getenv("GENIUS_ACCESS_TOKEN")
    genius = lg.Genius(api_key, 
                       skip_non_songs=True, 
                       excluded_terms=["(Remix)", "(Live)"], 
                       remove_section_headers=True, timeout=10, 
                       sleep_time=0.5, 
                       verbose=True)

    artist = genius.search_artist(artist, sort="popularity")

    with open(out_file, "w", encoding="utf-8") as f:
        for song in artist.songs:
            f.write(song.lyrics)
            f.write("\n\n")


if __name__ == "__main__":
    get_lyrics("blink-182", "lyrics_blink-182.txt")
