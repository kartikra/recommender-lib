import csv
import re
from collections import defaultdict
from surprise import Dataset
from surprise import Reader


class MovieLens:

    def __init__(self, ratings_path, movies_path, movie_content_path=None):
        """_summary_

        Args:
            ratings_path (Str): Path of file in csv format that contains ratings
            movies_path (Str): Path of file in csv format that contains movie details
            movie_content_path (Str): Path of file in csv format that contains movie metadata 9use for content based recommender)
        """
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.movie_content_path = movie_content_path
        self.move_id_to_name = {}
        self.name_to_movie_id = {}
    
    def load_dataset(self, line_format='user item rating timestamp', sep=',', skip_lines=1):
        dataset_ratings = 0
        reader = Reader(line_format=line_format, sep=sep, skip_lines=skip_lines)
        dataset_ratings = Dataset.load_from_file(self.ratings_path, reader=reader)
        return dataset_ratings
    
    def get_popularity_rankings(self):
        """
        Compute popularity rankings from csv file under ratings_path
        """
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csvfile:
            reader_rating = csv.reader(csvfile)
            next(reader_rating)
            for row in reader_rating:
                movie_id = int(row[1])
                ratings[movie_id] += 1
        rank = 1
        # Sort in descending order of popularity
        for movie_id, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1
        return rankings
    
    def get_genres(self):
        """
        Extract genres from csv file under movies_path
        """
        genres = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            reader_movie = csv.reader(csvfile)
            next(reader_movie)  #Skip header line
            for row in reader_movie:
                movie_id = int(row[0])
                list_genre = row[2].split('|')
                list_genre_id = []
                for genre in list_genre:
                    if genre in genre_ids:
                        genre_id = genre_ids[genre]
                    else:
                        genre_id = max_genre_id
                        genre_ids[genre] = genre_id
                        max_genre_id += 1
                    list_genre_id.append(genre_id)
                genres[movie_id] = list_genre_id
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movie_id, list_genre_id) in genres.items():
            bitfield = [0] * max_genre_id
            for genre_id in list_genre_id:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield            
        
        return genres
    
    def get_years(self):
        """
        Extract release year from csv file under movies_path
        """
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            reader_movie = csv.reader(csvfile)
            next(reader_movie)
            for row in reader_movie:
                movie_id = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movie_id] = int(year)
        return years
    
    def get_movie_content_attributes(self):
        """
        Extract movie meatadata from csv file under movie_content_path
        """
        mes = defaultdict(list)
        with open(self.movie_content_path, newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movie_id = int(row[0])
                avg_shot_length = float(row[1])
                mean_color_variance = float(row[2])
                stddev_color_variance = float(row[3])
                mean_motion = float(row[4])
                stddev_motion = float(row[5])
                mean_lighting_key = float(row[6])
                num_shots = float(row[7])
                mes[movie_id] = [avg_shot_length, mean_color_variance, stddev_color_variance,
                   mean_motion, stddev_motion, mean_lighting_key, num_shots]
        return mes
    
    def get_user_ratings(self, user):
        """Get user ratings for a given user
        Args:
            user (str): specify user_id

        Returns:
            _type_: user_ratings for the user
        """
        user_ratings = []
        hitUser = False
        with open(self.ratings_path, newline='') as csvfile:
            reader_rating = csv.reader(csvfile)
            next(reader_rating)
            for row in reader_rating:
                user_id = int(row[0])
                if (user == user_id):
                    movie_id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_id, rating))
                    hitUser = True
                if (hitUser and (user != user_id)):
                    break

        return user_ratings

    def create_movie_lookups(self):
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            reader_movie = csv.reader(csvfile)
            next(reader_movie)  #Skip header line
            for row in reader_movie:
                movie_id = int(row[0])
                movieName = row[1]
                self.move_id_to_name[movie_id] = movieName
                self.name_to_movie_id[movieName] = movie_id

    def get_movie_name(self, movie_id):
        if movie_id in self.move_id_to_name:
            return self.move_id_to_name[movie_id]
        else:
            return ""
        
    def get_movie_id(self, movieName):
        if movieName in self.name_to_movie_id:
            return self.name_to_movie_id[movieName]
        else:
            return 0
