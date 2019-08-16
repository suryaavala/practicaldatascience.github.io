from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import math
from itertools import combinations

class MRMovieSimilarity(MRJob):

    #(user_id,movie_id,rating,timestamp) -> (userid,(movie_id,rating))
    def mapper_extract_data(self, _, record):
        user_id, movie_id, rating, timestamp = record.split()
        yield user_id, (movie_id, rating)

    # (userid,(movie_id,rating)) -> (userid,[(m1,r1),(m2,r2)...])
    def reducer_combine_singleUser_ratings(self, user_id, rating_info):
        all_ratings = []
        for movie_id, rating in rating_info:
            all_ratings.append((movie_id, rating))
        yield None, all_ratings

    # (userid,[(m1,r1),(m2,r2)...]) -> ((m1,m2),(r1,r2))...
    def mapper_generate_rating_pairs(self,_,all_ratings):
        for movie1info,movie2info in combinations(all_ratings,2):
            m1 = movie1info[0]
            r1 = movie1info[1]
            m2 = movie2info[0]
            r2 = movie2info[1]
            if m1 < m2:
                yield (m1,m2),(r1,r2)
            if m2 < m1:
                yield (m2,m1),(r2,r1)

    def calculate_correlation(self,x,y):
        n = len(x)
        x = np.array(x)
        y = np.array(y)

        x_sum = x.sum()
        y_sum = y.sum()

        x_sqr = x ** 2
        y_sqr = y ** 2

        x_sqr_sum = x_sqr.sum().item()
        y_sqr_sum = y_sqr.sum().item()
        xy = x * y
        xy_sum = xy.sum()

        res = (n * xy_sum - x_sum * y_sum) / (
        math.sqrt(n * x_sqr_sum - x_sum ** 2) * math.sqrt(n * y_sqr_sum - y_sum ** 2))

        return res;

    # (m1,m2), cosine similarity score
    def reducer_cosine_similarity(self,movie_pair,rating_pair):
        rating_1 = []
        rating_2 = []
        for r1,r2 in rating_pair:
            rating_1.append(float(r1))
            rating_2.append(float(r2))
        if len(rating_1) > 10:
            #sim = dot(rating_1, rating_2) / (norm(rating_1) * norm(rating_2))
            sim = self.calculate_correlation(rating_1,rating_2)
            yield None,(movie_pair,sim)

    def map_movie_names(self,movie_pair):
        if  self.name_dict[movie_pair[0]] == self.name_dict[movie_pair[1]]:
            return None
        return self.name_dict[movie_pair[0]],self.name_dict[movie_pair[1]]

    def find_topK_moviepairs(self,_,sim_info):
        K = 20
        cnt = 0
        sorted_siminfo = sorted(list(sim_info), key=lambda a: float(a[1]), reverse=True)
        for movie_pair,score in sorted_siminfo:
            if self.map_movie_names(movie_pair):
                yield self.map_movie_names(movie_pair),score
                cnt += 1
            if cnt >= K:
                break

    def get_movies_names(self):
        self.name_dict = {}
        with open('u.item',encoding = "ISO-8859-1") as f:
            for line in f:
                fields = line.split('|')
                self.name_dict[(fields[0])] = fields[1]


    def configure_args(self):
        super(MRMovieSimilarity, self).configure_args()
        self.add_file_arg('--items', help='u.item (which contains movie info) path')


    def steps(self):
        return [MRStep(mapper=self.mapper_extract_data, reducer=self.reducer_combine_singleUser_ratings),
                MRStep(mapper=self.mapper_generate_rating_pairs, reducer = self.reducer_cosine_similarity),
                MRStep( reducer_init=self.get_movies_names, reducer=self.find_topK_moviepairs)]

if __name__ == '__main__':
    MRMovieSimilarity.run()
