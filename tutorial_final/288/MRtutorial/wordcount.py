from mrjob.job import MRJob
from mrjob.step import MRStep
class MRWordFrequencyCount(MRJob):
    def mapper(self, _, line):
        line = line.strip()
        words = line.split()
        for word in words:
            # write result to standard output
            yield word, 1

    def reducer(self, word, count):
        yield word,sum(count)

if __name__ == '__main__':
    MRWordFrequencyCount.run()