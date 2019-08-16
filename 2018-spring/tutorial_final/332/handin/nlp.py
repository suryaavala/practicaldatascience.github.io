import numpy as np
from collections import defaultdict
class LanguageModel:
    def __init__(self, docs, n):
        """
        Initialize an n-gram language model.
        
        Args:
            docs: list of strings, where each string represents a space-separated
                  document
            n: integer, degree of n-gram model
        """
        document_words = [doc.split() for doc in docs]
        self.n = n
        self.counts = defaultdict(lambda: defaultdict(int))
        self.count_sums = defaultdict(int)
        self.dictionary = set()
        
        for doc in document_words:
            self.dictionary.update(doc)
            for i in range(len(doc) -  n + 1):
                first = " ".join(doc[i:i+n-1])
                last = doc[i+n-1]
                
                self.count_sums[first] += 1
                self.counts[first][last] += 1
    
    def poss(self, first):
        dic = self.counts[first]
        total = self.count_sums[first]
        if total == 0:
            return None
        return np.random.choice(list(dic.keys()), p = np.array(list(dic.values()))/ total)
    
    def first(self):
        total = sum(self.count_sums.values())
        return np.random.choice(list(self.count_sums.keys()), p = np.array(list(self.count_sums.values()))/ total)
    
    def sample(self, k, first = None):
        """
        Generate a random sample of k words.
        
        Args:
            k: integer, indicating the number of words to sample
            
        Returns: text
            text: string of words generated from the model.
        """
        if not first:
            first = self.first()
        ans = first
        for i in range(k - self.n + 1):
            ans += " "
            last = self.poss(first)
            if last is None:
                return self.sample(k)
            ans += last
            first = first.split(' ', 1)[1]
            first += " " + last
        return ans