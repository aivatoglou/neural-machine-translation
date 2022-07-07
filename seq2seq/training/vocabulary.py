class LanguageIndex:
    def __init__(self, lang):

        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):

        for phrase in self.lang:
            self.vocab.update(phrase.split(" "))

        self.vocab = sorted(self.vocab)

        self.word2idx["<pad>"] = 0
        self.word2idx["<start>"] = 1
        self.word2idx["<end>"] = 2

        for index, word in enumerate(self.vocab[2:], start=2):  # ignore <start> <end>
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word
