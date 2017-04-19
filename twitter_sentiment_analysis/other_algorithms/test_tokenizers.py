from nltk.tokenize import TweetTokenizer


tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
print tknzr.tokenize(s1)
