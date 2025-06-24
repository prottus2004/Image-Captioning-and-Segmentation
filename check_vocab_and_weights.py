import pickle

with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

print("word2idx length:", len(word2idx))
print("idx2word length:", len(idx2word))

# Check mapping consistency for a few words/indices
for word, idx in list(word2idx.items())[:10]:
    print(word, idx, idx2word[idx])
