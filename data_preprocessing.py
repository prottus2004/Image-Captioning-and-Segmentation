# data_preprocessing.py

import json
import nltk
import os
from collections import Counter

# Download NLTK tokenizer data (only if not already present)
nltk.download('punkt')
nltk.download('punkt_tab')

# Define the path to the captions file (use forward slashes or raw string)
CAPTIONS_PATH = os.path.join( "C:\\Users", "Pratyush", "image_captioning_segmentation", "data", "coco2017", "annotations_trainval2017", "annotations", "captions_train2017.json"
)

# Check if the file exists before loading
if not os.path.isfile(CAPTIONS_PATH):
    raise FileNotFoundError(f"Could not find {CAPTIONS_PATH}. Please check your dataset setup.")

# Load captions
with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
    captions_data = json.load(f)

# Tokenize captions
captions = []
for annot in captions_data.get('annotations', []):
    caption = annot.get('caption')
    if caption is not None:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        captions.append(tokens)
    else:
        print(f"Warning: annotation without 'caption' key: {annot}")

if not captions:
    raise ValueError("No captions found in the dataset.")

print("Example tokenized caption:", captions[0])

# Create vocabulary (basic version)
word_counts = Counter()
for tokens in captions:
    word_counts.update(tokens)

# Remove rare words (e.g., words that appear <5 times)
threshold = 5
words = [word for word, count in word_counts.items() if count >= threshold]
print("Vocabulary size:", len(words))

# Create word2idx and idx2word
import pickle

special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
words = special_tokens + [w for w in words if w not in special_tokens]

word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for word, idx in word2idx.items()}

print("word2idx sample:", list(word2idx.items())[:10])

with open('word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)
with open('idx2word.pkl', 'wb') as f:
    pickle.dump(idx2word, f)
print("Vocabulary dictionaries saved to word2idx.pkl and idx2word.pkl")



