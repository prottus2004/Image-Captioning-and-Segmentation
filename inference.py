import torch
from torchvision import transforms
from PIL import Image
import pickle
import math

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

# Load vocabulary
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

SPECIAL_TOKENS = {'<start>', '<end>', '<pad>', '<unk>'}
ARTICLES_PREPS = {'a', 'an', 'the', 'of', 'with', 'to', 'at', 'in', 'on', 'for', 'by'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

# Load models
encoder = EncoderCNN(embed_size=256).to(device)
encoder.load_state_dict(torch.load('encoder_best_finetuned5.pth', map_location=device))
encoder.eval()

decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx)).to(device)
decoder.load_state_dict(torch.load('decoder_best_finetuned5.pth', map_location=device))
decoder.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_consecutive_duplicates(words):
    result = []
    prev = None
    for w in words:
        if w != prev:
            result.append(w)
        prev = w
    return result

def block_ngram_repetition(words, n=2):
    if len(words) < n:
        return words
    result = []
    ngrams = set()
    for i in range(len(words)):
        if i < n-1:
            result.append(words[i])
            continue
        ngram = tuple(words[i-n+1:i+1])
        if ngram in ngrams:
            continue
        ngrams.add(ngram)
        result.append(words[i])
    return result

def clean_caption(words):
    words = [w for w in words if w not in SPECIAL_TOKENS]
    words = remove_consecutive_duplicates(words)
    words = block_ngram_repetition(words, n=2)
    while words and words[-1] in ARTICLES_PREPS:
        words = words[:-1]
    if not words:
        return "No caption generated."
    caption = ' '.join(words).strip()
    caption = caption[0].upper() + caption[1:]
    if not caption.endswith('.'):
        caption += '.'
    return caption

def diverse_beam_search_decoder(
    features, decoder, word2idx, idx2word, device,
    beam_width=5, max_len=20, length_norm_alpha=0.7,
    num_groups=2, diversity_strength=0.5
):
    """
    Diverse Beam Search with length normalization.
    """
    k = beam_width
    num_groups = min(num_groups, k)
    group_sizes = [k // num_groups] * num_groups
    for i in range(k % num_groups):
        group_sizes[i] += 1

    # Initialize one beam per group
    sequences = []
    for group_id in range(num_groups):
        for _ in range(group_sizes[group_id]):
            sequences.append([[], 0.0, torch.tensor([[word2idx['<start>']]], device=device), None, [], group_id])
    finished = []

    for step in range(max_len):
        all_candidates = []
        group_tokens = [set() for _ in range(num_groups)]
        for i, (seq, score, input_word, states, group_history, group_id) in enumerate(sequences):
            if seq and seq[-1] == '<end>':
                # Don't expand finished sequences
                all_candidates.append((seq, score, input_word, states, group_history, group_id))
                continue
            if states is None:
                hiddens, states_ = decoder.lstm(features.unsqueeze(1), states)
            else:
                embeddings = decoder.embed(input_word)
                hiddens, states_ = decoder.lstm(embeddings, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            log_probs = torch.log_softmax(outputs, dim=1)
            # Penalize tokens that appear in this group's history
            for tok in group_tokens[group_id]:
                idx = word2idx.get(tok, None)
                if idx is not None:
                    log_probs[0, idx] -= diversity_strength
            # Top candidates for this beam
            top_log_probs, top_indices = log_probs.topk(group_sizes[group_id])
            for j in range(group_sizes[group_id]):
                idx = top_indices[0, j].item()
                word = idx2word.get(idx, '<unk>')
                new_seq = seq + [word]
                new_score = score + top_log_probs[0, j].item()
                new_group_history = group_history + [word]
                all_candidates.append((new_seq, new_score, torch.tensor([[idx]], device=device), states_, new_group_history, group_id))
                group_tokens[group_id].add(word)

        # For each group, keep best beams
        new_sequences = []
        for group_id in range(num_groups):
            group_beams = [c for c in all_candidates if c[-1] == group_id]
            # Use length normalization for sorting
            group_beams = sorted(group_beams, key=lambda tup: tup[1] / math.pow(len(tup[0]), length_norm_alpha), reverse=True)
            new_sequences.extend(group_beams[:group_sizes[group_id]])
        sequences = new_sequences

        # Early stop if all sequences ended
        if all(seq and seq[-1] == '<end>' for seq, _, _, _, _, _ in sequences):
            break

    # Select best finished sequence (with length normalization)
    finished = [s for s in sequences if s[0] and s[0][-1] == '<end>']
    if finished:
        best_seq = max(finished, key=lambda tup: tup[1] / math.pow(len(tup[0]), length_norm_alpha))[0]
    else:
        best_seq = sequences[0][0]
    if '<end>' in best_seq:
        best_seq = best_seq[:best_seq.index('<end>')]
    return best_seq

def generate_caption(
    image_path, 
    max_len=20, 
    beam_width=5, 
    length_norm_alpha=0.7, 
    num_groups=2, 
    diversity_strength=0.5
):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)
        caption_words = diverse_beam_search_decoder(
            features, decoder, word2idx, idx2word, device,
            beam_width=beam_width,
            max_len=max_len,
            length_norm_alpha=length_norm_alpha,
            num_groups=num_groups,
            diversity_strength=diversity_strength
        )
    caption = clean_caption(caption_words)
    return caption

# Example usage:
# print(generate_caption('your_image.jpg', beam_width=7, num_groups=3, diversity_strength=0.7))
