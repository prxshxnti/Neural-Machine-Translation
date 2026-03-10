import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter
import math
from constants import train_path, val_path, test_path, loss_acc_plot_path

BATCH_SIZE = 32

vocab_en = {'<pad>' : 0, '<unk>' : 1}
vocab_de = {'<pad>' : 0, '<unk>' : 1, '<sos>' : 2, '<eos>' :3}

def load_data():
    # train_df = load_dataset("bentrevett/multi30k", split="train").to_pandas()
    # train_df[:10000].to_csv(train_path)
    train_df = pd.read_csv(train_path)
    print(train_df.shape)

    build_vocabulary(list(train_df['en']), is_english=True)
    build_vocabulary(list(train_df['de']), is_english=False)
    print(f"En Vocab size {len(vocab_en)} and De vocab size {len(vocab_de)}")

    id_to_en = {v:k for k, v in vocab_en.items()}
    id_to_de = {v:k for k,v in vocab_de.items()}

    # val_df = load_dataset("bentrevett/multi30k", split="validation").to_pandas()
    # val_df[:500].to_csv(val_path)
    val_df = pd.read_csv(val_path)
    print(val_df.shape)

    # test_df = load_dataset("bentrevett/multi30k", split="test").to_pandas()
    # test_df[:500].to_csv(test_path)
    test_df = pd.read_csv(test_path)
    print(test_df.shape)

    return train_df, val_df, test_df, id_to_en, id_to_de

def build_vocabulary(data, is_english):
    for sentence in data:
        words = sentence.split()
        for word in words:
            if is_english:
                if word not in vocab_en:
                    vocab_en[word] = len(vocab_en)
            else:
                if word not in vocab_de:
                    vocab_de[word] = len(vocab_de)

class NMTDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        return self.df.iloc[item]["en"], self.df.iloc[item]["de"]

def tokenize(text: str) -> list[str]:
    return text.split()

def numericalize(tokens, is_eng=True):
    if is_eng:
        return [vocab_en.get(token, vocab_en.get('<unk>')) for token in tokens]
    return [vocab_de.get(token, vocab_de.get('<unk>')) for token in tokens]

def collate_fn(batch : list[tuple[str]]):
    eng_sentences, de_sentences = zip(*batch)
    assert len(eng_sentences) == len(de_sentences)

    encoder_sentences = [tokenize(sentence) for sentence in eng_sentences]
    decoder_sentences = [tokenize(sentence) for sentence in de_sentences]

    decoder_ip = [['<sos>'] + tokens for tokens in decoder_sentences]
    decoder_targets = [tokens + ['<eos>'] for tokens in decoder_sentences]

    enc_ip_lengths = [len(tokens) for tokens in encoder_sentences]
    dec_ip_lengths = [len(tokens) for tokens in decoder_ip]

    encoder_ip = [numericalize(tokens) for tokens in encoder_sentences]
    decoder_ip = [numericalize(tokens, is_eng=False) for tokens in decoder_ip]
    decoder_targets = [numericalize(tokens, is_eng=False) for tokens in decoder_targets]

    en_max_len = max([len(tokens) for tokens in encoder_ip])
    de_max_len = max([len(tokens) for tokens in decoder_ip])

    encoder_ip = [tokens + ( [vocab_en['<pad>']] * (en_max_len - len(tokens))) for tokens in encoder_ip]
    decoder_ip = [tokens + ( [vocab_de['<pad>']] * (de_max_len - len(tokens))) for tokens in decoder_ip]
    decoder_targets = [tokens + ( [vocab_de['<pad>']] * (de_max_len - len(tokens))) for tokens in decoder_targets]

    return (torch.tensor(encoder_ip), torch.tensor(decoder_ip), torch.tensor(decoder_targets),
            torch.tensor(enc_ip_lengths), torch.tensor(dec_ip_lengths))

def decode(sample , vocab_map):
    return [vocab_map.get(token.item()) for token in sample]

#Encoder - Decoder Architecture using RNN and context vector - Version 0
#This architecture has encoder to process the english text and provide a context vector as a summarised representation of
#the source sentence and this context vector is passed to the decoder to translate to target sentence
class TranslationV0(nn.Module):
    def __init__(self, encoder_emb_dim, decoder_emb_dim):
        super().__init__()
        #Encoder
        self.encoder_embeddings = nn.Embedding(len(vocab_en), encoder_emb_dim)
        self.encoder = nn.RNN(input_size=encoder_emb_dim, hidden_size=10, num_layers=1, batch_first=True)
        #Decoder
        self.decoder_embeddings = nn.Embedding(len(vocab_de), decoder_emb_dim)
        self.decoder = nn.RNN(input_size=decoder_emb_dim, hidden_size=10, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(in_features=10, out_features=len(vocab_de))

    def forward(self, encoder_inp, decoder_inp, enc_ip_lengths, dec_ip_lengths):
        #enc_inp : (B, S), (B, S)
        encoder_emb = self.encoder_embeddings(encoder_inp) #B, S, emb_dim
        packed_encoder = pack_padded_sequence(encoder_emb, enc_ip_lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.encoder(packed_encoder) #(B, S, hidden_size), (L, B, hidden_size)

        decoder_emb = self.decoder_embeddings(decoder_inp)#B, S, emb_dim
        packed_decoder = pack_padded_sequence(decoder_emb, dec_ip_lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.decoder(packed_decoder, hidden)  #(B, S, hidden_size), (L, B, hidden_size)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=decoder_inp.size(1))
        logits = self.output_layer(out) #(B, S, vocab_size)
        return logits

# BLEU evaluation metric in machine translation and text generation tasks.
# BLEU measures how similar a predicted sentence is to a reference sentence using n-gram overlap.

def bleu_score(ref, pred, max_n=4):
    def find_ngrams(tokens, n):
        """
        tokens = ["I", "love", "NLP"] for n = 2  returns [("I","love"), ("love","NLP")]
        """
        return [tuple(tokens[i : i+n]) for i in range(0, len(tokens)-n+1)]

    ref_tokens = ref.split()
    pred_tokens = pred.split()

    precision = [] # To hold the [p1, p2, p3, p4] where p1 is unigram precision, p2 is bigram precision etc....
    for i in range(1, max_n+1):
        ref_ngrams = find_ngrams(ref_tokens, i)
        pred_ngrams = find_ngrams(pred_tokens, i)

        #Count n grams
        ref_counts = Counter(ref_ngrams)
        pred_counts = Counter(pred_ngrams)

        #Find overlap
        overlap = ref_counts & pred_counts #returns minimum counts(Counter object) between both Counters.

        #Calculate precision
        matched = sum(overlap.values())
        total = sum(pred_counts.values())
        precision.append(matched / total if total > 0 else 0)

    #Brevity penalty - to penalize short sentences
    if len(pred_tokens) > len(ref_tokens):
        bp = 1
    else:
        bp = math.exp(1 - (len(ref_tokens)/len(pred_tokens)))

    # BLEU uses geometric mean of Precisions, instead of average and logs to avoid numerical overflow
    score = sum([math.log(p+1e-9) for p in precision]) / max_n
    bleu = bp * math.exp(score)
    # 0 → completely wrong
    # 1 → perfect match
    return bleu

def truncate_preds(tokens):
    eos_token = "<eos>"
    if eos_token in tokens:
        return tokens[:tokens.index(eos_token)]
    return tokens

def blue_wrapper(logits, y_batch_target, lengths, vocab_map):
    preds = torch.argmax(logits, dim=-1)
    B = preds.shape[0]
    scores = []
    for b in range(B):
        predicted, ref, ref_len = preds[b, :], y_batch_target[b, :], lengths[b]
        predicted, ref = decode(predicted, vocab_map), decode(ref, vocab_map)

        predicted = truncate_preds(predicted)
        ref = ref[:ref_len.item() - 1]

        predicted = " ".join(predicted)
        ref = " ".join(ref)

        score = bleu_score(ref, predicted)
        scores.append(score)

    return sum(scores) / len(scores)

def visualize(train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, NUM_EPOCHS):
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))

    axes[0].plot(epochs_range, train_loss_epoch, label = "Train Loss")
    axes[0].plot(epochs_range, val_loss_epoch, label = "Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Epochs vs Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, train_acc_epoch, label = "Train Accuracy")
    axes[1].plot(epochs_range, val_acc_epoch, label = "Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Epochs vs Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(loss_acc_plot_path)
    plt.show()

def inference(model, vocab_map):
    def preprocess(en_text, de_text):
        tokens_en = en_text.split()
        tokens_de = de_text.split()

        en_len, de_len = len(tokens_en), len(tokens_de)

        tokens_en = numericalize(tokens_en, is_eng=True)
        tokens_de = numericalize(tokens_de, is_eng=False)

        return torch.tensor([tokens_en]), torch.tensor([tokens_de]), torch.tensor([en_len]), torch.tensor([de_len])

    english_text = "A man is standing there"
    german_text = "<sos>"
    max_len = 10

    while len(german_text.split()) < max_len:
        enc_ip_, dec_ip_, enc_len_, dec_len_ = preprocess(english_text, german_text)
        logits_ = model(enc_ip_, dec_ip_, enc_len_, dec_len_) #B, S, Vocab size
        y_pred_ = torch.argmax(logits_, dim=-1)

        next_word = vocab_map[y_pred_[0][-1].squeeze().item()]
        german_text += " "
        german_text += next_word

        if next_word == "<eos>":
            break

    print(f"English text : {english_text}, Predicted German text : {german_text}")

if __name__ == "__main__":
    torch.manual_seed(22)

    train_df, val_df, test_df, id_to_en, id_to_de = load_data()
    train_dataset, val_dataset, test_dataset = NMTDataset(train_df), NMTDataset(val_df), NMTDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder_inp_batch, decoder_inp_batch, decoder_target_batch, enc_len, dec_len = next(iter(test_loader))
    print(encoder_inp_batch.shape, decoder_inp_batch.shape, decoder_target_batch.shape)
    enc_ip, dec_ip, dec_tar = decode(encoder_inp_batch[0], id_to_en), decode(decoder_inp_batch[0], id_to_de), decode(decoder_target_batch[0], id_to_de)
    print(f"English Sample : {" ".join(enc_ip)}")
    print(f"German Sample : {" ".join(dec_tar)}")

    model = TranslationV0(128, 128)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logits_sample = model(encoder_inp_batch[0:1], decoder_inp_batch[0:1], enc_len[0:1], dec_len[0:1]) #1, S, V
    y_preds_sample = torch.argmax(logits_sample, dim=-1).squeeze() #1, S -> S,
    print(f"Model's Predictions Before training {" ".join(decode(y_preds_sample, id_to_de))}")

    NUM_EPOCHS = 10
    train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch = [], [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, val_loss, train_acc, val_acc = [], [], [], []
        model.train()
        for X_batch, y_batch_ip, y_batch_tar, enc_lengths, dec_lengths in train_loader:
            logits = model(X_batch, y_batch_ip, enc_lengths, dec_lengths)
            B, S, V = logits.shape
            loss = loss_fn(logits.view(-1, V), y_batch_tar.view(-1))
            acc = blue_wrapper(logits, y_batch_tar, dec_lengths, id_to_de)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
            train_acc.append(acc)

        model.eval()
        with torch.inference_mode():
            for X_batch, y_batch_ip, y_batch_tar, enc_lengths, dec_lengths in val_loader:
                logits = model(X_batch, y_batch_ip, enc_lengths, dec_lengths)
                B, S, V = logits.shape
                loss = loss_fn(logits.view(-1, V), y_batch_tar.view(-1))
                acc = blue_wrapper(logits, y_batch_tar, dec_lengths, id_to_de)

                val_loss.append(loss.detach().item())
                val_acc.append(acc)

        train_loss_epoch.append((sum(train_loss) / len(train_loss)))
        val_loss_epoch.append((sum(val_loss) / len(val_loss)))
        train_acc_epoch.append((sum(train_acc)/ len(train_acc)))
        val_acc_epoch.append((sum(val_acc)/len(val_acc)))

        print(f"Epoch {epoch} / {NUM_EPOCHS} : Training Loss {train_loss_epoch[-1]:.3f} : "
              f"Validation Loss {val_loss_epoch[-1]:.3f} : "
              f"Training Accuracy : {train_acc_epoch[-1]:.3f} : Validation Accuracy : {val_acc_epoch[-1]:.3f}")

    test_loss, test_acc = [], []
    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch_ip, y_batch_tar, enc_lengths, dec_lengths in test_loader:
            logits = model(X_batch, y_batch_ip, enc_lengths, dec_lengths)
            B, S, V = logits.shape
            loss = loss_fn(logits.view(-1, V), y_batch_tar.view(-1))
            acc = blue_wrapper(logits, y_batch_tar, dec_lengths, id_to_de)

            test_loss.append(loss.detach().item())
            test_acc.append(acc)

    print(f"Mean Test Loss {(sum(test_loss) / len(test_loss)):.3f},"
          f" Mean Test Accuracy {(sum(test_acc)/len(test_acc)):.3f} ")

    logits_sample = model(encoder_inp_batch[0:1], decoder_inp_batch[0:1], enc_len[0:1], dec_len[0:1])
    y_preds_sample = torch.argmax(logits_sample, dim=-1).squeeze()
    print(f"Model's Predictions After training {" ".join(decode(y_preds_sample, id_to_de))}")

    inference(model, id_to_de)

    visualize(train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, NUM_EPOCHS)