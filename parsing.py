import numpy as np
from eisner import Eisner, _mst
import unicodedata
import torch
import argparse
from transformers import BertTokenizer, BertModel


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False


# clean text
def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    # subwords: [cls, sub_tok1, sub_tok2, ..., sep]
    # sentence: [w1, w2, ...]，没有拆分的sentence
    # return 与subwords一样长的list[]，内容是每个subword对应的word的序号
    token_subwords = np.zeros(len(sentence))
    sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]:
            continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token].lower()

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str): len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword

        if not current_token_normalized.startswith(subwords_str):
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)
    return token_ids


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_dep_matrix(args, model, tokenizer, dataset):
    '''
    :param args:
    :param model:
    :param tokenizer:
    :param dataset: [[t11, t12, t13, ...], [t21, t22, t23, ...], ...]
    :return:
    '''
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    model.eval()
    LAYER = 12
    LAYER += 1     # embedding layer
    out = [[] for _ in range(LAYER)]
    for sentence in dataset:
        # tokenized_text = tokenizer.tokenize(' '.join(sentence))
        # tokenized_text.insert(0, '[CLS]')
        # tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # mapping = match_tokenized_to_untokenized(tokenized_text, sentence)
        mapping = []
        indexed_tokens = []
        for idx, word in enumerate(sentence):
            bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            mapping.extend([idx] * len(bpes))  # 记录第idx个word被bpe成几个
            indexed_tokens.extend(bpes)
        
        mapping.append(-1)
        mapping.insert(0, -1)
        indexed_tokens.append(sep_id)
        indexed_tokens.insert(0, cls_id)

        # 1. Generate mask indices
        N = len(indexed_tokens)
        all_layers_matrix_as_list = [[] for _ in range(LAYER)]
        for i in range(0, N):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, N)]
            for j in range(0, N):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                # last_layer = model_outputs[0]
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((N, N))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((sentence, mapping, init_matrix))
    return out


def decoding(args, results):
    trees = []
    decoder = Eisner()

    for (line, mapping, matrix_as_list) in results:
        sentence = line
        init_matrix = matrix_as_list
        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for i, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'sum':
                        new_row.append(sum(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:', sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]

        # add root by heuristic
        root = final_matrix.sum(1).argmax()
        final_matrix[root] = 0
        final_matrix[root, 0] = 1
        final_matrix[0, 0] = 0

        # decoder: 'eisner':
        final_matrix = softmax(final_matrix)
        final_matrix = final_matrix.transpose()

        best_heads, _ = decoder.parse_proj(final_matrix)
        # best_heads = _mst(final_matrix.T)
        trees.append([(i, head) for i, head in enumerate(best_heads) if i != 0])
    return trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1, help='gpu id')
    #parser.add_argument('--cuda', action='store_true', default=False, help='invoke to use gpu')
    parser.add_argument('--decoder', default='eisner')
    parser.add_argument('--root', default='cls', help='gold or cls')
    parser.add_argument('--metric', default='dist', help='metrics for impact calculation, support [L2 dist, cosine] so far')
    parser.add_argument('--subword', default='avg')
    args = parser.parse_args()
    print(args)
    model = BertModel.from_pretrained('../bert/en_base', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('../bert/en_base', do_lower_case=True)

    dataset = ['I come from China'.split(),
               'Welcome to China'.split(),
               'He will go to Beijing tomorrow'.split()]
    results = get_dep_matrix(args, model, tokenizer, dataset)
    trees = decoding(args, results[-2])
    print(trees)

main()
