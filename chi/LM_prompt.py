import torch
from torch.nn.utils.rnn import pad_sequence
import copy
import itertools
from transformers import \
    BertTokenizer, BertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM


def LM_prompt(texts, tokenizer, maskedLM, model_name, top_k=7, batch_size=64):
    is_list = True
    if isinstance(texts, str):
        texts = [texts]
        is_list = False

    def preprocess_text(_text):
        _text = f" {_text} "
        custom_masks = ['_', '[mask]']
        for custom_mask in custom_masks:
            for _i in range(2):
                _text = _text.replace(
                    f" {custom_mask} ", f" {tokenizer.mask_token} "
                )
        _text = _text[1:-1]
        if 'uncased' in model_name:
            _text = _text.lower().replace(
                tokenizer.mask_token.lower(), tokenizer.mask_token
            )
        status = "success" if tokenizer.mask_token in _text else "accumulating"
        return _text, status

    def predict_batch(batch):
        ids = pad_sequence(batch, True, tokenizer.pad_token_id)
        mask_pos = torch.nonzero(ids == tokenizer.mask_token_id)
        ids = ids.cuda()
        with torch.no_grad():
            predictions = maskedLM(ids)[0]

        predicted_tokens = [[] for i in range(len(batch))]
        sample_output = copy.copy(ids)
        for item in mask_pos:
            top_k_preds = torch.topk(predictions[item[0], item[1]], k=top_k)[1]
            predicted_tokens[item[0]].append([
                tokenizer.convert_ids_to_tokens(idx.item())
                for idx in top_k_preds
            ])
            sample_output[item[0], item[1]] = top_k_preds[0]
        sample_output = map(tokenizer.decode, sample_output)
        return zip(predicted_tokens, sample_output)

    texts = list(map(preprocess_text, texts))
    valid_sentences = list(map(
        lambda x: torch.tensor(tokenizer.encode(
            x[0], truncation=True, max_length=512
        )).long(),
        filter(lambda x: x[1] == "success", texts)
    ))
    valid_sentences = [
        valid_sentences[i*batch_size: (i+1)*batch_size]
        for i in range((len(valid_sentences)+1)//batch_size)
    ]
    valid_predictions = itertools.chain(map(predict_batch, valid_sentences))
    output = []
    for _text in texts:
        if _text[1] == "accumulating":
            output.append(([], _text[0], _text[1]))
        else:
            output.append(valid_predictions.pop(0) + ("success",))

    if is_list:
        return output
    else:
        return output[0]


def get_LM(model_name):
    if model_name.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=False)
        maskedLM = BertForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True)
    elif "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_name)
        maskedLM = RobertaForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True)
    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    maskedLM.to(device)
    return tokenizer, maskedLM
