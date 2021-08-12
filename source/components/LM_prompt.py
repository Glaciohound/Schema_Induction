import os
import pickle
import torch
import copy
import itertools
from tqdm import tqdm
from transformers import \
    BertTokenizer, BertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM
from torch.nn.utils.rnn import pad_sequence

from components.constants import characters_to_strip
from components.logging import logger


def LM_prompt(texts, tokenizer, maskedLM,
              model_name=None, strip=False, top_k=7, batch_size=64,
              max_token_length=512, tokens_only=False, one_per_prompt=True,
              overwrite_prompt_cache=False):
    debug_cache_file = "LM-prompt.cache"
    is_list = True
    if model_name is None:
        model_name = maskedLM.name_or_path
    if isinstance(texts, str):
        texts = [texts]
        is_list = False

    logger.info(f"Masked-LM prompting texts of length {len(texts)}")

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
        if torch.cuda.device_count() >= 1:
            ids = ids.cuda()
        with torch.no_grad():
            predictions = maskedLM(ids)[0]

        predicted_tokens = [[] for i in range(len(batch))]
        sample_output = copy.copy(ids)
        for item in mask_pos:
            probs, top_k_preds = \
                torch.topk(predictions[item[0], item[1]], k=top_k)
            this_predicted_tokens = [
                tokenizer.convert_ids_to_tokens(idx.item())
                for idx in top_k_preds
            ]
            if strip:
                this_predicted_tokens = list(map(
                    lambda x: x.strip(characters_to_strip),
                    this_predicted_tokens
                ))
            if one_per_prompt:
                predicted_tokens[item[0]] = \
                    (this_predicted_tokens, probs.cpu().numpy().tolist())
            else:
                predicted_tokens[item[0]].append(
                    (this_predicted_tokens, probs.cpu().numpy().tolist()))
            sample_output[item[0], item[1]] = top_k_preds[0]
        sample_output = map(
            lambda x: tokenizer.decode(x).replace(tokenizer.pad_token, ""),
            sample_output
        )
        return list(zip(predicted_tokens, sample_output))

    texts = list(map(preprocess_text, texts))
    valid_sentences = list(map(
        lambda x: torch.tensor(tokenizer.encode(x[0]))
        .long()[-max_token_length:],
        filter(lambda x: x[1] == "success", texts)
    ))
    valid_sentences = [
        valid_sentences[i*batch_size: (i+1)*batch_size]
        for i in range((len(valid_sentences)-1)//batch_size+1)
    ]

    valid_predictions = None
    if os.path.exists(debug_cache_file) and not overwrite_prompt_cache:
        with open(debug_cache_file, 'rb') as f:
            debug_cache = pickle.load(f)
        if debug_cache[0] == texts:
            valid_predictions = debug_cache[1]
            logger.info(f"loaded LM-prompt cache from {debug_cache_file}")
    if valid_predictions is None:
        logger.info("prompting sentences from scratch")
        valid_predictions = list(itertools.chain(
            *list(map(
                predict_batch,
                tqdm(valid_sentences) if len(valid_sentences) >= 10
                else valid_sentences
            ))
        ))
        with open(debug_cache_file, 'wb') as f:
            pickle.dump((texts, valid_predictions), f)
            logger.info(f"dumping prompt results to {debug_cache_file}")

    output = []
    for _text in texts:
        if _text[1] == "accumulating":
            this_output = [], _text[0], _text[1]
        else:
            this_pred = valid_predictions.pop(0)
            this_output = (this_pred[0], this_pred[1], "success")
        if tokens_only:
            output.append(this_output[0])
        else:
            output.append(this_output)

    if is_list:
        return output
    else:
        return output[0]


def get_LM(model_name):
    logger.info(f"Get tokenizer and model {model_name}")
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
