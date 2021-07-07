from transformers import \
    BertTokenizer, BertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM
import torch
import copy


def LM_prompt(text, tokenizer, maskedLM, model_name, top_k=7):
    text = f" {text} "
    custom_masks = ['_', '[mask]']
    for custom_mask in custom_masks:
        while f" {custom_mask} " in text:
            text = text.replace(
                f" {custom_mask} ", f" {tokenizer.mask_token} "
            )
    text = text[1:-1]
    if tokenizer.mask_token not in text:
        return [], text, "accumulating"
    if 'uncased' in model_name:
        text = text.lower().replace(
            tokenizer.mask_token.lower(), tokenizer.mask_token
        )
    ids = torch.tensor([tokenizer.encode(
        text, truncation=True, max_length=512
    )]).long()
    mask_pos = torch.nonzero(ids == tokenizer.mask_token_id)
    ids = ids.cuda()
    with torch.no_grad():
        predictions = maskedLM(ids)[0]

    predicted_tokens = []
    sample_output = copy.copy(ids)
    for item in mask_pos:
        top_k_preds = torch.topk(predictions[item[0], item[1]], k=top_k)[1]
        predicted_tokens.append([
            tokenizer.convert_ids_to_tokens(idx.item())
            for idx in top_k_preds
        ])
        sample_output[item[0], item[1]] = top_k_preds[0]

    sample_output = tokenizer.decode(sample_output[0])
    return predicted_tokens, sample_output, "success"


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
