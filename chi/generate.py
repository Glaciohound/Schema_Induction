from transformers import BertTokenizer, BertForMaskedLM
import argparse
import readline
import os
import atexit
import torch
import copy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--input-file", type=str, default="input.txt")
    parser.add_argument("--output-file", type=str, default="output.txt")
    parser.add_argument("--model-name", type=str, default="bert-large-cased")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--custom-masks", type=str, nargs='*',
                        default=[
                            '_', '*', "-",
                            '[mask]'
                        ])
    args = parser.parse_args()
    return args


def set_readline():
    histfile = os.path.join(os.path.expanduser("~"), ".pyhist")
    if not os.path.exists(histfile):
        open(histfile, 'a').close()
    readline.read_history_file(histfile)
    readline.set_history_length(10000)
    atexit.register(readline.write_history_file, histfile)


def main(args):
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name, do_lower_case=False)
    maskedLM = BertForMaskedLM.from_pretrained(
        args.model_name, output_hidden_states=True)
    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    maskedLM.to(device)

    if args.interactive:

        accumulated = ""
        while True:
            try:
                if accumulated == "":
                    print()
                    print("[[ Input your prompt line ]]")
                    print("----------------------------")
                input_line = input()
                if input_line == "exit":
                    break
                elif input_line == "clear":
                    accumulated = ""
                    continue
                else:
                    if not (accumulated.endswith(' ') or
                            input_line.startswith(' ')):
                        accumulated += ' '
                    predictions, sample_output, status = \
                        LM_prompt(accumulated + input_line,
                                  tokenizer, maskedLM, args)
                if status == "success":
                    for one_prediction in predictions:
                        print(one_prediction)
                    print(sample_output)
                    if accumulated != "":
                        readline.add_history(accumulated + input_line)
                        accumulated = ""
                else:
                    accumulated = sample_output
            except KeyboardInterrupt:
                print("\nTo exit the program, type \"exit\" "
                      "(without the quotation mark).")

    else:
        assert os.path.exists(args.input_file)
        with open(args.input_file, 'r') as f:
            input_lines = f.readlines()
        with open(args.output_file, 'w') as f:
            for input_line in input_lines:
                predictions, sample_output, status = \
                    LM_prompt(input_line, tokenizer, maskedLM, args)
                f.write(input_line+'\n')
                for one_prediction in predictions:
                    f.write(one_prediction+'\n')
                f.write(sample_output+'\n')


def LM_prompt(text, tokenizer, maskedLM, args):
    text = f" {text} "
    for custom_mask in args.custom_masks:
        while f" {custom_mask} " in text:
            text = text.replace(
                f" {custom_mask} ", f" {tokenizer.mask_token} "
            )
    text = text[1:-1]
    if tokenizer.mask_token not in text:
        return [], text, "accumulating"
    if 'uncased' in args.model_name:
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
    for item in mask_pos:
        predicted_tokens.append([
            tokenizer.convert_ids_to_tokens(idx.item())
            for idx in torch.topk(
                predictions[item[0], item[1]], k=args.top_k
            )[1]
        ])
    sample_output = copy.copy(text)
    for i in range(mask_pos.shape[0]):
        sample_output = sample_output.replace(
            tokenizer.mask_token,
            predicted_tokens[i][0],
            1
        )
    return predicted_tokens, sample_output, "success"


if __name__ == "__main__":
    set_readline()
    args = get_args()
    main(args)
