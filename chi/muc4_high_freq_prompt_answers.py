import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

from load_muc4 import load_muc4
from tools.muc4_tools import \
    get_all_sentences, extract_relevant_sentences, \
    corpora_to_dict
from LM_prompt import get_LM, LM_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--output-file", default="high-frequency-names.json")
    args = parser.parse_args()
    return args


prompt_sentences = (
    "The reporter witnessed the _ .",
    "This is a typical _ incident.",
)


def prompt_all_sentences(tokenizer, maskedLM,
                         corpora, args):
    all_sentences = get_all_sentences(corpora)
    all_prompt_answers = [
        LM_prompt(
            sentence+' '+prompt_sentence,
            tokenizer, maskedLM,
            args.model_name, args.top_k
        )
        for sentence in
        tqdm(np.random.choice(all_sentences, args.num_samples))
        for prompt_sentence in prompt_sentences
    ]
    ranked_lists = list(map(lambda x: x[0], all_prompt_answers))
    full_ranking = defaultdict(lambda: 0)
    for single_list in ranked_lists:
        for i, word in enumerate(single_list[0]):
            full_ranking[word] += 1 / (i + 1)
    full_ranking = sorted(full_ranking.items(), key=lambda x: x[1],
                          reverse=True)
    with open(args.output_file, 'w') as f:
        json.dump(full_ranking, f)


def prompt_relevant_sentences(tokenizer, maskedLM, corpora, events, args):
    for _event in np.random.choice(events, args.num_samples):
        sentence = extract_relevant_sentences(
            _event, corpora[_event["MESSAGE: ID"]])
        if sentence is not None:
            prompt_list = LM_prompt(
                sentence+" "+prompt_sentences[0],
                tokenizer, maskedLM,
                args.model_name, args.top_k
            )
            print(_event["INCIDENT: TYPE"], prompt_list)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()
    dev_corpora_dict = corpora_to_dict(dev_corpora)

    tokenizer, maskedLM = get_LM(args.model_name)
    prompt_all_sentences(tokenizer, maskedLM, dev_corpora, args)
    tokenizer, maskedLM = None, None
    prompt_relevant_sentences(
        tokenizer, maskedLM,
        dev_corpora_dict, dev_events, args
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
