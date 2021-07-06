import argparse
from tqdm import tqdm

from load_muc4 import load_muc4
from tools.muc4_tools import all_sentences
from LM_prompt import get_LM, LM_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="bert-large-cased")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    return args


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()
    dev_all_sentences = all_sentences(dev_corpora)

    prompt_sentences = (
        "The reporter witnessed the _ .",
        "This is a typical _ incident.",
    )
    tokenizer, maskedLM = get_LM(args.model_name)
    for sentence in dev_all_sentences[:10]:
        print(LM_prompt(
            sentence+' '+prompt_sentences[0],
            tokenizer, maskedLM,
            args.model_name, args.top_k
        ))


if __name__ == "__main__":
    args = get_args()
    main(args)
