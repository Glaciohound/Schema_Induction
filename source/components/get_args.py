import argparse
from components.logging import getLogger
from pprint import pformat

logger = getLogger("get-args")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/muc34", type=str)
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--loading-cache-file",
                        default="data/muc34/outputs/muc4_loaded_cache.json")
    parser.add_argument("--overwrite-loading", action="store_true")

    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--list-len", type=int, default=10)
    parser.add_argument("--num-selected-names", type=int, default=60)
    parser.add_argument("--select-names-from", type=str, default="paragraph",
                        choices=["sentence", "paragraph"])
    parser.add_argument("--overwrite-prompting-all", action="store_true")
    # parser.add_argument("--top-num-synset-LM", type=int, default=10)

    parser.add_argument("--all-sentences-file", type=str,
                        default="data/muc34/diagnosis/muc4_all_sentences.txt")
    parser.add_argument("--statistics-file", type=str,
                        default="data/muc34/diagnosis/muc4_statistics.txt")
    parser.add_argument(
        "--prompt-all-sentences-results",
        default="data/muc34/outputs/all-sentences-prompt-results.json")
    parser.add_argument(
        "--prompt-all-paragraphs-results",
        default="data/muc34/outputs/all-paragraphs-prompt-results.json")

    parser.add_argument("--top-names-file",
                        default="data/muc34/outputs/top-names-paragraphs.json")
    parser.add_argument("--top-names-by-type-file", type=str,
                        default="data/muc34/diagnosis/top-names-by-type.json")

    args = parser.parse_args()
    logger.info("Parsing Args")
    logger.info(pformat(args.__dict__))
    return args
