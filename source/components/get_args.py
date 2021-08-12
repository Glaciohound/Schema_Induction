import argparse
from components.logging import logger
from pprint import pformat


def get_args():
    parser = argparse.ArgumentParser()

    """ Model and Data """
    parser.add_argument("--data-dir", default="data/muc34", type=str)
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--max-token-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--loading-cache-file",
                        default="data/muc34/outputs/muc4_loaded_cache.json")
    parser.add_argument("--overwrite-loading", action="store_true")
    parser.add_argument("--overwrite-prompt-cache", action="store_true")

    """ Basic Prompting """
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-selected-events", type=int, default=60)
    parser.add_argument("--event-element", type=str, default="paragraph",
                        choices=["sentence", "paragraph", "paragraph-split"])
    parser.add_argument("--overwrite-prompting-all", action="store_true")
    parser.add_argument("--overwrite-top-events", action="store_true")
    parser.add_argument("--num-contents-each-event", type=int, default=1)
    parser.add_argument("--merge-single-policy", type=str, default="max",
                        choices=["max", "power", "constant"])

    """ Prompting Contents """
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
    parser.add_argument(
        "--prompt-all-paragraphs-split-results",
        default="data/muc34/outputs/all-paragraphs-split-prompt-results.json")

    """ Top Events Types and Event Retrieval """
    parser.add_argument("--top-names-file",
                        default="data/muc34/outputs/top-names.json")
    parser.add_argument("--top-names-by-type-file", type=str,
                        default="data/muc34/diagnosis/top-names-by-type.json")
    parser.add_argument(
        "--retrieval-diagnosis", type=str,
        default="data/muc34/diagnosis/retrieval_diagnosis.json")
    parser.add_argument(
        "--sentence-retrieval-output", type=str,
        default="data/muc34/outputs/sentence-retrieval.json")
    parser.add_argument(
        "--paragraph-retrieval-output", type=str,
        default="data/muc34/outputs/paragraph-retrieval.json")
    parser.add_argument(
        "--paragraph-split-retrieval-output", type=str,
        default="data/muc34/outputs/paragraph-split-retrieval.json")

    """ Argument Naming and Extraction """
    parser.add_argument("--all-arguments", type=str,
                        default="data/muc34/outputs/all-arguments.json")
    parser.add_argument(
        "--argument-prompts", type=str,
        default="data/muc34/outputs/all-argument-prompt-results.json")
    parser.add_argument("--top-arguments", type=str,
                        default="data/muc34/outputs/top-arguments.json")
    parser.add_argument("--num-selected-arguments", type=int, default=30)
    parser.add_argument("--extraction-results", type=str,
                        default="data/muc34/outputs/full-extraction.json")
    parser.add_argument("--argument-list-len", type=int, default=5)
    parser.add_argument("--overwrite-ner", action="store_true")
    parser.add_argument("--overwrite-argument-naming", action="store_true")
    parser.add_argument("--overwrite-argument-filtering", action="store_true")

    args = parser.parse_args()
    logger.info(pformat(args.__dict__))
    return args
