import numpy as np
import json
from collections import Counter

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, get_all_paragraphs, get_all_type_contents, \
    load_selected_names
from components.constants import all_event_types
from components.logic_tools import \
    merge_ranked_list, random_choice, calculate_precision_recall
from components.get_args import get_args
from components.logging import getLogger

logger = getLogger("retrieve-events")


def retrieve_by_type(events, corpora, args):
    selected_names, all_selected_names_index = load_selected_names(
        args.top_names_file, all_event_types)

    if args.event_element == "sentence":
        all_sentences = np.array(list(get_all_sentences(corpora).values()))
        all_prompt_results_file = args.prompt_all_sentences_results
        retrieval_output = args.sentence_retrieval_output
    elif args.event_element == "paragraph":
        all_sentences = np.array(list(get_all_paragraphs(corpora).values()))
        all_prompt_results_file = args.prompt_all_paragraphs_results
        retrieval_output = args.paragraph_retrieval_output
    with open(all_prompt_results_file, 'r') as f:
        prompted_lists = json.load(f)

    type_sentences = get_all_type_contents(
        events, corpora, args.event_element,
        True, args.num_contents_each_event)
    type_gt_indexes = {
        _type: np.array([
            np.where(all_sentences == _sentence["sentence"])[0][0]
            for _sentence in _sentences
        ])
        for _type, _sentences in type_sentences.items()
    }

    prompted_lists = list(map(
            merge_ranked_list, prompted_lists
    ))
    prompted_top_names = np.array([
        all_selected_names_index.get(
            list(_list.keys())[0], None)
        for _list in prompted_lists
    ])
    retrieved_indexes = {
        _type: np.where(prompted_top_names == _keyword[0])[0]
        for _type, _keyword in all_types.items()
    }
    precision_recall = calculate_precision_recall(
        retrieved_indexes, type_gt_indexes)
    false_pos_neg_samples = {
        _type: {
            "hit": all_sentences[random_choice(
                np.intersect1d(type_gt_indexes[_type], _indexes),
                args.num_samples
            )].tolist(),
            "false-positive": all_sentences[random_choice(
                np.setdiff1d(_indexes, type_gt_indexes[_type]),
                args.num_samples
            )].tolist(),
            "false-negative": all_sentences[random_choice(
                np.setdiff1d(type_gt_indexes[_type], _indexes),
                args.num_samples
            )].tolist(),
        }
        for _type, _indexes in retrieved_indexes.items()
    }

    with open(args.retrieval_diagnosis, 'w') as f:
        json.dump((
            false_pos_neg_samples,
            dict(sorted(Counter(prompted_top_names).items(),
                        key=lambda x: x[1], reverse=True)),
            precision_recall,
        ), f, indent=2)
    with open(retrieval_output, 'w') as f:
        json.dump({_type: _indices.tolist() for _type, _indices in
                   retrieved_indexes.items()},
                  f)
    logger.info(precision_recall)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args=args)
    retrieve_by_type(dev_events, dev_corpora, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
