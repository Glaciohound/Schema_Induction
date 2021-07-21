import argparse
import numpy as np
import json
from collections import Counter


from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, merge_ranked_list, get_all_type_sentences, \
    rebalance_by_weight, calculate_precision_recall, random_choice, \
    load_selected_names
from components.constants import all_types


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--raw-output",
                        default="data/muc34/outputs/all-prompt-results.json")
    parser.add_argument("--selected-names-file",
                        default="data/muc34/outputs/selected-names.json")
    parser.add_argument("--retrieval-output", type=str,
                        default="data/muc34/outputs/retrieval-output.json")
    args = parser.parse_args()
    return args


def retrieve_by_type(events, corpora, args):
    selected_names = load_selected_names(
        args.selected_names_file, all_types)
    with open(args.raw_output, 'r') as f:
        prompted_lists = json.load(f)

    all_selected_names_index = {
        _fine_grained: _name
        for _name, _group in selected_names.items()
        for _fine_grained in _group["lemma_names"]
    }

    all_sentences = np.array(get_all_sentences(corpora))
    num_sentences = len(all_sentences)
    type_sentences = {
        _type: list(set([
            _sentence[0]
            for _sentences in _group
            for _sentence in _sentences[:2]
        ]))
        for _type, _group in
        get_all_type_sentences(events, corpora).items()
    }
    type_gt_indexes = {
        _type: np.array([
            np.where(all_sentences == _sentence)[0][0]
            for _sentence in _sentences
        ])
        for _type, _sentences in type_sentences.items()
    }

    prompted_lists = np.array(list(map(
        lambda _i: rebalance_by_weight(
            merge_ranked_list(prompted_lists[2*_i: 2*_i+2]),
            selected_names, all_selected_names_index
        ),
        range(num_sentences)
    )))
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

    with open(args.retrieval_output, 'w') as f:
        json.dump((
            false_pos_neg_samples,
            dict(sorted(Counter(prompted_top_names).items(),
                        key=lambda x: x[1], reverse=True)),
            precision_recall,
        ), f, indent=2)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()
    retrieve_by_type(dev_events, dev_corpora, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
