import numpy as np
import json
from collections import Counter, defaultdict

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, get_all_paragraphs, get_all_type_contents, \
    load_selected_names
from components.constants import all_event_types
from components.logic_tools import \
    merge_ranked_list, calculate_precision_recall
from components.get_args import get_args
from components.logging import logger


def retrieve_by_type(events, corpora, args):
    selected_names, all_selected_names_index = load_selected_names(
        args.top_names_file, all_event_types)

    if args.event_element == "sentence":
        all_sentences = get_all_sentences(corpora)
        all_prompt_results_file = args.prompt_all_sentences_results
        retrieval_output = args.sentence_retrieval_output
    elif args.event_element == "paragraph":
        all_sentences = get_all_paragraphs(corpora)
        all_prompt_results_file = args.prompt_all_paragraphs_results
        retrieval_output = args.paragraph_retrieval_output
    elif args.event_element == "paragraph-split":
        all_sentences = get_all_paragraphs(corpora)
        all_prompt_results_file = args.prompt_all_paragraphs_split_results
        retrieval_output = args.paragraph_split_retrieval_output
    else:
        raise NotImplementedError()
    all_sentences_keys = list(all_sentences.keys())
    all_sentences_values = np.array(list(all_sentences.values()))
    num_sentences = len(all_sentences)
    with open(all_prompt_results_file, 'r') as f:
        prompted_lists = json.load(f)

    type_sentences = get_all_type_contents(
        events, corpora, args.event_element,
        True, args.num_contents_each_event)
    type_gt_indices = {
        _type: np.array([
            all_sentences_keys.index(_id[:2])
            for _id in _sentences.keys()
        ])
        for _type, _sentences in type_sentences.items()
    }
    indices_to_gt_type = np.array(["null"] * num_sentences, dtype="<U20")
    for _type, _indices in reversed(type_gt_indices.items()):
        indices_to_gt_type[_indices] = _type

    prompted_lists = np.array(
        list(map(lambda x: merge_ranked_list(x, args.merge_single_policy),
                 prompted_lists)),
        dtype=object
    )
    prompted_top_names = np.array([
        all_selected_names_index.get(
            list(_list.keys())[0], None)
        for _list in prompted_lists
    ])
    retrieved_indices = {
        _type: np.where(prompted_top_names == _keyword[0])[0]
        for _type, _keyword in all_event_types.items()
    }
    precision_recall = calculate_precision_recall(
        retrieved_indices, type_gt_indices)
    precision_recall_dup = calculate_precision_recall(
        retrieved_indices, type_gt_indices, True)

    false_pos_neg_samples = defaultdict(lambda: {})
    for _type, _indices in retrieved_indices.items():
        hit_indices = np.intersect1d(type_gt_indices[_type], _indices)
        fp_indices = np.setdiff1d(_indices, type_gt_indices[_type])
        fn_indices = np.setdiff1d(type_gt_indices[_type], _indices)
        false_pos_neg_samples[_type]["hit"] = list(zip(
            all_sentences_values[hit_indices],
            indices_to_gt_type[hit_indices],
            prompted_lists[hit_indices],
        ))
        false_pos_neg_samples[_type]["false-positive"] = list(zip(
            all_sentences_values[fp_indices],
            indices_to_gt_type[hit_indices],
            prompted_lists[fp_indices],
        ))
        false_pos_neg_samples[_type]["false-negative"] = list(zip(
            all_sentences_values[fn_indices],
            indices_to_gt_type[hit_indices],
            prompted_lists[fn_indices],
        ))

    with open(args.retrieval_diagnosis, 'w') as f:
        json.dump({
            "positive negative samples": false_pos_neg_samples,
            "counter by type": dict(sorted(Counter(prompted_top_names).items(),
                                           key=lambda x: x[1], reverse=True)),
            "strict precision-recall": precision_recall,
            "duplicated precision-recall": precision_recall_dup,
        }, f, indent=2)
    with open(retrieval_output, 'w') as f:
        json.dump({_type: _indices.tolist() for _type, _indices in
                   retrieved_indices.items()},
                  f)
    logger.info(precision_recall)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args=args)
    retrieve_by_type(dev_events, dev_corpora, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
