import numpy as np

from collections import defaultdict
from components.constants import all_event_types
from components.logging import logger


def intersection(set1, set2):
    return len(set(set1).intersection(set(set2)))


def intersect(set1, set2):
    return intersection(set1, set2) != 0


def diff(list1, list2):
    return np.setdiff1d(np.array(list1), np.array(list2)).tolist()


def extend(list1, list2):
    list1.extend(diff(list2, list1))


def sort_rank(rank, key=lambda x: x[1]):
    return dict(sorted(rank.items(), key=key, reverse=True))


def merge_ranked_list(lists, merge_policy="power"):
    full_ranking = defaultdict(lambda: 0)
    for single_list in lists:
        if isinstance(single_list, list) or isinstance(single_list, tuple):
            single_list = dict(zip(single_list[0], single_list[1]))
        for _i, (_word, _score) in enumerate(single_list.items()):
            if merge_policy == "power":
                full_ranking[_word] += _score / (_i+1)
            elif merge_policy == "constant":
                full_ranking[_word] += _score
            elif merge_policy == "max":
                full_ranking[_word] = max(
                    full_ranking[_word], _score
                )
            else:
                raise NotImplementedError()
    full_ranked_list = sort_rank(full_ranking)
    return full_ranked_list


def random_choice(candidates, num, as_array=True):
    if num <= 0 or num >= len(candidates):
        logger.info(f"num={num}, defaulting to original list "
                    f"of length {len(candidates)}")
        if as_array:
            return np.array(candidates)
        else:
            return candidates
    logger.info(f"selecting {num} randomly")
    results = np.random.choice(candidates, min(len(candidates), num), False)
    return results


def calculate_precision_recall(pred_list, gt_list, count_duplicate=False):
    precision_recall = {}
    total_hit = 0
    all_hit = []
    for _type in all_event_types:
        if not count_duplicate:
            retrieved = pred_list[_type]
            gt = gt_list[_type]
            hit = np.intersect1d(retrieved, gt)
        else:
            retrieved = pred_list[_type]
            gt = gt_list[_type]
            hit = list(_item for _item in gt if _item in retrieved)
        precision_recall[_type] = {
            "hit": hit, "pred": retrieved, "gt": gt
        }

    for _type in all_event_types:
        if not count_duplicate:
            retrieved = pred_list[_type]
            gt = gt_list[_type]
            hit = np.intersect1d(retrieved, gt).shape[0]
            retrieved = len(set(retrieved))
            gt = len(set(gt))
        else:
            retrieved = pred_list[_type]
            gt = gt_list[_type]
            hit = sum(_item in retrieved for _item in gt)
            retrieved = retrieved.shape[0]
            gt = gt.shape[0]
        total_hit += hit
        precision_recall[_type] = {
            "precision": hit / retrieved if
            retrieved != 0 else np.inf,
            "recall": hit / gt,
            "F1": hit / (gt + retrieved) * 2,
            "hit": hit,
            "pred": retrieved,
            "gt": gt,
        }
    total_pred = sum(map(len, pred_list.values()))
    total_gt = sum(map(len, gt_list.values()))
    precision_recall["total"] = {
        "precision": total_hit / total_pred,
        "recall": total_hit / total_gt,
        "F1": total_hit / (total_pred + total_gt) * 2,
        "hit": total_hit,
        "pred": total_pred,
        "gt": total_gt,
    }
    return precision_recall


def reverse_dict(original, inner_key=None):
    all_selected_names_index = {
        _fine_grained: _name
        for _name, _group in original.items()
        for _fine_grained in (
            _group[inner_key] if inner_key is not None
            else _group
        )
    }
    return all_selected_names_index


def get_head_word(phrase):
    return phrase.split(" of ")[0].split(" ")[-1]
