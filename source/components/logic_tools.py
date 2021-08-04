import numpy as np

from collections import defaultdict
from components.constants import characters_to_strip
from components.logging import getLogger


def intersect(set1, set2):
    return len(set(set1).intersection(set(set2))) != 0


def diff(list1, list2):
    return np.setdiff1d(np.array(list1), np.array(list2)).tolist()


def extend(list1, list2):
    list1.extend(diff(list2, list1))


def sort_rank(rank, key=lambda x: x[1]):
    return dict(sorted(rank.items(), key=key, reverse=True))


def merge_ranked_list(lists, distribution="power"):
    full_ranking = defaultdict(lambda: 0)
    for single_list in lists:
        for _i, _word in enumerate(single_list):
            if distribution == "power":
                to_add = 1 / (_i + 1)
            elif distribution == "constant":
                to_add = 1
            else:
                raise NotImplementedError()
            full_ranking[_word.strip(characters_to_strip)] += to_add
    full_ranked_list = sort_rank(full_ranking)
    return full_ranked_list


def random_choice(candidates, num):
    logger = getLogger("random-choice")
    if num <= 0 or num >= len(candidates):
        logger.info(f"num={num}, defaulting to original list "
                    f"of length {len(candidates)}")
        return np.array(candidates)
    logger.info(f"selecting {num} randomly")
    return np.random.choice(candidates, min(len(candidates), num), False)
