import numpy as np
import json
from collections import defaultdict

from components.muc4_tools import \
    corpora_to_dict, load_selected_names
from components.logic_tools import get_head_word
from components.load_muc4 import load_muc4
from components.get_args import get_args
from components.constants import all_arguments_types
from components.logging import logger


def evaluate_extraction(all_arguments, selected_arguments):
    counter = defaultdict(lambda: dict(zip(
        ("gt", "pred", "hit"), np.zeros(3)
    )))

    for _article in all_arguments.values():
        for _element in _article["elements"].values():
            gt_arguments = list(map(
                lambda x: (x[0], all_arguments_types.get(x[2], None)[0],
                           get_head_word(x[1]).lower()),
                _element["gt-arguments"]))
            if "argument-naming" not in _element or\
                    "pred-event-type" not in _element:
                pred_arguments = []
            else:
                pred_arguments = [
                    (_element["pred-event-type"],
                     # reverse_arguments.get(_naming[0], None),
                     selected_arguments.get(_naming[0], None),
                     get_head_word(_ner).lower())
                    for _ner, _naming in _element["argument-naming"].items()
                ]

            # pred_dict = {_item[2]: _item for _item in pred_arguments}
            # if not (pred_arguments == [] and gt_arguments == []):
            #     for _arg in gt_arguments:
            #         print(_arg, pred_dict.get(_arg[2], "missing!"))
            #     input()
            for _argument in gt_arguments:
                counter["|".join(_argument[:2])]["gt"] += 1
                counter["|".join(_argument[:2])]["hit"] += \
                    _argument in pred_arguments
                counter["total"]["gt"] += 1
                counter["total"]["hit"] += _argument in pred_arguments
            for _argument in pred_arguments:
                if _argument[1] is not None:
                    counter["|".join(_argument[:2])]["pred"] += 1
                    counter["total"]["pred"] += 1

    prec_recall = {
        _type: {
            "prec": _values["hit"] / _values["pred"],
            "recall": _values["hit"] / _values["gt"],
            "F1": _values["hit"] / (_values["gt"] + _values["pred"]) * 2,
        }
        for _type, _values in counter.items()
    }
    logger.info(prec_recall)
    with open(args.extraction_results, 'w') as f:
        json.dump((prec_recall, counter), f, indent=2)


def main(args):
    dev_corpora, dev_events, _, _, _ = load_muc4(args=args)
    dev_corpora = corpora_to_dict(dev_corpora)

    with open(args.argument_prompts, 'r') as f:
        all_arguments = json.load(f)
    selected_arguments, selected_arguments_index = load_selected_names(
        args.top_arguments, all_arguments_types
    )

    evaluate_extraction(all_arguments, selected_arguments_index)


if __name__ == "__main__":
    args = get_args()
    main(args)
