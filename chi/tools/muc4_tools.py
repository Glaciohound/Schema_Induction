import itertools


def all_sentences(corpora):
    output = list(itertools.chain(
        *list(itertools.chain(
            *list(
                itertools.chain(_paragraph_split)
                for _paragraph_split in _article["content-cased-split"]
            )
        ) for _article in corpora)
    ))
    return output
