import os

from dcan.data.partial_loes_scores import get_partial_loes_scores

loes_scoring_folder = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring'
partial_scores_file_path = os.path.join(loes_scoring_folder, '9_7 MRI sessions Igor Loes score updated.csv')
partial_loes_scores = get_partial_loes_scores(partial_scores_file_path)

def get_items(test_dict, lvl):
    # querying for lowest level
    if lvl == 0:
        yield from ((key, val) for key, val in test_dict.items()
                    if not isinstance(val, dict))
    else:
        # recur for inner dictionaries
        yield from ((key1, val1) for val in test_dict.values()
                    if isinstance(val, dict) for key1, val1 in get_items(val, lvl - 1))

# initializing K
K = 1

# calling function
res = get_items(partial_loes_scores, K)
# printing result
for score in dict(res).values():
    print(score.loes_score)
