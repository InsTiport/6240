"""
Credit: adapted from https://github.com/arashkhoeini/itemrank
"""

import tqdm
import numpy as np
from ItemRank import ItemRank


def uniquify(a):
    unique_tuples = set()
    for row in a:
        unique_tuples.add(tuple(row))

    return np.array(list(unique_tuples))


def count_matched(l1, l2):
    return len(set(l1) & set(l2))


def count_matched_recall(predicted, actual):
    return len([e for e in actual if e in predicted])


# read train and test data
user_subreddit_rating = dict()
with open('user_subreddit.tsv', 'r') as r:
    line_counter = 0
    for line in tqdm.tqdm(r.readlines()):
        if line_counter >= 1:
            splits = line.split('\t')
            user, subreddit, rating = int(splits[0]), int(splits[1]), float(splits[2])
            user_subreddit_rating[user, subreddit] = rating

        line_counter += 1

all_data_list = []
line_counter = 0
with open('reddit.csv') as r:
    for line in tqdm.tqdm(r.readlines()):
        line_counter += 1

        # exclude header
        if line_counter > 1:
            # get all data attributes
            split = line.split(',')
            user_id, subreddit_id = int(split[0]), int(split[1])

            all_data_list.append([user_id, subreddit_id, user_subreddit_rating[user_id, subreddit_id]])

all_data = np.array(all_data_list)
unique_all_data = uniquify(all_data)

print('Training and evaluating...\n')
for split in range(1):
    print(f'Split {split}:')
    with open(f'split_{split}.txt', 'r') as r:
        training_idx = np.array(r.readline().split(','), dtype=int)
        test_idx = np.array(r.readline().split(','), dtype=int)

    np_data = all_data[training_idx]
    np_test_data = all_data[test_idx]

    unique_np_data = uniquify(np_data)
    unique_np_test_data = uniquify(np_test_data)

    # training and testing the model
    item_rank = ItemRank(unique_np_data, unique_all_data)
    item_rank.generate_graph()
    item_rank.generate_coef_from_graph()
    DOAs = []

    # macro_avg_accuracy_1 = []
    # macro_avg_accuracy_5 = []
    # macro_avg_accuracy_10 = []
    macro_avg_recall_1 = []
    macro_avg_recall_5 = []
    macro_avg_recall_10 = []

    # test_macro_avg_accuracy_1 = []
    # test_macro_avg_accuracy_5 = []
    # test_macro_avg_accuracy_10 = []
    test_macro_avg_recall_1 = []
    test_macro_avg_recall_5 = []
    test_macro_avg_recall_10 = []

    rounds = []
    for user_name in tqdm.tqdm(range(10000)):
        Tu_not_unique = []
        for i in range(len(np_test_data[:, 0])):
            if np_test_data[i, 0] == user_name:
                Tu_not_unique.append(np_test_data[i, 1])
        if len(Tu_not_unique) <= 10:
            continue
        Lu_not_unique = []
        for i in range(len(np_data[:, 0])):
            if np_data[i, 0] == user_name:
                Lu_not_unique.append(np_data[i, 1])
        if len(Lu_not_unique) <= 10:
            continue

        Tu = item_rank.calculate_Tu(unique_np_test_data, user_name)
        Lu = item_rank.calculate_Lu(user_name)

        d = item_rank.generate_d(user_name=user_name)
        # d = d = np.zeros(len(item_rank.movie_names))
        # d[:] = 1 / len(item_rank.movie_names)

        IR = np.ones(len(item_rank.movie_names))
        old_IR = IR
        converged = False
        counter = 0
        while not converged:
            counter += 1
            old_IR = IR
            IR = item_rank.item_rank(0.85, IR, d)
            converged = (old_IR - IR < 0.0001).all()
        # print(f'Converged after {str(counter)} counts.')
        rounds.append(counter)
        doa = item_rank.calculate_DOA(unique_np_test_data, user_name, IR)
        DOAs.append(doa)

        predicted_interested_subreddits_1 = item_rank.get_most_similar(IR, [], 1)
        predicted_interested_subreddits_5 = item_rank.get_most_similar(IR, [], 5)
        predicted_interested_subreddits_10 = item_rank.get_most_similar(IR, [], 10)

        # num_matched_1 = count_matched(Tu, predicted_interested_subreddits_1)
        # num_matched_5 = count_matched(Tu, predicted_interested_subreddits_5)
        # num_matched_10 = count_matched(Tu, predicted_interested_subreddits_10)
        macro_avg_recall_1.append(count_matched_recall(predicted_interested_subreddits_1, Lu_not_unique)
                                  / len(Lu_not_unique))
        macro_avg_recall_5.append(count_matched_recall(predicted_interested_subreddits_5, Lu_not_unique)
                                  / len(Lu_not_unique))
        macro_avg_recall_10.append(count_matched_recall(predicted_interested_subreddits_10, Lu_not_unique)
                                   / len(Lu_not_unique))
        # macro_avg_accuracy_1.append(num_matched_1 / 1)
        # macro_avg_accuracy_5.append(num_matched_5 / 5)
        # macro_avg_accuracy_10.append(num_matched_10 / 10)

        # num_matched_1 = count_matched(Lu, predicted_interested_subreddits_1)
        # num_matched_5 = count_matched(Lu, predicted_interested_subreddits_5)
        # num_matched_10 = count_matched(Lu, predicted_interested_subreddits_10)
        test_macro_avg_recall_1.append(count_matched_recall(predicted_interested_subreddits_1, Tu_not_unique)
                                       / len(Tu_not_unique))
        test_macro_avg_recall_5.append(count_matched_recall(predicted_interested_subreddits_5, Tu_not_unique)
                                       / len(Tu_not_unique))
        test_macro_avg_recall_10.append(count_matched_recall(predicted_interested_subreddits_10, Tu_not_unique)
                                        / len(Tu_not_unique))
        # macro_avg_accuracy_1.append(num_matched_1 / 1)
        # macro_avg_accuracy_5.append(num_matched_5 / 5)
        # macro_avg_accuracy_10.append(num_matched_10 / 10)

        # print(f'DOA for user {user_name} is : {doa}')
    print(f'Macro DOA for split {split} is: {sum(DOAs) / len(DOAs)}')
    print(f'Converses on average after {np.mean(rounds)} rounds')
    # print(f'Macro average accuracy@1 for split {split} is {np.mean(macro_avg_accuracy_1)}')
    print(f'Macro average recall@1 for the train set of split {split} is {np.mean(macro_avg_recall_1)}')
    print(f'Macro average recall@1 for the test set of split {split} is {np.mean(test_macro_avg_recall_1)}')

    # print(f'Macro average accuracy@5 for split {split} is {np.mean(macro_avg_accuracy_5)}')
    print(f'Macro average recall@5 for the train set of split {split} is {np.mean(macro_avg_recall_5)}')
    print(f'Macro average recall@5 for the test set of split {split} is {np.mean(test_macro_avg_recall_5)}')

    # print(f'Macro average accuracy@10 for split {split} is {np.mean(macro_avg_accuracy_10)}')
    print(f'Macro average recall@10 for the train set of split {split} is {np.mean(macro_avg_recall_10)}')
    print(f'Macro average recall@10 for the test set of split {split} is {np.mean(test_macro_avg_recall_10)}\n')
