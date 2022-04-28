import tqdm
import math
import numpy as np

line_count = 0

split_id = 0


'''
Read file
'''
sample_interactions = []
sample_size = math.inf
# item_embeddings = defaultdict(list)
with open('data/reddit.csv') as r:
    for line in tqdm.tqdm(r.readlines()):
        line_count += 1
        # exclude header
        if line_count > 1:
          if line_count <= sample_size:
            # get all data attributes
            split = line.strip().split(',')
            user_id, subreddit_id, timestamp, state_label, features = int(split[0]), int(split[1]), float(split[2]), int(split[3]), ','.join(split[4:])
            # print(features.shape)
            # item_embeddings[subreddit_id].append(features)
            sample_interactions.append([user_id, subreddit_id, state_label, timestamp, features])
          else:
            break

        
'''
Get Split
'''
def get_split(data, train_proportion=0.8, seed=0):

  size = len(data)

  train_size = round(size * train_proportion)
  np.random.seed(seed)
  train_ids = np.random.choice(np.arange(size), train_size, replace=False)
  test_ids = np.setdiff1d(np.arange(size), train_ids)

  return train_ids, test_ids

train_ids, test_ids = get_split(sample_interactions)

def write_file(dataset, ids, filename):
  '''
  Write processed data to tsv
  '''
  session_counter = 0
  with open(filename, 'w') as w:
    w.write('user_id\titem_id\tinteraction_type\tcreated_at\tfeatures\n')
    for i in ids:
      (user_id, subreddit_id, state_label, timestamp, features) = dataset[i]
      w.write(str(int(user_id)))
      w.write('\t')
      w.write(str(int(subreddit_id)))
      w.write('\t')
      w.write(str(int(state_label)))
      w.write('\t')
      w.write(str(int(timestamp)))
      w.write('\t')
      w.write(str(features))
      w.write('\n')

      session_counter += 1

    print(f'There are {session_counter} sessions.')

write_file(sample_interactions, train_ids, f'data/train_interactions_embed.tsv')
write_file(sample_interactions, test_ids, f'data/test_interactions_embed.tsv')