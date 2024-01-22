"""""""""""""""""""""""""""""
Project: Face_image_score
Author: Terance Jiang
Date: 1/17/2024
Based on: https://github.com/Tencent/TFace/tree/quality
I full used the broadcast and vectorization feature of numpy and multiprocessing to speed up the calculation 
of wasserstein distance. ~30x faster than the original version.
"""""""""""""""""""""""""""""
import argparse
import itertools
import multiprocessing

from numpy.linalg import norm
import numpy as np
import os
from tqdm import tqdm
import random

from scipy.stats import wasserstein_distance
from wxtools.io_utils import read_txt, replace_root_extension


def read_data_features(args):
    image_list_file = args.image_list_file
    id_key_index = args.id_key_index
    feature_root = args.feature_root
    load_feature = args.load_feature
    image_root = args.image_root

    id_feature_dict = {}

    image_list = read_txt(image_list_file)

    def group_key(x):
        return x.split('/')[id_key_index]

    image_list.sort(key=group_key)
    image_list_group = {k: list(g) for k, g in itertools.groupby(image_list, key=group_key)}

    for person_id, image_list in tqdm(image_list_group.items()):
        if len(image_list) < 2:
            continue

        # absolute path
        if os.path.isabs(image_list[0]):
            feature_list = replace_root_extension(image_list, image_root, feature_root, ['.jpg', '.png'], '.npy')

        # relative path
        else:
            feature_list = replace_root_extension(image_list, None, None, ['.jpg', '.png'], '.npy')
            feature_list = [os.path.join(feature_root, feature_path) for feature_path in feature_list]

        feature_list = list(zip(image_list, feature_list))
        # if load_feature:
        #     image_features = np.array([np.load(image_feature) for image_feature in feature_list])
        # else:
        #     image_features = feature_list

        id_feature_dict[person_id] = feature_list

    return id_feature_dict


def cosine_similarity(features):
    # Normalize each feature vector to unit length
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms

    # Compute cosine similarity, dot product of normalized vectors
    # because they are normalized, cosine similarity is just the dot product
    cosine_sim = np.dot(normalized_features, normalized_features.T)

    # Ensure the diagonal is zero before calculating mean, as we don't want to include self-similarity
    np.fill_diagonal(cosine_sim, 0)

    # Calculate the mean cosine similarity for each feature
    mean_cosine_sim = np.mean(cosine_sim, axis=1)

    return mean_cosine_sim


def get_mean_cosine_similarity(image_names, features):
    mean_cosine_sim = cosine_similarity(features)
    # Map image names to their corresponding mean cosine similarity
    return dict(zip(image_names, mean_cosine_sim))


def calculate_same_people_cos_sim(id_feature_dict, args):

    fix_num = args.fix_num
    global ids
    ids = list(id_feature_dict.keys())  # Convert to list for consistent indexing

    same_sim_score = {}

    print("Calculating same_sim_score")
    sim_pool = multiprocessing.Pool(processes=args.process_num)
    #
    # with multiprocessing.Pool(processes=10) as pool:
    #     results = tqdm(
    #         pool.map(calculate_cosine_similarity, ))

    process_args = [(person_id, id_feature_dict, fix_num, True) for person_id in ids]
    # Process each image in parallel
    results = []
    for _ in tqdm(sim_pool.imap_unordered(calculate_cosine_similarity, process_args), total=len(process_args)):
        results.append(_)

    for res in results:
        same_sim_score.update(res)

    return same_sim_score


def cosine_similarity_n1(v1, v2):
    # Ensure the vectors are normalized
    v1_normalized = v1 / norm(v1)
    v2_normalized = v2 / norm(v2)
    return np.dot(v1_normalized, v2_normalized)



def calculate_cosine_similarity(args):
    person_id, id_feature_dict, fix_num, same_person = args

    # n * 512
    # feature_mat = np.asarray([np.load(image_path[1])])
    feature_mat = []
    for image_path in id_feature_dict[person_id]:
        if not os.path.exists(image_path[1]):
            feature = np.zeros((512,))
            print("feature not found: ", image_path[1])
        else:
            feature = np.load(image_path[1])
        feature_mat.append(feature)
    feature_mat = np.asarray(feature_mat)

    # n * 24 * 512
    feature_mat_3d = np.empty((feature_mat.shape[0], fix_num, feature_mat.shape[1]))

    # for each image in the person_id
    for w in range(feature_mat_3d.shape[0]):
        # if calculating sim between different people, sample fix_num random people
        if not same_person:
            random_ids = random.sample(ids, fix_num)
            if person_id in random_ids:
                random_ids.remove(person_id)
                random_ids.append(random.sample(ids, 1)[0])
        # if calculating sim between same people, sample fix_num random images from the same person
        else:
            random_ids = [person_id for _ in range(fix_num)]

        # for each random person, sample one random image
        random_feature_matrix = np.asarray(
            [np.load(random.sample(id_feature_dict[random_id], 1)[0][1]) for random_id in random_ids])

        feature_mat_3d[w, :, :] = random_feature_matrix

    feature_mat_norm = feature_mat / np.linalg.norm(feature_mat, axis=1, keepdims=True)
    feature_mat_3d_norm = feature_mat_3d / np.linalg.norm(feature_mat_3d, axis=2, keepdims=True)

    cosine_sim = np.sum(feature_mat_norm[:, np.newaxis, :] * feature_mat_3d_norm, axis=2)

    result = {}
    for j, image_path in enumerate([image_path[0] for image_path in id_feature_dict[person_id]]):
        result[image_path] = cosine_sim[j]

    return result


def calculate_diff_people_cos_sim(id_feature_dict, args):
    fix_num = args.fix_num
    global ids
    ids = list(id_feature_dict.keys())  # Convert to list for consistent indexing

    diff_sim_score = {}

    print("Calculating diff_sim_score")
    sim_pool = multiprocessing.Pool(processes=args.process_num)
    #
    # with multiprocessing.Pool(processes=10) as pool:
    #     results = tqdm(
    #         pool.map(calculate_cosine_similarity, ))

    process_args =[(person_id, id_feature_dict, fix_num, False) for person_id in ids]
    # Process each image in parallel
    results = []
    for _ in tqdm(sim_pool.imap_unordered(calculate_cosine_similarity, process_args), total=len(process_args)):
        results.append(_)

    for res in results:
        diff_sim_score.update(res)

    return diff_sim_score


def calculate_w_distance(arg):
    key, pos_sim, neg_sim = arg
    return key, wasserstein_distance(pos_sim, neg_sim)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-list-file",
                        default='', )
    parser.add_argument("--image-root", default='', )
    parser.add_argument("--score-dst", default='', )
    parser.add_argument("--id-key-index", default=-3, help="id_key_index")
    parser.add_argument("--feature-root",
                        default='', )
    parser.add_argument("-load-feature", default=False, help="load_feature")
    parser.add_argument("--process-num", default=20, help="process number")
    parser.add_argument("--fix-num", default=24, help="fix number")
    args = parser.parse_args()
    return args


def norm_labels(wdistance_dict):
    quality_scores = wdistance_dict.values()
    quality_scores = list(quality_scores)
    quality_scores = np.asarray(quality_scores)
    quality_scores = (quality_scores - np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100

    return quality_scores

def check_npy_exist(image_path):
    if not os.path.exists(image_path):
        print("feature not found: ", image_path)
        return False
    return True


if __name__ == "__main__":

    args = parse_args()
    print("load data")
    id_feature_dict = read_data_features(args)

    id_feature_dict_exist = {}
    image_count = 0
    print("check npy exist")
    for key, value in tqdm(id_feature_dict.items()):
        value_exist = []
        for image_path in value:
            if check_npy_exist(image_path[1]):
                value_exist.append(image_path)
        if len(value_exist) > 0:
            image_count += len(value_exist)
            id_feature_dict_exist[key] = value_exist

    id_feature_dict = id_feature_dict_exist

    image_names = []
    quality_scores_metrix = np.zeros((82118, 12))
    for i in range(12):
        pos_similarity_dist = calculate_same_people_cos_sim(id_feature_dict, args)
        neg_similarity_dist = calculate_diff_people_cos_sim(id_feature_dict, args)

        id_keys = pos_similarity_dist.keys()
        process_args = [(key, pos_similarity_dist[key], neg_similarity_dist[key]) for key in id_keys]

        pool = multiprocessing.Pool(processes=args.process_num)

        w_distance = {}
        for img_key, wd in tqdm(pool.imap_unordered(calculate_w_distance, process_args), total=len(process_args)):
            w_distance[img_key] = wd

        pool.close()
        pool.join()

        quality_scores = norm_labels(w_distance)
        quality_scores_metrix[:, i] = quality_scores

        quality_scores_with_name = [f'{name} {quality_scores[idx]}' for idx, name in enumerate(w_distance.keys())]

        with open(os.path.join(args.score_dst, f'score_{i}.txt'), 'w') as f:
            f.writelines(x + '\n' for x in quality_scores_with_name)

    quality_pseudo_labels = np.mean(quality_scores_metrix, axis=1)
    quality_scores_with_name = [f'{name} {quality_pseudo_labels[idx]}' for idx, name in enumerate(w_distance.keys())]

    with open(os.path.join(args.score_dst, 'labels.txt'), 'w') as f:
        f.writelines(x + '\n' for x in quality_scores_with_name)

