import os
import json
import numpy
import subprocess
import argparse


def get_metrics(metrics_json):
    try:
        with open(metrics_json, 'r') as f:
            metrics_dict = json.load(f)
        denotation_acc = metrics_dict["best_validation_denotation_accuracy"]
        consistency = metrics_dict["best_validation_consistency"]
    except:
        print("Error reading: {}".format(metrics_json))
        denotation_acc, consistency = 0.0, 0.0
    return denotation_acc, consistency


def print_iterative_training_metrics(checkpoint_dir, all_max_decoding_steps=[12, 14, 16, 18, 20, 22]):
    """Iteratively train a parer by alternating between MML and ERM parsers.

    The directory structure inside checkpoint root is:
    SEED_S/
        MML/Iter${ITER}_MDS${MDS}/
        PairedERM/Iter${ITER}_MDS${MDS}/
        GenData/train_ERM_Iter${ITER}.json


    Parameters:
    -----------
    full_train_json, full_dev_json: Complete grouped NLVR train/dev data.

    """

    print("\nCheckpoint dir: {}".format(checkpoint_dir))

    # This directory should contain multiple SEED_X directories, each contains MML & ERM directories
    # Each of those contains Iter1_MDS14/ Iter2_MDS16/ Iter3_MDS18/ Iter4_MDS20/ Iter5_MDS22/
    seed_dirs = os.listdir(checkpoint_dir)
    print(seed_dirs)

    total_den, total_con = 0.0, 0.0
    count = 0
    denotations, consistencies = [], []
    for seed_dir in seed_dirs:
        serdir = os.path.join(checkpoint_dir, seed_dir, "ERM", "Iter{}_MDS{}".format(4, 20))
        metrics_json = os.path.join(serdir, "metrics.json")
        if os.path.exists(metrics_json):
            count += 1
            den, con = get_metrics(metrics_json)
            total_den += den
            total_con += con
            denotations.append(den)
            consistencies.append(con)
        else:
            print("Error reading: {}".format(metrics_json))

    print(denotations)
    print(consistencies)

    avg_denotation = numpy.average(denotations)
    std_denotation = numpy.std(denotations)
    avg_consistency = numpy.average(consistencies)
    std_consistency = numpy.std(consistencies)

    print("Average consistency: {}  std: {}".format(avg_consistency, std_consistency))
    print("Average denotation: {}  std: {}".format(avg_denotation, std_denotation))

    exit()

    mml_ckpt_dir = os.path.join(checkpoint_dir, "MML")
    erm_ckpt_dir = os.path.join(checkpoint_dir, "ERM")

    # Print metrics
    iteration = 0
    mds = all_max_decoding_steps[iteration]
    metrics_json = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
    print(metrics_json)
    den, con = get_metrics(metrics_json)
    print("MML Iteration: {}  MDS: {}".format(iteration, mds))
    print("Denotation Acc: {} Consistency: {}".format(den, con))
    for mds in all_max_decoding_steps[1:]:
        iteration += 1
        metrics_json = os.path.join(erm_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        print("\nERM Iteration: {}  MDS: {}".format(iteration, mds))
        den, con = get_metrics(metrics_json)
        print("Denotation Acc: {} Consistency: {}".format(den, con))

        metrics_json = os.path.join(mml_ckpt_dir, "Iter{}_MDS{}".format(iteration, mds), "metrics.json")
        print("MML Iteration: {}  MDS: {}".format(iteration, mds))
        den, con = get_metrics(metrics_json)
        print("Denotation Acc: {} Consistency: {}".format(den, con))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="CKPT dir containing MML/ ERM/ dirs")

    args = parser.parse_args()
    print(args)
    print_iterative_training_metrics(checkpoint_dir=args.ckpt_dir)
