import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init
import pandas as pd

from vasnet_tools import *
from vst import *
from vst_keyframe import *

class Parameters:
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)

        self.datasets=['datasets/eccv16_dataset_summe_google_pool5.h5',
                       'datasets/eccv16_dataset_tvsum_google_pool5.h5',
                       'datasets/eccv16_dataset_ovp_google_pool5.h5',
                       'datasets/eccv16_dataset_youtube_google_pool5.h5']

        self.splits = self.splits.split(',')
        
    def __str__(self):
        attribute_list = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]
        parameters = pd.DataFrame({
            'Parameter': attribute_list,
            'Value': list(map(lambda x: getattr(self, x), attribute_list))
        })
        return str(parameters)


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="Implementation of (insert project name here) (AKA VST) in PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train", 
        action='store_true',
        help="Train model with splits in config"
    )

    parser.add_argument(
        "--verbose", 
        action='store_true',
        help="Prints more details while running"
    )

    parser.add_argument(
        "--cuda", 
        action='store_true',
        help="Whether to use GPU for training"
    )

    parser.add_argument(
        "--model_summary", 
        action='store_true',
        help="Print model summary"
    )

    parser.add_argument(
        "--epochs", 
        type=int,
        default=50,
        help="Number of epochs to train with"
    )

    parser.add_argument(
        "--log_interval", 
        type=int,
        default=1,
        help="Number of epochs between consecutive logs"
    )

    parser.add_argument(
        "--eval_interval", 
        type=int,
        default=1,
        help="Number of epochs between consecutive evaluations of the model"
    )

    parser.add_argument(
        "--dropout", 
        type=float,
        default=0.1,
        help="Dropout probability applied for each dropout layer"
    )

    parser.add_argument(
        "--smoothing", 
        type=float,
        default=0.0,
        help="Smoothing used in label smoothing"
    )

    parser.add_argument(
        "--lr_factor", 
        type=float,
        default=1,
        help="Factor times by learning rate over time"
    )

    parser.add_argument(
        "--attention_heads", 
        type=int,
        default=8,
        help="Number of attention heads per encoder/decoder layer"
    )

    parser.add_argument(
        "--layers", 
        type=int,
        default=6,
        help="Number of encoder/decoder layers"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default='importance_score',
        help="Model to use, (importance_score, keyframe) "
    )

    parser.add_argument(
        "--beam_width", 
        type=int,
        default=0,
        help="Width to use for beam search used to generate the output sequence"
    )

    parser.add_argument(
        "--feed_forward_size", 
        type=int,
        default=1024,
        help="Width to use for each feed forward layer."
    )

    parser.add_argument(
        "--splits", 
        type=str,
        default='splits/tvsum_splits.json,splits/summe_splits.json,splits/tvsum_aug_splits.json,splits/summe_aug_splits.json',
        help="Comma seperated list of split files"
    )

    parser.add_argument(
        "--model_dir", 
        type=str,
        default='models/',
        help="Directory to put/get models from"
    )


    return parser

if __name__ == "__main__":
    # Read arguements in
    args = make_arg_parser().parse_args()

    # Create parameters class to hold our preferences
    parameters = Parameters(args)
    print("——— Parameters:")
    print(parameters, '\n')


    if parameters.train:
        # Train the model
        print("—————— Starting training for splits:", *parameters.splits, sep='\n• ')
        
        f_scores = []

        for split_file in parameters.splits:
            dataset_name, dataset_type, splits = parse_splits_filename(split_file)
            if dataset_type:
                datasets = parameters.datasets
            else:
                datasets = [filename for filename in parameters.datasets if dataset_name in filename]
 
            avg_dataset_f_score_mean = 0
            avg_dataset_f_score_max = 0

            for i in range(len(splits)):
                print(f"———— Training for split: {i+1} / {len(splits)} in {split_file}")
                if parameters.model == "importance_score":
                    vst = VST(parameters)
                elif parameters.model == "keyframe":
                    vst = VST_keyframe(parameters)
                        
                os.makedirs(parameters.model_dir, exist_ok=True)

                vst.load_h5_datasets(datasets, dataset_name)
                vst.splits = splits
                vst.split_file = split_file 
                vst.dataset_type = dataset_type
                vst.get_split_at(i, datasets[0])

                split_f_score_mean, split_f_score_max, split_max_epoch = vst.train()

                # Move model with highest F-score to models directory
                model_file = os.path.join(parameters.model_dir + 'temp', 'epoch-'+str(split_max_epoch)+'.pth')
                new_model_file = os.path.splitext(os.path.basename(split_file))[0]

                os.system('mv '+model_file+' '+parameters.model_dir+parameters.model+'_'+new_model_file+'_'+str(i+1))
                os.system('rm -r ' + parameters.model_dir +'temp/')

                avg_dataset_f_score_mean += split_f_score_mean
                avg_dataset_f_score_max += split_f_score_max

            avg_dataset_f_score_mean /= len(splits)
            avg_dataset_f_score_max /= len(splits)

            print(f"———— Mean of Mean F-score for {split_file}: {avg_dataset_f_score_mean}")
            print(f"———— Mean of Max F-score for {split_file}: {avg_dataset_f_score_max}")

            f_scores.append(avg_dataset_f_score_max)

            # "write results"
        
        final_results = pd.DataFrame({
            'Split': parameters.splits,
            'F-score': f_scores
        })
        
        print(final_results)
    else:
        # Evaluate all splits and print their F-score
        i = 0
        fscores = []

        for split_file in parameters.splits:
            average_fscore = 0
            dataset_name, dataset_type, splits = parse_splits_filename(split_file)
            if dataset_type:
                datasets = parameters.datasets
            else:
                datasets = [filename for filename in parameters.datasets if dataset_name in filename]

            for i in range(len(splits)):
                if parameters.model == "importance_score":
                    vst = VST(parameters)
                elif parameters.model == "keyframe":
                    vst = VST_keyframe(parameters)


                vst.load_h5_datasets(datasets, dataset_name)
                vst.splits = splits
                vst.split_file = split_file 
                vst.dataset_type = dataset_type
                vst.get_split_at(i, datasets[0])

                model_file = os.path.splitext(os.path.basename(split_file))[0]
                filename = parameters.model_dir+parameters.model+'_'+model_file+'_'+str(i+1)
                
                print(f"——— Loading model from file {filename}")
                vst.load_model_from_file(filename)

                print(f"———— Evaluating split: {i+1} / {len(splits)} in {split_file}")
                f_score, video_scores = vst.eval(0, vst.test_keys)

                average_fscore += f_score
            
            fscores.append(average_fscore/len(splits))
        
        final_results = pd.DataFrame({
            'Split': parameters.splits,
            'F-score': fscores
        })

        print(final_results)
