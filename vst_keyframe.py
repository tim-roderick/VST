'''
------------------------------------------------
The following code was created by me, with small functions 
taken from Fajtl et al for specific utility. Wherever I write
"from VASnet", I am referring to code taken/amended from the following
repository and paper:

https://github.com/ok1zjf/VASNet

@article{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    journal={arXiv:1812.01969},
    year={2018}
}

------------------------------------------------
'''

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import os
import h5py
import pandas as pd
from tqdm import tqdm

from vasnet_tools import *
from vsum_tools import *
from model import *
from train import *
from vst import *

class VST_keyframe(VST):
    def __init__(self, parameters):
        print("——— Initialising VST keyframe")
        rnd_seed = 16259
        
        # Set random seed for all packages that use randomness
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.parameters = parameters
        self.summary_writer = None

        # 0 or 1 so vocab size is 2, 3 for start symbol
        self.model = make_model(tgt_vocab=3, d_model=1024, d_ff=1024, h=8, dropout=self.parameters.dropout)

        if parameters.model_summary:
            string, _, _ = torch_summarize(self.model)
            print(string)

        self.optimizer = None
        self.criterion = LabelSmoothing(size=3, padding_idx=2, smoothing=0.15)
        self.dataset_name = None
        self.dataset_type = None
        self.datasets = {}
        self.splits = None
        self.split_num = None
        self.split_file = None
        self.train_keys = None
        self.test_keys = None
        self.model.eval()

        if self.parameters.cuda:
            torch.cuda.set_device(0)
            torch.cuda.manual_seed(rnd_seed)
            self.model.cuda()

    def run_epoch(self, epoch, keys, loss_compute):
        if self.parameters.verbose:
            iterable = tqdm(keys) 
        else:
            iterable = keys

        average_loss = 0

        for i, key in enumerate(iterable):
            # key is of the form dataset/video_id
            dataset, video = key.split('/')
            data = self.datasets[dataset][video]

            input_sequence = data['features'][...]
            target_scores = data['gtsummary'][...]

            # Add <BOS> character which in our case is 2
            target_scores = np.insert(target_scores, 0, 2)

            # Make them tensors, making the 'batch' of size 1
            input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
            target_scores = torch.from_numpy(target_scores).unsqueeze(0).long()
            
            # target_scores_y = torch.from_numpy(target_scores_y).unsqueeze(0)
            target_scores_y = target_scores[:, 1:] # Attends to all but the first
            target_scores = target_scores[:, :-1] # shifted right by 1

            enc_mask, dec_mask = make_std_mask(input_sequence, target_scores)
            
            if self.parameters.cuda:
                input_sequence = input_sequence.cuda()
                target_scores = target_scores.cuda()
                target_scores_y = target_scores_y.cuda()
                enc_mask = enc_mask.cuda()
                dec_mask = dec_mask.cuda()

            # FORWARD MODEL AND OPTIMISER #

            y = self.model.forward(input_sequence, target_scores, enc_mask, dec_mask)


            loss = loss_compute(y, target_scores_y, target_scores_y.size(-1))
            average_loss += loss

        average_loss /= len(keys)

        return average_loss

    def eval(self, epoch, keys):
        
        print(f"— Evaluating Test Samples for epoch: {epoch+1}")

        machine_summary = {}

        self.model.eval()   

        if self.parameters.verbose:
            iterable = tqdm(keys) 
        else:
            iterable = keys

        with torch.no_grad():
            for i, key in enumerate(iterable):
                # key is of the form dataset/video_id
                dataset, video = key.split('/')
                data = self.datasets[dataset][video]

                input_sequence = data['features'][...]

                # Make them tensors, making the 'batch' of size 1
                input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)

                enc_mask, _ = make_std_mask(input_sequence)

                if self.parameters.cuda:
                    input_sequence = input_sequence.float().cuda()
                    enc_mask = enc_mask.float().cuda()

                if self.parameters.beam_width > 0:
                    machine_summary[key] = beam_search(self, self.parameters.beam_width, input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=2)[0][1:]
                else:
                    machine_summary[key] = self.greedy_decode(input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=2)[0][1:]
    
        mean_f_score, video_scores = self.eval_summary(machine_summary, keys, metric=self.dataset_name)
        print(f"Mean F-score: {mean_f_score}")

        # average_loss = 0
        
        # for _, key in enumerate(self.test_keys):
        #     dataset, video = key.split('/')
        #     # Get data from dataset at the video we care about
        #     data = self.datasets[dataset][video]

        #     # Extract the input sequence from the h5 data for testing
        #     input_sequence = data['features'][...]
        #     target_scores = data['gtscore'][...]
        #     summary = data['gtsummary'][...]
            

        #     original_scores = torch.from_numpy(target_scores).unsqueeze(0)

        #     target_scores = np.tile(target_scores, (1024, 1)).transpose()

        #     # Make them tensors, making the 'batch' of size 1
        #     input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
        #     target_scores = torch.from_numpy(target_scores).unsqueeze(0)

        #     # (?) Normalise frame scores
        #     target_scores -= target_scores.min()
        #     target_scores /= target_scores.max()
            
        #     original_scores -= original_scores.min()
        #     original_scores /= original_scores.max()

        #     # Create masks for the encoder and decoder
        #     # enc_mask = torch.ones(size=(input_sequence.size(1), input_sequence.size(1))).unsqueeze(0)
        #     # dec_mask = subsequent_mask(input_sequence.size(1))
        #     enc_mask, dec_mask = make_std_mask(input_sequence, target_scores)

        #     # (?) Cuda
        #     if self.parameters.cuda:
        #             input_sequence = input_sequence.float().cuda()
        #             target_scores = target_scores.float().cuda()
        #             enc_mask = enc_mask.float().cuda()
        #             dec_mask = dec_mask.float().cuda()

        #     # Forward pass of the model
        #     y = self.model.forward(input_sequence, target_scores[:, :-1, :], enc_mask, dec_mask[:, :-1, :-1])

        #     # Calculate loss to log using summary writer
        #     loss, generated = loss_backprop(self.model.generator, self.criterion, y, original_scores[:, 1:], 1, bp=False)
        #     average_loss += loss

        #     machine_summary[key] = generated

        # mean_f_score, video_scores = self.eval_summary(machine_summary, self.test_keys, metric=self.dataset_name)
        # # Log results for this eval
        # self.summary_writer.add_scalar('Loss/test', average_loss/len(self.test_keys), epoch)
        # self.summary_writer.add_scalar('Accuracy/test', mean_f_score, epoch)

        # Print results for this eval
        # print(f"Mean Loss: {average_loss/len(self.test_keys)}")
        # print(f"Mean F-score: {mean_f_score}")

        return mean_f_score, video_scores

    def eval_single(self, epoch, key):
        
        print(f"— Evaluating Test Samples: {key}")

        machine_summary = {}

        self.model.eval()   

        with torch.no_grad():
            # key is of the form dataset/video_id
            dataset, video = key.split('/')
            data = self.datasets[dataset][video]

            input_sequence = data['features'][...]

            input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)

            enc_mask, _ = make_std_mask(input_sequence)

            if self.parameters.cuda:
                input_sequence = input_sequence.float().cuda()
                enc_mask = enc_mask.float().cuda()

            if self.parameters.beam_width > 0:
                machine_summary[key] = beam_search(self, self.parameters.beam_width, input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=2, norm=1)[0][1:]
            else:
                machine_summary[key] = self.greedy_decode(input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=2)[0][1:]
            


        f_score, summary, data = self.eval_single_summary(machine_summary, key, metric=self.dataset_name)

        return machine_summary[key], summary, f_score, data

    def greedy_decode(self, src, src_mask, max_len, start_symbol=0):
        # This requires a start symbol, this is represented by the value -1 in our target scores
        memory = self.model.encode(src, src_mask)
            
        ys = torch.ones(1, 1).long().fill_(start_symbol)
        
        for i in range(max_len):
            out = self.model.decode(memory, src_mask, 
                            Variable(ys).cuda() if self.parameters.cuda else Variable(ys), 
                            Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).long().fill_(next_word)], dim=1)
        
        final_y = ys.view(1,-1).data.cpu().numpy()[:, 1:]
        return final_y 

    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):
        """
        Amended from VASnet
        """
        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            dataset, video = key.split('/')
            data = self.datasets[dataset][video]

            probs = machine_summary_activations[key]

            if 'change_points' not in data:
                print("ERROR: No change points in dataset/video ",key)

            cps = data['change_points'][...]
            num_frames = data['n_frames'][()]
            nfps = data['n_frame_per_seg'][...].tolist()
            positions = data['picks'][...]
            user_summary = data['user_summary'][...]

            machine_summary = self.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

        mean_fm = np.mean(fms)

        return mean_fm, video_scores

    def generate_summary(self, ypred, cps, n_frames, nfps, positions):
        """
        Amended from VASnet
        Generate keyshot-based video summary i.e. a binary vector.
        Args:
        ---------------------------------------------
        - ypred: predicted importance scores.
        - cps: change points, 2D matrix, each row contains a segment.
        - n_frames: original number of frames.
        - nfps: number of frames per segment.
        - positions: positions of subsampled frames in the original video.
        """
        n_segs = cps.shape[0]
        frame_scores = np.zeros((n_frames), dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])

        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i+1]
            if i == len(ypred):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = ypred[i]

        return frame_scores

    def save_model(self, epoch):
        os.makedirs(self.parameters.model_dir + 'temp', exist_ok=True)
        path = os.path.join(self.parameters.model_dir + 'temp', 'epoch-'+str(epoch)+'.pth')
        print(f"Saving model to {path}")
        torch.save({
            'epoch': self.parameters.epochs,
            'arguments': self.parameters,
            'model_state_dict': self.model.state_dict(),
            'encoder_attention': self.model.encoder.layers,
            'decoder_attention': self.model.decoder.layers
        }, path)

    def get_summary_writer_log_dir(self):
        tb_log_dir_prefix = (
        f"VST_keyframe_"
        f"dropout={self.parameters.dropout}_"
        f"epochs={self.parameters.epochs}_"
        f"lr_factor={self.parameters.lr_factor}_"
        f"split_file={self.split_file}_{self.split_num}"
        f"run_"
        )

        i = 0
        while i < 1000:
            tb_log_dir = "logs/" + tb_log_dir_prefix + str(i)
            if not os.path.exists(tb_log_dir):
                return str(tb_log_dir)
            i += 1
        return str(tb_log_dir)
