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
from torch.autograd import Variable

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
from beam_search import beam_search

class VST:
    def __init__(self, parameters):
        print("——— Initialising VST")
        rnd_seed = 16259
        
        # Set random seed for all packages that use randomness
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.parameters = parameters
        self.summary_writer = None

        # as the total size of the vocab is np.arange(0, 1, 0.00001) + 1.00000 + <BOS> = 100,002
        # index 0..100000 = 0...0.99999,1 | index 100001 = start_symbol
        # to incorporate this, scale everything so that it is 1 greater
        self.model = make_model(tgt_vocab=100002, d_model=1024, d_ff=self.parameters.feed_forward_size, h=8, dropout=self.parameters.dropout)

        if parameters.model_summary:
            string, _, _ = torch_summarize(self.model)
            print(string)

        self.optimizer = None
        self.criterion = LabelSmoothing(size=100002, padding_idx=100001, smoothing=self.parameters.smoothing)
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

    def load_h5_datasets(self, datasets, dataset_name):
        self.dataset_name = dataset_name
        for dataset in datasets:
            file = os.path.splitext(os.path.basename(dataset))[0]
            self.datasets[file] = h5py.File(dataset, 'r')
    
    def load_model_from_file(self, file):
        if self.parameters.cuda:
            location = None
        else:
            location=torch.device('cpu')

        self.model.load_state_dict(torch.load(file, map_location=location)['model_state_dict'])
    
    def get_split_at(self, i, dataset):
        assert i < len(self.splits), "Split index out of range"
        split = self.splits[i]
        self.split_num = i
        self.train_keys = self.fix_keys(split['train_keys'], dataset) 
        self.test_keys = self.fix_keys(split['test_keys'], dataset)

    def fix_keys(self, keys, dataset_name = None):
        """
        Taken from VASnet
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out
    
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
            target_scores = data['gtscore'][...]
            target_scores = np.around(target_scores*100000)

            if not (target_scores <= 100000).all():
                target_scores /= max(target_scores) / 100000

            # Add <BOS> character which in our case is 100001
            target_scores = np.insert(target_scores, 0, 100001)

            # Make them tensors, making the 'batch' of size 1
            input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
            target_scores = torch.from_numpy(target_scores).unsqueeze(0).long()
            
            # target_scores_y = torch.from_numpy(target_scores_y).unsqueeze(0)
            target_scores_y = target_scores[:, 1:] # Attends to all but the first TODO: Try without this as it may be wrong
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

            # Calculate loss, zeroing the gradients before backward pass (sometimes necessary).
            # Also, loss here is normalised for the batch, as the batch size is 1, we don't normalise again
            loss = loss_compute(y, target_scores_y, target_scores_y.size(-1)) # 1 for now as I havent accounted properly for normalisation

            average_loss += loss

            if self.parameters.cuda:
                torch.cuda.empty_cache()
            # self.summary_writer.add_scalar('Loss/train', loss, self.writer_step)
            # self.writer_step += 1
        
        average_loss /= len(keys)

        # normalised
        return average_loss

    def train(self):
        self.summary_writer = SummaryWriter(self.get_summary_writer_log_dir())
        self.optimizer = get_std_opt(self.model, 1024, self.parameters.lr_factor)
        self.writer_step = 0

        # average_loss = []
        f_scores_max = 0
        f_scores_max_epoch = 0

        for epoch in range(self.parameters.epochs):
            print(f"—— Epoch ( {epoch+1} / {self.parameters.epochs} )")
            self.model.train()
            random.shuffle(self.train_keys)
            
            # train epoch
            train_loss = self.run_epoch(epoch, self.train_keys, SimpleLossCompute(self.model.generator, self.criterion, self.optimizer))

            # validation epoch, optimiser is None to avoid backprop
            test_loss = self.run_epoch(epoch, self.test_keys, SimpleLossCompute(self.model.generator, self.criterion, None))

            if epoch % self.parameters.log_interval == 0:
                self.log(epoch, train_loss)
                self.log(epoch, test_loss, training=False)
                print(f"— Train Loss: {train_loss}")
                print(f"— Test Loss: {test_loss}")

                # evaluate to get F-score

            if epoch % self.parameters.eval_interval == 0:
                mean_f_score_test, video_scores = self.eval(epoch, self.test_keys)
                self.log(epoch, score=mean_f_score_test, training=False)
                
                # only do this if not ovp/youtube
                # if not self.dataset_type:
                #     mean_f_score_train, _ = self.eval(epoch, self.train_keys)
                #     self.log(epoch, score=mean_f_score_train)
                
                # need to record which epoch this is so that it can be chosen from the saved models
                if f_scores_max < mean_f_score_test:
                    f_scores_max = mean_f_score_test
                    f_scores_max_epoch = epoch

                if self.parameters.verbose:
                    scores = pd.DataFrame({
                        "Video": [row[1] for row in video_scores],
                        "F-score": [row[2] for row in video_scores]
                    })
                    print("- Individual Video Scores:")
                    print(scores, '\n')
                
            # SAVE MODEL  
            self.save_model(epoch)

            # perform final eval after last epoch
        mean_f_score, _ = self.eval(self.parameters.epochs, self.test_keys)
                
        # RETURN STUFF
        return mean_f_score, f_scores_max, f_scores_max_epoch 


                # print(f"— Evaluating Test Samples for epoch: {epoch+1}")
                # mean_f_score, _ = self.eval(epoch)

                # if f_scores_max < mean_f_score:
                #     f_scores_max = mean_f_score

                # Keep track of loss for average
                # average_loss.append(loss)
                # FORWARD MODEL AND OPTIMISER #

                # LOG #                
                # log loss and 'accuracy'
                # machine_summary = {}
                # machine_summary[key] = generated
                

                # # Can't caluculate the f-score properly due to a lack of changepoints
                # # in ovp / youtube, so we only do this step for those that aren't augmented
                # if not self.dataset_type:
                #     mean_f_score, _ = self.eval_summary(machine_summary, [key], metric=self.dataset_name)
                #     self.summary_writer.add_scalar('Accuracy/train', mean_f_score, step)

                # step += 1
                # LOG #

            # Evaluate test samples every X epochs
            # if epoch % self.parameters.log_interval == 0:
            #     print(f"— Evaluating Test Samples for epoch: {epoch+1}")
            #     mean_f_score, _ = self.eval(epoch)

            #     if f_scores_max < mean_f_score:
            #         f_scores_max = mean_f_score

                
        
        # print("— Evaluating Test Samples after last epoch")
        # mean_f_score, video_scores = self.eval(self.parameters.epochs)
        
        # if self.parameters.verbose:
        #     scores = pd.DataFrame({
        #         "Video": [row[1] for row in video_scores],
        #         "F-score": [row[2] for row in video_scores]
        #     })
        #     print("- Individual Video Scores:")
        #     print(scores, '\n')
        
        # # SAVE MODEL
        # self.save_model(self.parameters.epochs, mean_f_score)

        # RETURN STUFF
        # return mean_f_score, f_scores_max

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

                # start symbol is 0 as start symbol is -1 +1
                if self.parameters.beam_width > 0:
                    machine_summary[key] = beam_search(self, self.parameters.beam_width, input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=100001)[0][1:]
                else:
                    machine_summary[key] = self.greedy_decode(input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=100001)[0][1:]
                


        mean_f_score, video_scores = self.eval_summary(machine_summary, keys, metric=self.dataset_name)
        print(f"Mean F-score: {mean_f_score}")

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

            # Make them tensors, making the 'batch' of size 1
            input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)

            enc_mask, _ = make_std_mask(input_sequence)

            if self.parameters.cuda:
                input_sequence = input_sequence.float().cuda()
                enc_mask = enc_mask.float().cuda()

            # start symbol is 0 as start symbol is -1 +1
            if self.parameters.beam_width > 0:
                machine_summary[key] = beam_search(self, self.parameters.beam_width, input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=100001)[0][1:]
            else:
                machine_summary[key] = self.greedy_decode(input_sequence, enc_mask, max_len=input_sequence.size(1), start_symbol=100001)[0][1:]
            


        f_score, summary, data = self.eval_single_summary(machine_summary, key, metric=self.dataset_name)

        return machine_summary[key], summary, f_score, data

    def log(self, epoch, loss=None, score=None, training=True):
        type_string = 'train' if training else 'test'

        if loss is not None:
            self.summary_writer.add_scalar('Loss/' + type_string, loss, epoch) 

        if score is not None:
            self.summary_writer.add_scalar('Accuracy/' + type_string, score, epoch) 

    def greedy_decode(self, src, src_mask, max_len, start_symbol=0):
        memory = self.model.encode(src, src_mask)
            
        ys = torch.ones(1, 1).long().fill_(start_symbol)
        
        for i in range(max_len-1):
            out = self.model.decode(memory, src_mask, 
                            Variable(ys).cuda() if self.parameters.cuda else Variable(ys), 
                            Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).long().fill_(next_word)], dim=1)
        # vocab is probabilities turned to ints, so convert back here
        final_y = np.around( (ys.view(1,-1).data.cpu().numpy()) / 100000, decimals=5)
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

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

        mean_fm = np.mean(fms)

        return mean_fm, video_scores

    def eval_single_summary(self, machine_summary_activations, key, results_filename=None, metric='tvsum', att_vecs=None):
        """
        Amended from VASnet
        """
        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        # if results_filename is not None:
        #     h5_res = h5py.File(results_filename, 'w')

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

        machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
        fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)


        return fm, machine_summary, data

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
        # Edit the following to more closely represent our problem
        tb_log_dir_prefix = (
        f"VST_"
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
