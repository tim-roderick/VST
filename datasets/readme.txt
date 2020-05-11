Amendment by Tim Roderick:
I would like to thank Fajtl et al for curating
the datasets and this readme. I have added their paper
to the citations below. I provide this file as it was
used by me with no changes, taken from the repository:

https://github.com/ok1zjf/VASNet


################ Instructions ################
This folder contains four datasets for video summarization:
(1): eccv16_dataset_summe_google_pool5.h5
(2): eccv16_dataset_tvsum_google_pool5.h5
(3): eccv16_dataset_ovp_google_pool5.h5
(4): eccv16_dataset_youtube_google_pool5.h5

Each dataset follows the same data structure:
***********************************************************************************************************************************************
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    posotions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
***********************************************************************************************************************************************
Note: OVP and YouTube only contain the first three keys, i.e. ['features', 'gtscore', 'gtsummary']


If you want to create your own h5 dataset following this format, you can do:
```
import h5py
h5_file_name = 'blah blah blah'
f = h5py.File(h5_file_name, 'w')

# video_names is a list of strings containing the 
# name of a video, e.g. 'video_1', 'video_2'
for name in video_names:
    f.create_dataset(name + '/features', data=data_of_name)
    f.create_dataset(name + '/gtscore', data=data_of_name)
    f.create_dataset(name + '/user_summary', data=data_of_name)
    f.create_dataset(name + '/change_points', data=data_of_name)
    f.create_dataset(name + '/n_frame_per_seg', data=data_of_name)
    f.create_dataset(name + '/n_frames', data=data_of_name)
    f.create_dataset(name + '/picks', data=data_of_name)
    f.create_dataset(name + '/n_steps', data=data_of_name)
    f.create_dataset(name + '/gtsummary', data=data_of_name)
    f.create_dataset(name + '/video_name', data=data_of_name)
f.close()
```


################ Acknowledgement ################
We greatly thank Ke Zhang and Wei-Lun Chao for
providing GoogLeNet features.



################ Citation ################
If you use this data, please kindly cite the following papers:

@inproceedings{zhang2016video,
  title={Video summarization with long short-term memory},
  author={Zhang, Ke and Chao, Wei-Lun and Sha, Fei and Grauman, Kristen},
  booktitle={ECCV},
  year={2016}
}

@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward}, 
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}

@article{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    journal={arXiv:1812.01969},
    year={2018}
}
