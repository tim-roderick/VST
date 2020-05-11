# VST: Video Summarisation Transformer
This repository is an implementation of the model found in the project Generating Summarised Videos Using Transformers.
This was my Masters Project from 2020. The implementation of the model is in PyTorch with the following details.

## Requirements

| Package | Version  |
| ------- | :------: |
| Python  |  3.6.8   |
| PyTorch |  1.4.0   |
|  NumPy  |  1.18.4  |
|  h5py   |  2.10.0  |
| ortools | 7.5.7466 |

## Installation
Cloning this repository as is should get you mostly what you need. You will also need the datasets as provided
in [VASnet](https://github.com/ok1zjf/VASNet#datasets-and-pretrained-models). Or alternatively [pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). Place these datasets in the datasets folder provided.

Split files have been provided, taken from [VASnet](https://github.com/ok1zjf/VASNet/tree/master/splits). To generate your own, please use the guide given in [VASnet](https://github.com/ok1zjf/VASNet#training).

Models produced by me and utilised for this project will be available to download shortly.

## How to ...
### Train
To train the model, make sure you have the datasets described above and ensure you have some train/test splits.
By running the following command, you will execute training for all splits. Details of parameters can be found by running main.py with --help.

```python3 main.py --train --model_dir models/```

### Evaluate
To evaluate the models you have created, run the following command. Details of parameters can be found by running main.py with --help. To utilise beam search, provide a non-zero ``--beam_width``.

```python3 main.py --model_dir models/```

### Visualise
Limited visualisation examples can be found in the notebook visualisations.ipynb. Examples include how to select a specific output from the evaluation set to isolate a machine summary. Example visualisations include...

<object data="./images/attention_visual_decoder.pdf" type="application/pdf" width="40%" height="40%">
    <embed src="./images/attention_visual_decoder.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it</p>
    </embed>
</object>

To generate actual summaries, [pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce#visualize-summary) provide details on how to generate an MP4 video from a set of frames using a machine summary produced by the model.

### Use your own data
Although not utilised by me, details for this can be found in [pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce#how-to-use-your-own-data)

## Acknowledgement
Thanks to the work of Zhou et al and Fajtl et al, and OpenNMT this implementation was possible. Where their code has been utilised, a reference should follow. Thank you also to Zhang et al also for contributing to the processing of the datasets referenced prior alongside Zhou et al and Fajtl et al. Citations can be found below for their work. If I have missed a reference of any sort please submit an issue.

```
@misc{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    year={2018},
    eprint={1812.01969},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
```
@article{zhou2017reinforcevsumm,
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao},
   journal={arXiv:1801.00054},
   year={2017}
}
```
```
@inproceedings{zhang2016video,
  title={Video summarization with long short-term memory},
  author={Zhang, Ke and Chao, Wei-Lun and Sha, Fei and Grauman, Kristen},
  booktitle={ECCV},
  year={2016}
}
```