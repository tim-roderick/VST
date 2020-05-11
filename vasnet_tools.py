'''
------------------------------------------------
The following code was created by Fajtl et al, utilised by me for specific utility.

https://github.com/ok1zjf/VASNet

@article{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    journal={arXiv:1812.01969},
    year={2018}
}

- Tim Roderick
------------------------------------------------
'''


import json
import os
import torch
import numpy as np
import subprocess
import platform
import sys
import pkg_resources
import h5py
import json
import ortools

from torch.nn.modules.module import _addindent


def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    parameters = 0
    convs = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [torch.nn.modules.container.Container, torch.nn.modules.container.Sequential]:
            modstr, p, cnvs = torch_summarize(module)
            parameters += p
            convs += cnvs
        else:
            modstr = module.__repr__()
            convs += len(modstr.split('Conv2d')) - 1

        modstr = _addindent(modstr, 2)
        # if 'conv' in key:
        #     convs += 1

        params = sum([np.prod(p.size()) for p in module.parameters()])
        parameters += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={} / {}'.format(params, parameters)
        tmpstr += ', convs={}'.format(convs)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, parameters, convs

def run_command(command):
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return '\n'.join([ '\t'+line.decode("utf-8").strip() for line in p.stdout.readlines()])

def ge_pkg_versions():
    dep_versions = {}
    dep_versions['display'] = run_command('cat /proc/driver/nvidia/version')

    dep_versions['cuda'] = 'NA'
    cuda_home = '/usr/local/cuda/'
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']

    cmd = cuda_home+'/version.txt'
    if os.path.isfile(cmd):
        dep_versions['cuda'] = run_command('cat '+cmd)

    dep_versions['cudnn'] = torch.backends.cudnn.version()
    dep_versions['platform'] = platform.platform()
    dep_versions['python'] = sys.version_info[:3]
    dep_versions['torch'] = torch.__version__
    dep_versions['numpy'] = np.__version__
    dep_versions['h5py'] = h5py.__version__
    dep_versions['json'] = json.__version__
    dep_versions['ortools'] = ortools.__version__
    dep_versions['torchvision'] = pkg_resources.get_distribution("torchvision").version

    # dep_versions['PIL'] = Image.VERSION
    # dep_versions['OpenCV'] = 'NA'
    # if 'cv2' in sys.modules:
    #     dep_versions['OpenCV'] = cv2.__version__


    return dep_versions