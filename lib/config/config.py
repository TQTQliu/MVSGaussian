from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs


cfg = CN()

os.environ['workspace'] = "."
cfg.workspace = os.environ['workspace']
print('Workspace: ', cfg.workspace)

cfg.level = 32.
cfg.resolution = 256

cfg.vis_encoder = ''
cfg.feat_vis_len = 8

cfg.cache_data = False
cfg.sample_keypoints_epoch = -1

cfg.write_video = False
cfg.interested_mask = False
cfg.render_path = False
cfg.render_emb = 0
cfg.render_ixt = 0
cfg.code_id = -1
cfg.time_weight = 0.
cfg.render_static = True
cfg.pretrain_path = ''
cfg.scene = 'test'
cfg.last_view = False
cfg.exp_hard = False
cfg.pos_encoding_t = False
cfg.render_time = False
cfg.render_time_skip = [0, -1, 1]
cfg.start_time = [2009, 8, 1]
cfg.end_time = [2013, 12, 1]
cfg.discrete_3views = False
cfg.fps = 24
cfg.dcat = False
cfg.min_y = -100000000.
cfg.time_discrete = -1
cfg.render_day = False
cfg.render_date = [2013, 1, 1]
cfg.rand_t = -1.
cfg.semantic_mask = False
cfg.product_combine = False
cfg.unisample = False
cfg.render_emb_2 = -1
cfg.render_num = 30
cfg.render_ext = 0
cfg.time_geo = False
cfg.reg_beta = False
cfg.fix_beta = False
cfg.hard_lap = False
cfg.render_octree = False
cfg.render_mask = False
cfg.environment_map = False

cfg.save_result = False
cfg.clear_result = False
cfg.save_tag = 'default'

# experiment name
cfg.exp_name = 'gitbranch_hello'
cfg.exp_name_tag = ''
cfg.pretrain = ''

# network
cfg.distributed = False

# task
cfg.task = 'hello'

# gpus
cfg.gpus = list(range(4))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1
cfg.eval_ep = 1
log_interval: 20


cfg.task_arg = CN()
cfg.task_arg.sample_more_on_mask = -1.
cfg.task_arg.sample_on_mask = False

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({})
cfg.train.shuffle = True
cfg.train.eps = 1e-8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 5e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({})

# trained model
cfg.trained_model_dir = os.path.join(os.environ['workspace'], 'trained_model')
cfg.clean_tag = 'debug'

# recorder
cfg.record_dir = os.path.join(os.environ['workspace'], 'record')

# result
cfg.result_dir = os.path.join(os.environ['workspace'], 'result')

# evaluation
cfg.skip_eval = False

cfg.fix_random = False

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    if -1 not in cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if 'bbox' in cfg:
        bbox = np.array(cfg.bbox).reshape((2, 3))
        center, half_size = np.mean(bbox, axis=0), (bbox[1]-bbox[0]).max().item() / 2.
        bbox = np.stack([center-half_size, center+half_size])
        cfg.bbox = bbox.reshape(6).tolist()

    if len(cfg.exp_name_tag) != 0:
        cfg.exp_name +=  ('_' + cfg.exp_name_tag)
    cfg.exp_name = cfg.exp_name.replace('gitbranch', os.popen('git describe --all').readline().strip()[6:])
    cfg.exp_name = cfg.exp_name.replace('gitcommit', os.popen('git describe --tags --always').readline().strip())
    print('EXP NAME: ', cfg.exp_name)
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name, cfg.save_tag)
    cfg.local_rank = args.local_rank
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def make_cfg(args):
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        print(cfg_file)
        return cfg
    cfg_ = merge_cfg(args.cfg_file, cfg)
    try:
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except:
        cfg_.merge_from_list(args.opts)
    parse_cfg(cfg_, args)
    return cfg_


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/mvsgs/dtu_pretrain.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
