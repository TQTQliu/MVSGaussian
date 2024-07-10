from lib.config import cfg, args
import numpy as np
from fusion import fusion

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))

def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    scenes = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                if 'novel_view' in k:
                    for v in batch[k]:
                        batch[k][v] = batch[k][v].cuda()
                elif k == 'rendering_video_meta':
                    for i in range(len(batch[k])):
                        for v in batch[k][i]:
                            batch[k][i][v] = batch[k][i][v].cuda()
                else:
                    batch[k] = batch[k].cuda()
        if cfg.save_video:
            with torch.no_grad():
                network(batch)
        else:
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()
                output = network(batch)
                torch.cuda.synchronize()
                end_time = time.time()
            net_time.append(end_time - start_time)
            evaluator.evaluate(output, batch)
        scenes.append(batch['meta']['scene'][0])
            
    if not cfg.save_video:
        evaluator.summarize()
        if len(net_time) > 1:
            # print('net_time: ', np.mean(net_time[1:]))
            print('FPS: ', 1./np.mean(net_time[1:]))
        else:
            # print('net_time: ', np.mean(net_time))
            print('FPS: ', 1./np.mean(net_time))
    if cfg.save_ply:
        for scene in scenes:
            fusion(cfg.dir_ply, scene)
        
if __name__ == '__main__':
    globals()['run_' + args.type]()
