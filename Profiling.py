# run_exp.py

import os
import sys
import time
import logging
from datetime import datetime
import importlib
import random
import psutil

import numpy as np
import torch
import yaml

from server import Server
from client import Client

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False

def setup_client_selection(server, CS_algo, CS_args):
    fn = getattr(server, f"client_selection_{CS_algo}", None)
    if fn is None:
        raise ValueError(f"Unknown client selection algorithm: {CS_algo}")
    return fn, CS_args

def setup_aggregation(server, Agg_algo, Agg_args):
    fn = getattr(server, f"aggregate_{Agg_algo}", None)
    if fn is None:
        raise ValueError(f"Unknown aggregation algorithm: {Agg_algo}")
    return fn, Agg_args

def main(rounds, seed, client_list, client_selection, CS_args,
         aggregation, Agg_args, client_train_config, client_test_config):
    fix_seed(seed)
    proc = psutil.Process(os.getpid())

    cumulative_stats = {
        'cs_time': 0.0,
        'train_time': 0.0,
        'agg_time': 0.0,
        'val_time': 0.0
    }

    for rnd in range(1, rounds + 1):
        CS_args['round'] = rnd
        Agg_args['round'] = rnd

        # --- Client selection timing ---
        t0 = time.time()
        selected = client_selection(client_list, CS_args)
        dt = time.time() - t0
        cumulative_stats['cs_time'] += dt
        logging.info(f"ROUND {rnd} SELECTED_CLIENTS={selected} CS_TIME={dt:.4f}s")

        # --- Per-client training + CPU-mem logging ---
        for cid in selected:
            mem_before = proc.memory_info().rss
            t_start = time.time()

            num_batches = client_list[cid].train(round_id=rnd, args=client_train_config)

            t_elapsed = time.time() - t_start
            mem_after = proc.memory_info().rss
            cpu_delta = mem_after - mem_before

            avg_batch = (t_elapsed / num_batches) if num_batches else 0.0
            cumulative_stats['train_time'] += t_elapsed

            logging.info(
                f"CID {cid} ROUND {rnd} "
                f"TIME={t_elapsed:.4f}s BATCHES={num_batches} AVG_BATCH={avg_batch:.4f}s "
                f"CPU_MEM_DELTA={cpu_delta//1024}KiB"
            )

        # --- Aggregation timing ---
        agg_start = time.time()
        global_wts, client_list = aggregation(
            selected_cids=selected, client_list=client_list, **Agg_args
        )
        agg_dt = time.time() - agg_start
        cumulative_stats['agg_time'] += agg_dt
        logging.info(f"ROUND {rnd} AGG_TIME={agg_dt:.4f}s")

        server.model.load_state_dict(global_wts)

        # --- Validation timing ---
        v_start = time.time()
        server.test(round_id=rnd)
        v_dt = time.time() - v_start
        cumulative_stats['val_time'] += v_dt
        logging.info(f"ROUND {rnd} VAL_TIME={v_dt:.4f}s")

        # Round summary
        print(f"\n--- Round {rnd} Summary ---")
        print(f"  CS time:   {dt:.4f}s")
        print(f"  Train sum: {cumulative_stats['train_time']:.4f}s")
        print(f"  Agg time:  {agg_dt:.4f}s")
        print(f"  Val time:  {v_dt:.4f}s")
        print("---------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp.py <config.yaml>")
        sys.exit(1)

    # Load YAML config
    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f)

    # Federated-learning settings
    CS_algo           = cfg['FL_config']['CS_algo']
    Agg_algo          = cfg['FL_config']['Agg_algo']
    total_rounds      = cfg['FL_config']['total_rounds']
    total_clients     = cfg['FL_config']['total_num_clients']
    clients_per_round = cfg['FL_config']['clients_per_round']

    # Model & data
    model_id         = cfg['ML_config']['model_id']
    dataset_id       = cfg['ML_config']['dataset_id']
    model_path       = cfg['ML_config']['model_file_path']
    model_args       = cfg['ML_config']['model_args']
    data_dir         = cfg['ML_config']['dataset_dir']
    init_model_dir   = cfg['ML_config']['initial_model_path']

    # Server config
    seed      = cfg['server_config']['seed']
    save_root = cfg['server_config']['save_path']

    device = "cpu"

    # Dynamically import model class
    sys.path.append(os.path.dirname(os.path.dirname(model_path)))
    mod_name = os.path.splitext(os.path.basename(model_path))[0]
    module   = importlib.import_module(f"models.{mod_name}")
    ModelCls = getattr(module, model_id)

    # Ensure an initial model exists
    init_path = os.path.join(init_model_dir, f"{model_id}.pth")
    if not os.path.exists(init_path):
        init_model = ModelCls(cid="init", args=model_args)
        torch.save(init_model, init_path)
        print("Created initial model at", init_path)

    # Base CS/Agg args (only update if dict)
    base_CS_args = {
        "round": 0,
        "total_rounds": total_rounds,
        "num_clients_per_round": clients_per_round
    }
    cs_extra = cfg['FL_config'].get('CS_args')
    if isinstance(cs_extra, dict):
        base_CS_args.update(cs_extra)

    base_Agg_args = {}
    agg_extra = cfg['FL_config'].get('Agg_args')
    if isinstance(agg_extra, dict):
        base_Agg_args.update(agg_extra)

    client_test_cfg       = cfg['ML_config']['test_config']
    client_minibatch_time = cfg['client_config']['minibatch_time']

    # Iterate over batch sizes: 32, 64, 80
    for bs in [32, 64, 80]:
        print(f"\n=== Experiment with train_bs={bs} ===")
        train_cfg = dict(cfg['ML_config']['train_config'])
        train_cfg['train_bs'] = bs

        exp_name = f"{CS_algo}_{model_id}_{dataset_id}_bs{bs}"
        out_dir  = os.path.join(save_root, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # Setup logging
        log_file = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_bs{bs}.log"
        logging.basicConfig(
            filename=os.path.join(out_dir, log_file),
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S"
        )
        logger = logging.getLogger("fedexp")
        logger.info(f"Started CPU-only experiment with train_bs={bs}")

        # Initialize clients
        clients = []
        for cid in range(total_clients):
            client = Client(
                logger=logger,
                cid=cid,
                device=device,
                model_class=ModelCls,
                model_args=model_args,
                data_path=data_dir,
                dataset_id=dataset_id,
                train_batch_size=bs,
                test_batch_size=client_test_cfg['test_bs'],
                minibatch_time=client_minibatch_time
            )
            # load with weights_only=False
            ckpt = torch.load(init_path, map_location=device, weights_only=False)
            client.model.load_state_dict(ckpt.state_dict())
            clients.append(client)

        # Initialize server
        server = Server(
            logger=logger,
            device=device,
            model_class=ModelCls,
            model_args=model_args,
            data_path=data_dir,
            dataset_id=dataset_id,
            test_batch_size=client_test_cfg['test_bs']
        )
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        server.model.load_state_dict(ckpt.state_dict())

        # Resolve dynamic functions
        cs_fn, cs_args  = setup_client_selection(server,  CS_algo,  base_CS_args.copy())
        agg_fn, agg_args= setup_aggregation(server,       Agg_algo, base_Agg_args.copy())

        # Run federated rounds
        main(
            total_rounds,
            seed,
            clients,
            cs_fn,
            cs_args,
            agg_fn,
            agg_args,
            train_cfg,
            client_test_cfg
        )

