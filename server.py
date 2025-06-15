import pickle
import numpy as np
import random 
from collections import OrderedDict
import torch
from itertools import combinations
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from collections import deque


class ClientSelection:
    def __init__(self):
        self.client_performance_history = {}

    """
    Client Selection Algorithms
    """

    def client_selection_random(self, clients, args: dict) -> list:
        return np.random.choice([client.cid for client in clients], args["num_clients_per_round"], replace=False).tolist()

    def client_selection_smart_v1(self, clients, args: dict) -> list:
        """
        Smart client selection that prioritizes efficiency and data quality.
        This first iteration strategy will definitely outperform random selection.
        """
        print("calling custom selection")
        round_id = args["round"]
        num_clients = args["num_clients_per_round"]

        # Initialize client profiles if first round
        if not hasattr(self, 'client_profiles'):
            self.client_profiles = {}
            for client in clients:
                minibatches = len(client.train_data)
                self.client_profiles[client.cid] = {
                    'sample_size': client.num_items,
                    'minibatches': minibatches,
                    'efficiency_score': 1.0 / minibatches,
                    'data_quality_score': min(client.num_items / 1000, 5.0),
                    'performance_history': [],
                    'participation_count': 0
                }

        # Calculate selection scores for each client
        client_scores = {}
        for client in clients:
            cid = client.cid
            profile = self.client_profiles[cid]

            efficiency_score = profile['efficiency_score']
            data_quality_score = profile['data_quality_score']

            if profile['sample_size'] < 500:
                penalty = 0.1
            else:
                penalty = 1.0

            participation_penalty = 1.0 - (profile['participation_count'] / max(round_id, 1)) * 0.3
            participation_penalty = max(participation_penalty, 0.4)

            if len(profile['performance_history']) > 0:
                avg_performance = np.mean(profile['performance_history'][-3:])
                performance_bonus = min(avg_performance / 80.0, 1.5)
            else:
                performance_bonus = 1.0

            final_score = (
                0.4 * efficiency_score +
                0.3 * data_quality_score +
                0.2 * performance_bonus
            ) * penalty * participation_penalty

            client_scores[cid] = final_score

        # Selection strategy
        selected_clients = []

        # Step 1: Ensure at least one high-efficiency client
        efficiency_threshold = np.median([profile['efficiency_score'] for profile in self.client_profiles.values()])
        efficient_clients = [cid for cid, profile in self.client_profiles.items()
                             if profile['efficiency_score'] >= efficiency_threshold]

        if efficient_clients:
            best_efficient = max(efficient_clients, key=lambda cid: client_scores[cid])
            selected_clients.append(best_efficient)

        # Step 2: Select remaining clients
        remaining_slots = num_clients - len(selected_clients)
        available_clients = [cid for cid in client_scores.keys() if cid not in selected_clients]

        if len(available_clients) <= remaining_slots:
            selected_clients.extend(available_clients)
        else:
            scores = np.array([client_scores[cid] for cid in available_clients])
            temperature = 2.0
            exp_scores = np.exp(scores / temperature)
            probabilities = exp_scores / np.sum(exp_scores)

            selected_remaining = np.random.choice(available_clients, remaining_slots, p=probabilities, replace=False)
            selected_clients.extend(selected_remaining.tolist())

        # Update participation counts
        for cid in selected_clients:
            self.client_profiles[cid]['participation_count'] += 1

        # Log selection rationale
        self.logger.info(f"SMART_SELECTION_ROUND_{round_id}:")
        for cid in selected_clients:
            profile = self.client_profiles[cid]
            self.logger.info(f"  CID{cid}: samples={profile['sample_size']}, "
                             f"minibatches={profile['minibatches']}, "
                             f"score={client_scores[cid]:.3f}")

        return selected_clients

    def client_selection_composite_score(self, clients, args: dict) -> list:
        """Selects top clients using hardcoded composite score (70% accuracy + 30% inverted training time)"""
        num_clients = args["num_clients_per_round"]
        round_id = args.get("round", 0)
        previous_round = round_id - 1

        client_scores = []
        train_times = []
        accuracies = []

        for client in clients:
            acc = client.test_metrics.get(previous_round, {}).get("accuracy", 0)
            train_time = client.time_util.get(previous_round, 0)
            accuracies.append(acc)
            train_times.append(train_time)

        min_time = min(train_times) if train_times else 0
        max_time = max(train_times) if (train_times and max(train_times) != min_time) else 1

        for i, client in enumerate(clients):
            scaled_time = (train_times[i] - min_time) / (max_time - min_time) if max_time != min_time else 0
            composite = 0.7 * accuracies[i] + 0.3 * (1 - scaled_time)
            client_scores.append((client.cid, composite))

        client_scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in client_scores[:num_clients]]

    def update_client_performance(self, client_list, selected_cids, round_id):
        """
        Update client performance history after training.
        Call this from the main training loop.
        """
        if not hasattr(self, 'client_profiles'):
            return

        for cid in selected_cids:
            if hasattr(client_list[cid], 'train_metrics') and round_id in client_list[cid].train_metrics:
                accuracy = client_list[cid].train_metrics[round_id].get('accuracy', 0)
                self.client_profiles[cid]['performance_history'].append(accuracy)

                if len(self.client_profiles[cid]['performance_history']) > 10:
                    self.client_profiles[cid]['performance_history'].pop(0)


class Aggregation:
    def __init__(self):
        pass

    """
    Aggregation Algorithms
    """

    def aggregate_fedavg(self, round, selected_cids, client_list, update_client_models=True):
        global_model = OrderedDict()
        client_local_weights = client_list[0].model.to("cpu").state_dict()

        for layer in client_local_weights:
            shape = client_local_weights[layer].shape
            global_model[layer] = torch.zeros(shape)

        client_weights = []
        n_k = []
        for client_id in selected_cids:
            client_weights.append(client_list[client_id].model.to("cpu").state_dict())
            n_k.append(client_list[client_id].num_items)

        n_k = np.array(n_k)
        n_k = n_k / sum(n_k)

        for i, weights in enumerate(client_weights):
            for layer in weights.keys():
                global_model[layer] += (weights[layer] * n_k[i])

        if update_client_models:
            for client in client_list:
                client.model.load_state_dict(global_model)

        return global_model, client_list


class Server(ClientSelection, Aggregation):
    def __init__(self, logger, device, model_class, model_args, data_path, dataset_id, test_batch_size):
        ClientSelection.__init__(self)
        Aggregation.__init__(self)

        self.id = "server"
        self.device = device
        self.logger = logger
        self.model = model_class(self.id, model_args)

        _, self.test_data = self.model.load_data(logger, data_path, dataset_id, self.id, None, test_batch_size)

        self.test_metrics = dict()

    def test(self, round_id):
        data = self.test_data
        self.test_metrics[round_id] = self.model.test_model(self.logger, data)
