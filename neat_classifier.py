"""
NEAT (NeuroEvolution of Augmenting Topologies) Classifier
Multi-task email intelligence using neuroevolution
Author: Sneha Vasudev
"""

import neat
import time
import numpy as np
import pickle
import tempfile
import os
from typing import Dict, Tuple, Callable
from sklearn.metrics import f1_score
from pathlib import Path
from config import TASK_SIZES, TASK_NAMES, TASK_WEIGHTS, FITNESS_WEIGHTS, NEAT_CONFIG


class NEATClassifier:
    """NEAT-based multi-task classifier"""
    
    def __init__(self, n_inputs: int, task_sizes: list = None):
        """
        Initialize NEAT classifier
        
        Args:
            n_inputs: Number of input features
            task_sizes: List of output sizes for each task [3, 3, 3, 1]
        """
        self.n_inputs = n_inputs
        self.task_sizes = task_sizes or TASK_SIZES
        self.n_outputs = sum(self.task_sizes)
        
        self.winner = None
        self.config = None
        self.fitness_history = []
        self.weight_history = []
        self.generation_count = 0
        
    def write_neat_config(self, config_path: str = 'neat.cfg'):
        """Write NEAT configuration file"""
        cfg = f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.95
pop_size              = {NEAT_CONFIG['pop_size']}
reset_on_extinction   = True
no_fitness_termination  = False

[DefaultGenome]
num_inputs            = {self.n_inputs}
num_outputs           = {self.n_outputs}
num_hidden            = 0
feed_forward          = True
initial_connection    = full

activation_default      = {NEAT_CONFIG['activation_default']}
activation_mutate_rate  = {NEAT_CONFIG['activation_mutate_rate']}
activation_options      = {NEAT_CONFIG['activation_options']}

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_attr_mutation      = gaussian 0.5 0.05 -30.0 30.0
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

response_attr_mutation  = gaussian 0.0 0.1 -30.0 30.0
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

connection_add_prob     = 0.5
connection_delete_prob  = 0.5

conn_weight_mutation    = gaussian 0.5 0.5 -30.0 30.0
conn_weight_init_mean   = 0.0
conn_weight_init_stdev  = 1.0
conn_weight_max_value   = 30.0
conn_weight_min_value   = -30.0
conn_weight_mutate_power = 0.5
conn_weight_mutate_rate = 0.8
conn_weight_replace_rate = 0.1

feed_forward_only       = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.4

enabled_default         = True
enabled_mutate_rate     = 0.01

node_add_prob           = 0.2
node_delete_prob        = 0.2

network_depth_coefficient = 1.0
network_complexity_coefficient = 0.0

single_structural_mutation = False
structural_mutation_surer = False

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        with open(config_path, 'w') as f:
            f.write(cfg)
        return config_path
    
    @staticmethod
    def safe_novelty(w_vec: np.ndarray, history: list) -> float:
        """
        Calculate novelty reward for genetic diversity
        Compares weight vectors of potentially different lengths safely
        """
        if not history:
            return 0.0
        
        distances = []
        for wh in history[-10:]:  # Compare with last 10 genomes
            min_len = min(len(w_vec), len(wh))
            if min_len > 0:
                dist = np.linalg.norm(w_vec[:min_len] - wh[:min_len])
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def create_fitness_function(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray
    ) -> Callable:
        """
        Create multi-objective fitness function
        
        Fitness = 0.5 * F1_multi + 0.2 * Novelty - 0.3 * Latency_penalty
        """
        
        def eval_genomes(genomes, config):
            """Evaluate all genomes in population"""
            
            for genome_id, genome in genomes:
                # Build neural network from genome
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                
                # Forward pass on validation set
                predictions = []
                inference_times = []
                
                for x in X_val[:1000]:  # Use subset for speed
                    t0 = time.time()
                    output = net.activate(x)
                    inference_times.append(time.time() - t0)
                    predictions.append(output)
                
                predictions = np.array(predictions)
                
                # Extract task-specific predictions
                task_preds = []
                idx = 0
                for task_size in self.task_sizes[:len(Y_val[0])]:
                    task_pred = predictions[:, idx:idx+task_size]
                    if task_size > 1:
                        task_pred = np.argmax(task_pred, axis=1)
                    else:
                        task_pred = (task_pred[:, 0] > 0.5).astype(int)
                    task_preds.append(task_pred)
                    idx += task_size
                
                # Calculate weighted multi-task F1
                f1_scores = []
                for i, task_name in enumerate(TASK_NAMES):
                    if i < len(task_preds) and len(task_preds[i]) > 0:
                        f1 = f1_score(Y_val[:len(task_preds[i]), i], task_preds[i],
                                     average='weighted', zero_division=0)
                        f1_scores.append(f1)
                
                f1_multi = np.average(f1_scores, weights=TASK_WEIGHTS[:len(f1_scores)])
                
                # Novelty reward
                w_vec = np.concatenate([
                    genome.connections[cg].weight 
                    for cg in genome.connections.values()
                ])
                novelty = self.safe_novelty(w_vec, self.weight_history)
                
                # Latency penalty
                avg_latency = np.mean(inference_times)
                latency_penalty = avg_latency * 1000  # Convert to ms
                
                # Combined fitness
                fitness = (
                    FITNESS_WEIGHTS['f1_score'] * f1_multi +
                    FITNESS_WEIGHTS['novelty'] * novelty +
                    FITNESS_WEIGHTS['latency_penalty'] * latency_penalty
                )
                
                genome.fitness = fitness
                self.weight_history.append(w_vec)
                self.fitness_history.append(fitness)
            
            self.generation_count += 1
        
        return eval_genomes
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        generations: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train NEAT classifier
        
        Args:
            X_train: Training features (n_samples, n_features)
            Y_train: Training labels (n_samples, 4) for multi-task
            generations: Number of generations (default from config)
            verbose: Print progress
        
        Returns:
            Training results dictionary
        """
        
        if generations is None:
            generations = NEAT_CONFIG.get('generations', 50)
        
        # Split train/val
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:].astype(np.float32)
        X_train = X_train[:split_idx].astype(np.float32)
        Y_val = Y_train[split_idx:]
        Y_train = Y_train[:split_idx]
        
        if verbose:
            print(f"\n[NEAT Training]")
            print(f"  Input features: {self.n_inputs}")
            print(f"  Output nodes: {self.n_outputs} (tasks: {self.task_sizes})")
            print(f"  Train samples: {len(X_train)}")
            print(f"  Val samples: {len(X_val)}")
            print(f"  Generations: {generations}")
        
        # Create NEAT config
        config_path = self.write_neat_config()
        self.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create population
        p = neat.Population(self.config)
        
        # Add reporters
        p.add_reporter(neat.StdOutReporter(verbose))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        # Train
        t0 = time.time()
        fitness_fn = self.create_fitness_function(X_train, Y_train, X_val, Y_val)
        self.winner = p.run(fitness_fn, generations)
        train_time = time.time() - t0
        
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
        
        results = {
            'winner': self.winner,
            'fitness': self.winner.fitness,
            'nodes': len(self.winner.nodes),
            'connections': len(self.winner.connections),
            'training_time': train_time,
            'generations': self.generation_count,
            'fitness_history': self.fitness_history
        }
        
        if verbose:
            print(f"\n✓ Training complete in {train_time:.2f}s")
            print(f"  Best fitness: {self.winner.fitness:.4f}")
            print(f"  Nodes: {len(self.winner.nodes)}")
            print(f"  Connections: {len(self.winner.connections)}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with trained NEAT network
        
        Returns:
            predictions: Multi-task predictions (n_samples, 4)
            probabilities: Raw network outputs
        """
        if self.winner is None:
            raise ValueError("Model not trained. Call train() first.")
        
        net = neat.nn.FeedForwardNetwork.create(self.winner, self.config)
        
        predictions = []
        probabilities = []
        
        for x in X:
            output = net.activate(x.astype(np.float32))
            probabilities.append(output)
            
            # Decode multi-task outputs
            task_preds = []
            idx = 0
            for task_size in self.task_sizes:
                if task_size > 1:
                    task_pred = np.argmax(output[idx:idx+task_size])
                else:
                    task_pred = int(output[idx] > 0.5)
                task_preds.append(task_pred)
                idx += task_size
            
            predictions.append(task_preds)
        
        return np.array(predictions), np.array(probabilities)
    
    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'winner': self.winner,
                'config': self.config,
                'n_inputs': self.n_inputs,
                'task_sizes': self.task_sizes
            }, f)
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.winner = data['winner']
            self.config = data['config']
            self.n_inputs = data['n_inputs']
            self.task_sizes = data['task_sizes']


if __name__ == "__main__":
    print("NEAT classifier module imported successfully")
