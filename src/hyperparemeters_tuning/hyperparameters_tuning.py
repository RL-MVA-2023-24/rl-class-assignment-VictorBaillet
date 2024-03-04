import wandb
import random
import os
import numpy as np
import torch
import yaml

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def compute_train_function(agent_config_dict):
    def train():
        config_dict = agent_config_dict
        with wandb.init(config=None):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            config_dict = update_config(config_dict, config)

            agent = ProjectAgent()
            agent.train(config_dict)
            #agent.load()
            seed_everything(seed=42)
            score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
            score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)

            wandb.log({'score_agent' : score_agent})
            wandb.log({'score_agent_dr' : score_agent_dr})
            
    return train


def update_config(config_dict, config):
    for key in config_dict:
        if key in config:
            config_dict[key] = config[key]
        elif isinstance(config_dict[key], dict):
            config_dict[key] = update_config(config_dict[key], config)
    
    return config_dict


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent_config_path = os.path.join('src/config/init_agent.yaml')
    sweep_config_path = os.path.join('src/config/sweep.yaml')
    
    with open(agent_config_path) as f:
        agent_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(sweep_config_path) as f:
        sweep_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config_dict, project="RL-MVA")

    train = compute_train_function(agent_config_dict)

    #wandb.agent(sweep_id, train, count=100)
    wandb.agent(sweep_id, train, count=100)
    
    #with open(file="score.txt", mode="w") as f:
    #    f.write(f"{score_agent}\n{score_agent_dr}")


