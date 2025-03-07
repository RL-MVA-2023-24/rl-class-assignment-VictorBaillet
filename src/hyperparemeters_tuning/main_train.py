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


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent_config_path = os.path.join('src/config/best_5.yaml')
    
    with open(agent_config_path) as f:
        agent_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    agent = ProjectAgent()
    agent.train(agent_config_dict, use_wandb=False)
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")

    
    #with open(file="score.txt", mode="w") as f:
    #    f.write(f"{score_agent}\n{score_agent_dr}")


