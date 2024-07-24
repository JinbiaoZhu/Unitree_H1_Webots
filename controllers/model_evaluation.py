"""
Run Evaluation.
"""
import torch
from torch.distributions.normal import Normal

from base_declaration import DTYPE
from PPO_config import ppo_config
from UnitreeH1StandingEnv import UnitreeH1StandingV0, UnitreeH1StandingV1

if __name__ == '__main__':

    # env = UnitreeH1StandingV0()
    env = UnitreeH1StandingV1()
    env.metadata["mode"] = "test"

    num_episodes = 10

    # load policy here
    policy = torch.load("./ckpts/ppo/actor_20240724170556.pth")

    env.record_movie(True, "/media/isaacgym/extend/isaacgym/UnitreeH1_RLTraining/controllers/record/ppo"
                           "/actor_20240724170556.mp4")
    for i in range(num_episodes):

        state = env.reset()

        while env.render() != -1:
            state = torch.as_tensor(state, dtype=DTYPE, device=ppo_config["device"])
            # No deterministic
            mean, logstd = policy(state)
            dist = Normal(mean, logstd.exp())
            action = dist.sample().data.cpu().numpy()
            next_state, reward, done, _ = env.step(action)

            state = next_state

            if done:
                break

    env.record_movie(False, "/media/isaacgym/extend/isaacgym/UnitreeH1_RLTraining/controllers/record/ppo"
                            "/actor_20240724170556.mp4")
    env.close()
