``` python
import argparse
import os
import pickle

import torch
from hover_env import HoverEnv
from hover_env import HoverEnv  # 앞서 정의한 HoverEnv 환경 코드 (드론 호버링 환경)
from model.sac import SACAgent  # SAC (Soft Actor-Critic) 에이전트 구현 모듈
from model.replay_buffer import ReplayBuffer  # 리플레이 버퍼 구현 모듈 (경험 저장소)

import genesis as gs



def main():
    # 명령행 인자 대신 직접 인자값 설정 (Jupyter 환경에서 실행)
    exp_name = "ex1) drone"       
    path = f"logs/{exp_name}"

    ckpt = 300                      
    # 저장된 state_dict를 불러옵니다.
    checkpoint_path = os.path.join(path, "actor.pth")
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))  # 또는 'cuda:0'

    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open(f"logs/{exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # 상태(관측) 차원과 행동 차원 설정
    state_dim = obs_cfg["num_obs"]         # 예: 17 차원 관측 벡터
    action_dim = env_cfg["num_actions"]      # 예: 4개의 액션 (프로펠러 제어)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 사용 가능한 디바이스 설정

    # SAC 관련 하이퍼파라미터 정의
    critic_cfg = {'hidden_dim': 256, 'hidden_depth': 3}  # 크리틱 네트워크 구성
    actor_cfg = {'hidden_dim': 256, 'hidden_depth': 3, 'log_std_bounds': [-20, 2]}  # 액터 네트워크 구성
    action_range = [-1, 1]      # 액션 값 범위
    discount = 0.99             # 감가율(할인율)
    init_temperature = 0.1      # 초기 온도 (탐사-활용 균형 조절)
    alpha_lr = 1e-4             # 온도 파라미터(alpha) 학습률
    alpha_betas = (0.9, 0.999)    # 온도 파라미터 최적화 모멘텀 값
    actor_lr = 1e-4             # 액터 학습률
    actor_betas = (0.9, 0.999)   # 액터 최적화 모멘텀 값
    actor_update_frequency = 2  # 액터 업데이트 빈도
    critic_lr = 1e-4            # 크리틱 학습률
    critic_betas = (0.9, 0.999)  # 크리틱 최적화 모멘텀 값
    critic_tau = 0.005          # 타깃 네트워크 업데이트 시 혼합 비율
    critic_target_update_frequency = 2  # 크리틱 타깃 네트워크 업데이트 빈도
    batch_size = 1024          # 미니배치 크기
    learnable_temperature = False  # 온도(alpha)를 학습할지 여부

    # SAC 에이전트 초기화: 관측, 행동 차원 및 하이퍼파라미터를 전달하여 SAC 에이전트 생성
    agent = SACAgent(
        obs_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        device=device,
        critic_cfg=critic_cfg,
        actor_cfg=actor_cfg,
        discount=discount,
        init_temperature=init_temperature,
        alpha_lr=alpha_lr,
        alpha_betas=alpha_betas,
        actor_lr=actor_lr,
        actor_betas=actor_betas,
        actor_update_frequency=actor_update_frequency,
        critic_lr=critic_lr,
        critic_betas=critic_betas,
        critic_tau=critic_tau,
        critic_target_update_frequency=critic_target_update_frequency,
        batch_size=batch_size,
        learnable_temperature=learnable_temperature
    )
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        for _ in range(max_sim_step):
            action_tensor, _ = agent.act(obs, True)
            obs, _, rews, dones, infos = env.step(torch.tensor(action_tensor))


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py

# Note
If you experience slow performance or encounter other issues 
during evaluation, try removing the --record option.
"""

```