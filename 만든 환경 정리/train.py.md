``` python
import os
import pickle
import shutil

from env import HoverEnv  # HoverEnv 클래스 import
import genesis as gs  # genesis 라이브러리 import


# 학습 설정을 반환하는 함수
# 모델 설정해야 하는 공간

# --------------------------------- #
# 환경 설정을 반환하는 함수
def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # 종료 조건
        "termination_if_roll_greater_than": 180,
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # 초기 위치 및 자세
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # 시각화 설정
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "urdf_file": [
            "/root/Docker_src/실험/drones_urdf/drones/cf2x.urdf",
            "/root/Docker_src/실험/drones_urdf/drones/cf2p.urdf",
            "/root/Docker_src/실험/drones_urdf/drones/racer.urdf",
        ],
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg





def main():
    # 직접 변수에 값 할당 (ArgumentParser 미사용)
    exp_name = "drone-hovering"
    vis = False
    num_envs = 8192
    max_iterations = 300

    gs.init(logging_level="error")  # genesis 초기화 (로그 레벨 설정)

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name, max_iterations)

    # 기존 로그 디렉토리 삭제 후 새로 생성
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 시각화 옵션이 활성화된 경우 타겟 시각화 활성화
    if vis:
        env_cfg["visualize_target"] = True

    # 환경 생성
    env = HoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=vis,
    )



if __name__ == "__main__":
    main()

```