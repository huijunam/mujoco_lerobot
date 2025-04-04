# import mujoco
# import mujoco_viewer
# import numpy as np
# from datasets import load_dataset

# train_dataset = load_dataset("xhaka3456/so100_test_0329_1", split="train")

# # 마지막 에피소드 (`episode_000008`) 필터링
# last_episode_data = [row for row in train_dataset if row['episode_index'] == 8]  
# observations = np.array([row['observation.state'] for row in last_episode_data])  # 모든 timestamp 가져오기

# # MuJoCo 모델 로드
# model = mujoco.MjModel.from_xml_path('../mujoco_menagerie/trs_so_arm100/scene.xml')
# data = mujoco.MjData(model)

# viewer = mujoco_viewer.MujocoViewer(model, data)

# # 노란색 구체를 위한 Marker 추가
# marker_id = len(data.site_xpos)  # 새로운 마커를 위한 ID 할당

# for obs in observations:
#     if viewer.is_alive:
#         data.qpos[:] = obs[:6]  # 6축 로봇의 joint 위치 적용
#         mujoco.mj_step(model, data)  # 물리 엔진 한 스텝 실행
        
#         viewer.render()  # 화면 렌더링
#     else:
#         break

# 종료
# viewer.close()

import mujoco
import mujoco_viewer
import numpy as np
from datasets import load_dataset

# 데이터셋 로드
train_dataset = load_dataset("xhaka3456/so100_test_0329_1", split="train")

# 마지막 에피소드 필터링
last_episode_data = [row for row in train_dataset if row['episode_index'] == 8]  
observations = np.array([row['observation.state'] for row in last_episode_data])  

# MuJoCo 모델 로드
model = mujoco.MjModel.from_xml_path('../mujoco_menagerie/trs_so_arm100/scene.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

for obs in observations:
    if not viewer.is_alive:
        break

    # 1️⃣ 관절 상태 업데이트 (degree → radian 변환)
    data.qpos[:] = np.radians(obs[:6])  
    mujoco.mj_forward(model, data)  # 물리 엔진 상태 업데이트

    # 2️⃣ 각 축의 위치 얻기 (base 제외한 6개 조인트)
    joint_names = ["Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"]
    joint_positions = [data.xpos[model.body(joint).id] for joint in joint_names]

    viewer.render()  # 화면 렌더링

viewer.close()
