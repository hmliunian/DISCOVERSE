import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import time
import os
import argparse
import traceback
import multiprocessing as mp
import mujoco
import mujoco.viewer

from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg, args):
        super().__init__(config)
        self.doors=args.doors
        self.doors_handle="hinge_door_"+str(args.doors)+"_handle"
        self.door_index=self.nj+self.doors
        # self.arm_ori_pos = self.mj_model.body("arm_pose").pos.copy()
        # self.max_drawer_rate=args.max_drawer
        # self.min_drawer_rate=args.min_drawer
        self.random_position=args.random_position
        self.ab_arm_pos=[0.25, 0.454, -0.23]
        # self.ab_arm_pos=[0.95, 0.454, -0.23]
        self.mj_model.body("arm_pose").pos[:3] = self.mj_model.body("hinge_door_" + str(self.doors)).pos[:3]  + self.ab_arm_pos
        self.arm_ori_pos = self.mj_model.body("arm_pose").pos.copy()

    def domain_randomization(self):
        # self.mj_data.qpos[self.drawer_index] += np.random.uniform(self.min_drawer_rate,self.max_drawer_rate)
        # self.origin_drawer_pos = self.mj_data.qpos[self.drawer_index]
        # self.mj_model.body("arm_pose").pos[:3] = self.mj_model.body("drawer_" + str(self.drawer)).pos[:3] + self.ab_arm_pos
        # self.mj_data.qpos[self.nj+1] += 2.*(np.random.random()-0.5) * 0.05
        self.origin_pos=self.mj_data.qpos.copy()
        if self.random_position:
            random_bias = 2.*(np.random.random(3) - 0.5) * 0.20
            random_bias[0] /= 4
            self.mj_model.body("arm_pose").pos[:3] = self.arm_ori_pos[:3] + random_bias
        # self.mj_data.site(self.drawers_handle).xpos[0] += 2

        # mujoco.mj_forward(self.mj_model, self.mj_data)

    def check_success(self):
        # return (self.mj_data.qpos[9] > 0.15)
        # return (self.mj_data.qpos[9] > 0.3)
        return self.get_joint_position("hinge_joint_"+str(self.doors)) < -1.3
        # diff=np.sum(np.square(self.mj_data.qpos-self.origin_pos))
        # return diff > 7.5
        # return False


    def set_site_tmat(mj_data, site_name):
        tmat = np.eye(4)
        tmat[:3,:3] = mj_data.site(site_name).xmat.reshape((3,3))
        tmat[:3,3] = mj_data.site(site_name).xpos
        return tmat

cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"] = "scene/lab3/room_with_empty_carbinet.ply"
cfg.gs_model_dict["drawer_1"]   = "hinge/drawer1.ply"
cfg.gs_model_dict["drawer_2"]   = "hinge/drawer2.ply"
cfg.gs_model_dict["drawer_3"]   = "hinge/drawer3.ply"
cfg.gs_model_dict["drawer_4"]   = "hinge/drawer4.ply"
cfg.gs_model_dict["drawer_5"]   = "hinge/drawer5.ply"
cfg.gs_model_dict["drawer_6"]   = "hinge/drawer6.ply"
cfg.gs_model_dict["hinge_door_1"]   = "hinge/door1.ply"
cfg.gs_model_dict["hinge_door_2"]   = "hinge/door2.ply"
cfg.gs_model_dict["hinge_door_3"]   = "hinge/door3.ply"
cfg.gs_model_dict["hinge_door_4"]   = "hinge/door4.ply"
cfg.gs_model_dict["hinge_door_5"]   = "hinge/door5.ply"
# cfg.gs_model_dict["hinge_door_6"]   = "hinge/door6.ply"
cfg.init_qpos[:] = [0,0,0,  0,  0, 0,  0.]
# cfg.init_qpos[:] = [-0.2, 0,  0,  0,  0, 0,  0.]
cfg.mjcf_file_path = "mjcf/tasks_airbot_play/open_lots_of_hinge_doors.xml"
cfg.obj_list     = ["hinge_door_1","hinge_door_2","hinge_door_3","hinge_door_4","hinge_door_5",
                    "drawer_1","drawer_2","drawer_3","drawer_4","drawer_5","drawer_6"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    # parser.add_argument("--max_drawer", type=float, default=0, help="Maximum drawer open rate (0.0 to 0.3). Recommended: 0.05.")
    # parser.add_argument("--min_drawer", type=float, default=0, help="Minimum drawer open rate (0.0 to 0.3). Recommended: 0.")
    parser.add_argument("--random_position", action="store_true", help="Randomly initialize airbot's position.")
    parser.add_argument("--doors", type=int, default=1, help="select drawers(from 1 to 5)")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg, args)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
        
    arm_ik = AirbotPlayIK()

    if args.doors == 1:
        trmat = Rotation.from_euler("xyz", [np.pi/2, 0., 0], degrees=False).as_matrix()
    else:
        trmat = np.eye(3)

    stm = SimpleStateMachine()
    stm.max_state_cnt =12
    max_time = 10.0 #s

    action = np.zeros(7)
    # drawers="drawer_"+str(args.drawers)+"_handle"
    doors="hinge_door_"+str(args.doors)+"_handle"
    move_speed = 2.0
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger(): 
                tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))
                if stm.state_idx == 0:
                    # use the last goal pose ik result for reference joints in case of wrist flip (just decrease the risk but not eliminate it)
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.35, 0.35, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + 0.1 * tmat_handle[:3, 0]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[4] *= 10 # make sure the elbow is up
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.target_control[:6])
                    sim_node.target_control[6] = 1
                    # move_speed = 1.5
                elif stm.state_idx == 1: # 伸到把手位置
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    # move_speed = 1.0  
                    # sim_node.target_control[5] = 1.57
                elif stm.state_idx == 2: # 抓住把手
                    
                    sim_node.target_control[6] = 0
                elif stm.state_idx == 3: # 抓稳把手 sleep 0.5s
                    sim_node.delay_cnt = int(0.5/sim_node.delta_t)
                elif stm.state_idx == 4: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + np.array([-0.0519, 0.013, 0])
                    # 0.1 * tmat_handle[:3, 0]+ -0.1 * tmat_handle[:3, 1]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + np.array([-0.0484, 0.0229, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0430, 0.0319, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0359, 0.0396, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 8: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0275, 0.0459, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    
                elif stm.state_idx == 9: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0180, 0.0504, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    
                elif stm.state_idx == 10: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0079, 0.0529, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    
                elif stm.state_idx == 11: # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, doors)
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] +  np.array([-0.0026, 0.0535, 0])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 12: # 松开把手
                    sim_node.target_control[6] = 1
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            print(f"{data_idx} Mistake")
            traceback.print_exc()
            sim_node.reset()
        
        # if args.doors==1:
        #     sim_node.target_control[5] = 1.57
        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop('img')
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            for encoder in encoders.values():
                encoder.close()
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")
                for encoder in encoders.values():
                    encoder.remove_av_file()

            sim_node.reset()

# echo "export http_proxy='http://192.168.211.162:7890'" >> ~/.bashrc
# echo "export https_proxy='https://192.168.211.162:7890'" >> ~/.bashrc
# source ~/.bashrc
