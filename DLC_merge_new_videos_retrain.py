import deeplabcut
import ruamel.yaml as yaml
import os

# Path to the existing project configuration file
config_path = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\config.yaml'

# Step 1: Create a new training dataset to ensure all labeled data is considered
deeplabcut.create_training_dataset(config_path, num_shuffles=1)

# Step 2: Ensure the init_weights path is correctly set in pose_cfg.yaml
pose_cfg_path = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\dlc-models\iteration-0\xray_rat_hindlimb_cam1Nov17-trainset95shuffle1\train\pose_cfg.yaml'
with open(pose_cfg_path) as f:
    pose_cfg = yaml.safe_load(f)

pose_cfg['init_weights'] = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\dlc-models\iteration-0\xray_rat_hindlimb_cam1Nov17-trainset95shuffle1\train\snapshot-316000'

with open(pose_cfg_path, 'w') as f:
    yaml.safe_dump(pose_cfg, f)

# Step 3: Train the network for 2000 iterations, continuing from the previous snapshot
deeplabcut.train_network(config_path, shuffle=1, displayiters=100, saveiters=100, maxiters=50000)

# Step 4: Export the trained model
deeplabcut.export_model(config_path, shuffle=1, trainingsetindex=0)

# Ensure the export directory structure
export_dir = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\exported-models\DLC_xray_rat_hindlimb_cam1_resnet_50_iteration-0_shuffle-1'
if not os.path.exists(export_dir):
    raise FileNotFoundError(f"The export directory {export_dir} does not exist.")

# Step 5: Evaluate the retrained network
deeplabcut.evaluate_network(config_path, plotting=True)

# Path to the new video
new_video_path = [r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\videos\2023-02-07_15-17_Evt01-Camera1.avi']

# Step 6: Analyze the new video using the exported model and save the tracking data as CSV
# deeplabcut.analyze_videos(config_path, new_video_path, videotype='avi', save_as_csv=True, modelprefix=export_dir)
deeplabcut.analyze_videos(config_path, new_video_path, videotype='avi', save_as_csv=True)

# Step 7: Create labeled video with tracked points
deeplabcut.create_labeled_video(config_path, new_video_path, videotype='avi', draw_skeleton=False, save_frames=False, codec='mp4v')
