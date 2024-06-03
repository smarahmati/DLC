import deeplabcut
import ruamel.yaml as yaml

# Path to the existing project configuration file
config_path = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\config.yaml'

# Step 1: Create a new training dataset to ensure all labeled data is considered
deeplabcut.create_training_dataset(config_path, num_shuffles=1)

# Step 2: Ensure the init_weights path is correctly set in pose_cfg.yaml
pose_cfg_path = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\dlc-models\iteration-1\xray_rat_hindlimb_cam1Nov17-trainset95shuffle1\train\pose_cfg.yaml'
with open(pose_cfg_path) as f:
    pose_cfg = yaml.safe_load(f)

pose_cfg['init_weights'] = r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\dlc-models\iteration-1\xray_rat_hindlimb_cam1Nov17-trainset95shuffle1\train\snapshot-3000'

with open(pose_cfg_path, 'w') as f:
    yaml.safe_dump(pose_cfg, f)

# Step 3: Train the network for 2000 iterations, continuing from the previous snapshot
deeplabcut.train_network(config_path, shuffle=1, displayiters=100, saveiters=200, maxiters=1000)

# Step 4: Evaluate the retrained network
deeplabcut.evaluate_network(config_path, plotting=True)

# Path to the new video
new_video_path = [r'E:\DLCModel_Retrain\ProjectFiles\xray_rat_hindlimb_cam1-Nathan-2021-11-17\videos\2023-02-07_15-17_Evt01-Camera1.avi']

# Step 5: Analyze the new video using the retrained network
deeplabcut.analyze_videos(config_path, new_video_path, videotype='avi')

# Step 6: Create a labeled video
deeplabcut.create_labeled_video(config_path, new_video_path, videotype='avi', draw_skeleton=False, save_frames=False, codec='mp4v')

# Step 7: Save the tracking results as a CSV file
deeplabcut.create_video_with_all_detections(config_path, new_video_path, save_as_csv=True)
