#%matplotlib inline
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import pickle
import os
import tensorflow as tf

model_fold = 'models/'

models = os.listdir(model_fold)
print(models)

import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

def visualize(im, detections, poses3d, poses2d, edges, fig):
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(im)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    return fig

def fig_to_np(_fig, dpi=60):
    io_buf = io.BytesIO()    
    _fig.savefig(io_buf, format="png", dpi=dpi)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)
    io_buf.close()
    return img

# load videos!
import numpy as np
from tqdm import tqdm
import time
import cv2

#vid_name_back = '/media/tobi/Elements/thesis_videos/gopro/25_exp.MP4'
#vid_name_front = '/media/tobi/Elements/thesis_videos/alpha/25_exp.MP4'
vid_name_front_small = '25_exp.mp4'
vid_name = vid_name_front_small

bss = [8, 16, 32, 64]

WIDTH, HEIGHT = 1200, 600

#mod = 'metrabs_eff2l_y4_360' #ALL poses?
#mod = 'metrabs_mob3s_y4t' #fast?
#mod = 'metrabs_eff2l_y4' #biggest

for mod in models:
    t_load = time.time()
    model = tf.saved_model.load(os.path.join(model_fold, mod))
    t_load = time.time() - t_load
    print('loaded in', t_load)

    for bs in bss:
        print(mod)
        vid_file = 'full/%s_bs=%d.mp4'%(mod, bs)
        doc_file = 'full/%s_bs=%d.txt'%(mod, bs)
        res_file = 'full/%s_bs=%d.pkl'%(mod, bs)
        
        print(res_file)
        if os.path.exists(res_file):
            continue

        with open(doc_file, 'w') as fh:
            fh.write('load: %.2fs\n'%t_load)
        
        cap = cv2.VideoCapture(vid_name)
        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_vid = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (WIDTH, HEIGHT))

        frames = []
        results = []
        ts = []
        t_full = time.time()
        for i in tqdm(range(vid_len)):
            ret, frame = cap.read()
            if ret is None:
                break
            if i < 2*30*60:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) == bs:
                t = time.time()
                pred = model.detect_poses_batched(np.array(frames))
                t = time.time() - t
                with open(doc_file, 'a') as fh:
                    fh.write('%f\n'%(t*1000))

                results.append(pred)

                # draw to video!
                fig = plt.figure(figsize=(20, 10))

                t = time.time()
                for j, frame in enumerate(frames):
                    visualize(
                        frame, 
                        pred['boxes'][j].numpy(),
                        pred['poses3d'][j].numpy(),
                        pred['poses2d'][j].numpy(),
                        model.per_skeleton_joint_edges['smpl+head_30'].numpy(), fig)
                    img = fig_to_np(fig)
                    fig.clf()
                    out_vid.write(img)
                plt.close('all')
                t = time.time()
                if len(results)*bs >= 10000:
                    break

                frames = []
        t_full = time.time() - t_full
        with open(doc_file, 'a') as fh:
            fh.write('full_time: %.2fs\n'%t_full)
        print('full run took', t_full)

        print('save video')
        out_vid.release()
        pickle.dump(results, open(res_file, 'wb'))
