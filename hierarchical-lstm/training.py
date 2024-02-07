import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

dataset_path = '../data/videos'
video_idxes = os.listdir(dataset_path)
video_idxes.remove('Info.txt')
video_idxes.remove('Info.txt~')

# Decision: using videos-level train-test split as specified by paper
train_video_idxes, test_video_idxes = train_test_split(video_idxes, train_size=0.66)

TRAIN_TIME_STEPS = 9

all_width_annotations = []
all_height_annotations = []
all_group_activity_annotations = set()
all_individual_activity_annotations = set()
for video_idx in video_idxes:
    annotations = open(f"{dataset_path}/{video_idx}/annotations.txt", 'r').readlines()
    for frame_annotation in annotations:
        main_frame_file_name, group_activity_name, *player_annotations = frame_annotation.strip().split(' ')
        all_group_activity_annotations.add(group_activity_name)

        while len(player_annotations) != 0:
            _, _, width_annotation, height_annotation, individual_activity_name, *player_annotations = player_annotations
            all_individual_activity_annotations.add(individual_activity_name)
            width_annotation = int(width_annotation)
            height_annotation = int(height_annotation)
            all_width_annotations.append(width_annotation)
            all_height_annotations.append(height_annotation)

XCEPTION_MIN_ACCEPTABLE_DIMENSION = 71
XCEPTION_resize_width = max(XCEPTION_MIN_ACCEPTABLE_DIMENSION, min(all_width_annotations))
XCEPTION_resize_height = max(XCEPTION_MIN_ACCEPTABLE_DIMENSION, min(all_height_annotations))


def load_video_idxes_from_volleyball_dataset(dataset_path, video_idxes, frames_per_tracklet, image_resize_height,
                                             image_resize_width):
    frames_before_main_frame = frames_per_tracklet // 2
    frames_after_main_frame = (
            frames_before_main_frame - 1) if frames_before_main_frame % 2 == 0 else frames_before_main_frame

    X_group_LSTM = []
    ragged_row_lengths = []
    Y_group_LSTM = []
    X_individual_LSTM = []
    Y_individual_LSTM = []

    for video_idx in video_idxes:
        annotations = open(f"{dataset_path}/{video_idx}/annotations.txt", 'r').readlines()
        for frame_annotation in annotations:
            main_frame_file_name, group_activity_name, *player_annotations = frame_annotation.strip().split(' ')

            Y_group_LSTM.append(group_activity_name)

            # read all frames (with count equal to frames_per_tracklet) and keep in memory, each individual annotation gets a part of all frames to extract an individual tracklet
            main_frame_id, _ = main_frame_file_name.split('.')
            main_frame_id = int(main_frame_id)
            selected_frames = [f"{dataset_path}/{video_idx}/{main_frame_id}/{i}.jpg" for i in
                               range(main_frame_id - frames_before_main_frame,
                                     main_frame_id + frames_after_main_frame + 1)]
            selected_frames = [np.asarray(Image.open(f)) for f in selected_frames]
            current_ragged_row_length = 0

            while len(player_annotations) != 0:
                x_annotation, y_annotation, width_annotation, height_annotation, individual_activity_name, *player_annotations = player_annotations
                x_annotation, y_annotation, width_annotation, height_annotation = int(x_annotation), int(
                    y_annotation), int(
                    width_annotation), int(height_annotation)

                individual_tracklet = []
                Y_individual_LSTM.append(individual_activity_name)

                for frame in selected_frames:
                    try:
                        cropped_frame = tf.image.crop_to_bounding_box(frame, y_annotation, x_annotation,
                                                                      height_annotation,
                                                                      width_annotation)
                    except ValueError as e:
                        print(
                            f"inconsistent annotation at video_idx: {video_idx}, main_frame_file_name: {main_frame_file_name}")
                        print(
                            f"details: y_annotation: {y_annotation}, x_annotation: {x_annotation}, height_annotation: {height_annotation}, width_annotation: {width_annotation}")
                        raise e
                    individual_tracklet.append(cropped_frame)

                # todo optimize resizing in a batch call or layer
                individual_tracklet = tf.image.resize(individual_tracklet,
                                                      size=(image_resize_height, image_resize_width))
                X_individual_LSTM.append(individual_tracklet)

                current_ragged_row_length += 1
                X_group_LSTM.append(individual_tracklet)

            ragged_row_lengths.append(current_ragged_row_length)

    # RaggedTensor Shape: [batch, (count_individual_tracklet), frames_per_tracklet, image_resize_height, image_resize_width, 3]
    # batch is equal to the number of all annotated lines in all annotations.txt files
    X_group_LSTM = tf.RaggedTensor.from_row_lengths(values=X_group_LSTM, row_lengths=ragged_row_lengths)
    X_individual_LSTM = tf.stack(X_individual_LSTM)

    return X_group_LSTM, Y_group_LSTM, X_individual_LSTM, Y_individual_LSTM

video_idxes_to_load = train_video_idxes[0] # for testing the implementation
X_train_group_LSTM, Y_train_group_LSTM, X_train_individual_LSTM, Y_train_individual_LSTM = (
    load_video_idxes_from_volleyball_dataset(dataset_path, video_idxes_to_load, TRAIN_TIME_STEPS, XCEPTION_resize_height, XCEPTION_resize_width))