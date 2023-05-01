import cv2
import os

def extract_frames(video_path, video_id, label, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    success, frame = video.read()

    frame_count = 0
    saved_count = 0

    while success:

        if frame_count % fps*10 == 0:
            target_size = (224, 224)
            image_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            file_name = os.path.join(output_folder, f'{label}_{video_id}_{saved_count}.png')

            saved = cv2.imwrite(file_name, image_resized)

            if not saved:
                print(f'Failed to save {file_name}')

            saved_count += 1

        frame_count += 1
        success, frame = video.read()

    video.release()
    print(f'Extracted {saved_count} of {video.get(cv2.CAP_PROP_FRAME_COUNT)} frames from {video_path} to {output_folder}')

def videos_to_frames(input_folder, output_folder):
    files = os.listdir(input_folder)
    video_files = [file for file in files if file.lower().endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        
        video_attributes = os.path.splitext(video_file)[0].split('_')
        video_label = video_attributes[0]
        video_id = video_attributes[1]

        extract_frames(video_path, video_id, video_label, output_folder)


videos_to_frames('dataset/raw_videos/csgo', 'dataset/frames')