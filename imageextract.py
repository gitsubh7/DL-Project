import subprocess
subprocess.run(["pip", "install", "opencv-python-headless", "matplotlib"])
import cv2
import os

def run_image_extraction(input , output):
    def extract_frames(video_path, output_path):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        while success:
            if count % int(fps/25) == 0:
                cv2.imwrite(os.path.join(output_path, f"image_{count}.jpg"), image)

                blue, green, red = cv2.split(image)
                if (green.mean() > red.mean()) and (green.mean() > blue.mean()):
                    cv2.imwrite(os.path.join(output_path, "process_image", f"green_{count}.jpg"), image)

            success, image = vidcap.read()
            count += 1

    def create_video(images_path, output_video_path, fps):
        img_array = []
        files = sorted(os.listdir(images_path))
        size = None
        for file in files:
            if file.startswith("green_"):
                filename = os.path.join(images_path, file)
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)

        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])

        out.release()


    video_path = f'{input}'

    output_path = 'output/images'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'process_image'), exist_ok=True)

    extract_frames(video_path, output_path)

    output_video_path = 'output_video3.mp4'
    fps = 25
    create_video(os.path.join(output_path, 'process_image'), output_video_path, fps)

    # green percentage

    def calculate_green_percentage(frame):
        b, g, r = cv2.split(frame)
        total = np.sum(frame)
        green_percentage = (np.sum(g) / total) * 100 if total > 0 else 0
        return green_percentage

    def save_frames_with_green_percentage(input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            green_percentage = calculate_green_percentage(frame)

            if green_percentage > 34:
                out.write(frame)

            current_frame += 1

        cap.release()
        out.release()

    video_path = 'output_video3.mp4'
    output_video_path = 'output_video2.mp4'
    save_frames_with_green_percentage(video_path, output_video_path)

    # brightness

    def calculate_brightness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = int(round(cv2.mean(gray)[0]))
        return brightness

    video_path = 'output_video2.mp4'
    output_path = f'{output}'

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    output_video = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    previous_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        brightness = calculate_brightness(frame)

        if brightness <= 150:
            output_video.write(frame)

            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_frame is not None:
                gradient = cv2.absdiff(current_frame, previous_frame)

            previous_frame = current_frame

    cap.release()
    output_video.release()


run_image_extraction(input , output)
# write it as, run_image_extraction(football.mp4 , football_processed.mp4)
# these files must be present at the base directory of the code
