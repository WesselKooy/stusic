# create_video.py

import cv2
from moviepy.editor import VideoClip, AudioFileClip
import numpy as np
from animate_image import generate_frame_at_time

def create_video(img_rgb, depth_map, audio_path, output_video_path, fps=24, duration=None):
    # Desired video dimensions for Instagram Reels (1080x1920 pixels)
    desired_width = 1080
    desired_height = 1920
    desired_aspect = desired_width / desired_height

    h, w, _ = img_rgb.shape
    frame_aspect = w / h

    # Function to generate a frame at time t
    def make_frame(t):
        warped_img_cropped = generate_frame_at_time(img_rgb, depth_map, t, duration)

        # Resize frames or add padding to match desired aspect ratio
        if frame_aspect > desired_aspect:
            # Add vertical padding
            new_height = int(w / desired_aspect)
            pad_height = (new_height - h) // 2
            frame_padded = cv2.copyMakeBorder(warped_img_cropped, pad_height, pad_height, 0, 0,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif frame_aspect < desired_aspect:
            # Add horizontal padding
            new_width = int(h * desired_aspect)
            pad_width = (new_width - w) // 2
            frame_padded = cv2.copyMakeBorder(warped_img_cropped, 0, 0, pad_width, pad_width,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            frame_padded = warped_img_cropped

        # Resize frames to desired dimensions
        frame_resized = cv2.resize(frame_padded, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        return frame_rgb

    # Create a VideoClip using the make_frame function
    video_clip = VideoClip(make_frame, duration=duration)
    video_clip = video_clip.set_fps(fps)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the video file
    video_clip.write_videofile(
        output_video_path,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='medium',
        bitrate='8000k',
        audio_bitrate='192k',
    )
