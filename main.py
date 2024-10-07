# main.py

import sys
from depth_estimation import estimate_depth
from animate_image import generate_frame_at_time
from create_video import create_video
from moviepy.audio.io.AudioFileClip import AudioFileClip

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py image_path audio_path output_video_path")
        sys.exit(1)

    image_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_video_path = sys.argv[3]

    # Step 1: Estimate depth
    print("Estimating depth...")
    img_rgb, depth_map = estimate_depth(image_path)

    # Step 2: Prepare for animation
    print("Preparing animation parameters...")
    fps = 24  # Reduced FPS to lower the number of frames
    # Load audio to get duration
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    # Step 3: Create video
    print("Creating video...")
    create_video(img_rgb, depth_map, audio_path, output_video_path, fps=fps, duration=audio_duration)

    print("Video saved to", output_video_path)

if __name__ == "__main__":
    main()
