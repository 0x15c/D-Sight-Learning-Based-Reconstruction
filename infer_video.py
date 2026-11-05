"""Run sensor reconstruction on full, uncropped camera video."""
from demo import main

# Project-relative or absolute filename. The checkpoint crop is applied.
VIDEO_PATH = "video/eval.mp4"

if __name__ == "__main__":
    main(video_path=VIDEO_PATH)
