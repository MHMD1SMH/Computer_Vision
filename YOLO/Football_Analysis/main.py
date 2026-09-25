from utils import read_video, save_video
from tracker import Tracker

def main():
    video_frames = read_video("Input_Videos/08fd33_4.mp4")
    tracker = Tracker("Training/football_players-2/weights/best.pt")
    tracks = tracker.get_opject_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl")

    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, "Output_Videos/output_video.avi")


if __name__ == "__main__":
    main()