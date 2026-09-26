from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from ball_assigner import BallAssigner
from cam_move_est import CamMoveEst
def main():
    video_frames = read_video("Input_Videos/08fd33_4.mp4")
    tracker = Tracker("Training/football_players-2/weights/best.pt")
    tracks = tracker.get_opject_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl")

    tracks["ball"] = tracker.interpolate_ball(tracks["ball"])
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])
    for frame, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame],track['bbox'],player_id)
            tracks['players'][frame][player_id]['team'] = team
            tracks['players'][frame][player_id]['team_color']= team_assigner.team_colors[team]

    ball_assigner = BallAssigner()
    team_controlling = []
    for frame, player_track in enumerate(tracks['players']):
        assigned_player = ball_assigner.assign(player_track,tracks['ball'][frame][1]['bbox'])
        if assigned_player != -1:
            tracks['players'][frame][assigned_player]['has_ball']= True
            team_controlling.append(tracks['players'][frame][assigned_player]['team'])
        else:
            team_controlling.append(team_controlling[-1])
    
    output_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_frames, "Output_Videos/output_video.avi")


if __name__ == "__main__":
    main()