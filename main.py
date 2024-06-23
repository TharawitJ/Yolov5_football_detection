import cv2
from utils import read_video, save_video
from tracker import Tracker
from team_assign import TeamAssigner


def main():
    # Read
    file_name = '08fd33_4'
    video_frame = read_video('input\\'+ file_name +'.mp4')

    # Initialize Tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_obj_track(video_frame,
                                   read_from_stub=True,
                                   stub_path ='stubs/'+ file_name +'.pkl')
    
    # # To save cropped img for Kmeans
    # for track_id,player in tracks['players'][1].items():
    #     frame = video_frame[0]
    #     bbox = player['bbox']
    #     crop_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # crop img frame y1 to y2 bbox[1]:bbox[3] and x1 to x2 bbox[0]:bbox[2]
    #     cv2.imwrite(f'output/crop_img.jpg',crop_img)
    #     break # need one img

    # Assign player team
    team_assign = TeamAssigner()
    team_assign.assign_team(video_frame[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assign.get_player_team(video_frame[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assign.team_colors[team]

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frame, tracks)

    # Save
    save_video(output_video_frames,'output/output'+ file_name +'.avi')

if __name__ == '__main__':
    main() 