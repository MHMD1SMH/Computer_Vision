import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class BallAssigner():
    def __init__(self):
        self.max_distance = 70


    def assign(self, players, ball_bbox):
        assigned_player = -1
        
        ball_position = get_center_of_bbox(ball_bbox)
        for player_id, player in players.items():
            distance = min(measure_distance((player['bbox'][0],player['bbox'][-1]),ball_position),
                           measure_distance((player['bbox'][2],player['bbox'][-1]),ball_position))

            if distance < self.max_distance:
                if distance < 9999999999:
                    assigned_player = player_id

        return assigned_player