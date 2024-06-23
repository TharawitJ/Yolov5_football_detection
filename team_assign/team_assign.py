
from sklearn.cluster import KMeans
from collections import defaultdict

class TeamAssigner:
    def __init__(self):
        self.team_colors = defaultdict(list)
        self.player_team_dict = {} # to separate player's team

    def get_clustering_model(self, img):
        # reshape to 2D array
        img_2d = img.reshape(-1,3)  
        kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(img_2d)
        return kmeans_model

    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = img[0: int(img.shape[0]/2),:]

        # cluster model
        kmeans_model = self.get_clustering_model(top_half)
        labels = kmeans_model.labels_
        clustered_img = labels.reshape(top_half.shape[0],top_half.shape[1])

        corner_cluster = [clustered_img[0,0],
                  clustered_img[0,-1],
                  clustered_img[-1,0],
                  clustered_img[-1,-1]]
        
        background_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - background_cluster
        player_color = kmeans_model.cluster_centers_[player_cluster]

        return player_color

    def assign_team(self, frame, player_detection):
        player_colors = []
        for _,player_detection in player_detection.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)

        kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)

        self.kmeans_model = kmeans_model

        self.team_colors[1] = kmeans_model.cluster_centers_[0]
        self.team_colors[2] = kmeans_model.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)
        team_id = self.kmeans_model.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id


