**Football Match Detection project**
---
### This project focusing on detecting and tracking ball, players and referees during football match.

* Using image dataset to train and video dataset to detect and track.
    * image dataset from https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc.
    * video dataset from https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips.

* To detect the object by using Yolov5 model.
    * I train model by using 9 sets of image dataset from souce that I mention above 10 epochs for each sets on google colab notebook.

* To separate players by using KMeans model.
    * KMeans using for cluster color of player's shirt and background for color assignment to ellispe tracker under the players.

* Design the tracker.
    * Original tracker are bounding box on the objects and the last design are ellispe under the players using ellispe's color same as players shirt by KMeans to separate the team and referees and green rectangle above the ball. 

---
