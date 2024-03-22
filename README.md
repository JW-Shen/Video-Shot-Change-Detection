# Video Shot Change Detection

* Video shot change detection in six algorithms:
    | Algorithm                    | Description |
    | -------                      | -------   |
    | Color Histogram Comparison   | calculate histogram correlation between two frames |
    | Frame Difference             | calculate frame difference between two frames |
    | Edge Change Ratio            | calculate edge change ratio |
    | Motion Vectors               | calculate L1 distance of motion vectors between two frames |
    | Twin-Comparison Approach     | aply twin-comparison approach based on color histogram |
    | Convolutional neural network | calculate L1 distance of extracted features between two frames |
    | Ensemble                     | using the above six algorithms for voting |

## Imput/Output

* Input: video
* Output: frame number at which the shot change occurs


## Enviroment

* Programming Language Version: Python 3.8.12
* Operating Platform: macOS Ventura 13.5
* Install Related Packages or Modules:
  ```bash
  pip install -r requirements.txt
  ```
  
## Run Code

```bash
  python -m video_shot_change_detection --video-name ${video_name} --algorithm ${algorithm}
```

Replace `${video_name}` with the video file name <br />
Replace `${algorithm}` with one of `histogram`, `frame_diff`, `ECR`, `motion`, `twin`, `cnn`, `ensemble`, `all`


## Result

* Video: climate.mp4 (news about climate)

    | Algorithm                    | Precision | Recall  | F1 Score |
    | -------                      | -------   | ------- | -------  |
    | Color Histogram Comparison   | 0.923     | 0.800   | 0.857    |
    | Frame Difference             | 1.000     | 0.667   | 0.800    |
    | Edge Change Ratio            | 0.800     | 0.533   | 0.640    |
    | Motion Vectors               | 0.818     | 0.600   | 0.692    |
    | Twin-Comparison Approach     | 0.923     | 0.800   | 0.857    |
    | Convolutional neural network | 0.765     | 0.867   | 0.812    |
    | Ensemble                     | 0.917     | 0.733   | 0.815    |

* Video: news.mpg 

    | Algorithm                    | Precision | Recall  | F1 Score |
    | -------                      | -------   | ------- | -------  |
    | Color Histogram Comparison   | 1.000     | 1.000   | 1.000    |
    | Frame Difference             | 1.000     | 1.000   | 1.000    |
    | Edge Change Ratio            | 0.250     | 0.571   | 0.348    |
    | Motion Vectors               | 0.833     | 0.714   | 0.769    |
    | Twin-Comparison Approach     | 1.000     | 1.000   | 1.000    |
    | Convolutional neural network | 1.000     | 0.857   | 0.923    |
    | Ensemble                     | 1.000     | 1.000   | 1.000    |

* Video: ngc.mpeg (National Geographic)

    | Algorithm                    | Precision | Recall  | F1 Score |
    | -------                      | -------   | ------- | -------  |
    | Color Histogram Comparison   | 0.875     | 0.778   | 0.824    |
    | Frame Difference             | 1.000     | 0.611   | 0.759    |
    | Edge Change Ratio            | 0.800     | 0.778   | 0.789    |
    | Motion Vectors               | 0.690     | 0.556   | 0.615    |
    | Twin-Comparison Approach     | 0.763     | 0.806   | 0.784    |
    | Convolutional neural network | 0.963     | 0.722   | 0.825    |
    | Ensemble                     | 0.962     | 0.694   | 0.806    |
