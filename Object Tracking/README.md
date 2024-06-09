## Object Tracking using YOLOv8 and Deep Sort

<img src="https://github.com/Irash-Perera/Computer-Vision-Projects/blob/main/Object%20Tracking/people_output.gif" alt="Output File" width="800"/>

This project is an implementation of object tracking using YOLOv8 and Deep Sort. The YOLOv8 model is used to detect objects in the video frames. The detected objects are then tracked using the Deep Sort algorithm. 

Here, people are detected and tracked in the video frames. The output video is saved in the `people_output.mp4` file.

### Getting Started

#### 1.  Clone the repository
```
git clone https://github.com/Irash-Perera/Computer-Vision-Projects.git
```
#### 2. Install the required libraries
Open the terminal in the particular project directory and run the following command.
```
pip install -r requirements.txt
```
#### 3. Clone the forked deep_sort repository
In this project I am using a forked version of the original deep_sort repository. That is because the original repository is not compatible with the latest version of TensorFlow. Clone the forked repository using the following command.
```
git clone https://github.com/Irash-Perera/Deep-Sort.git
```
Note:
 - `model_data` contains the deep-sort feature exctraction model. Download the model from [here](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)
