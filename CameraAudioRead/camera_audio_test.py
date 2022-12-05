from AudioRead import read_audio
from CameraRead import read_camera

def main_test():
    camera_frames = read_camera(resize_h=32, resize_w=32, frames_number=1)
    audio_frames = read_audio(record_second=1,frames_number=1)

    print(len(camera_frames))
    print("camera frame shape: ", camera_frames[0].shape) #(32,32,3)
    print(len(audio_frames))
    print("audio frame shape:", audio_frames[0].shape) #(1,44,13)


if __name__=="__main__":
    main_test()
    
    