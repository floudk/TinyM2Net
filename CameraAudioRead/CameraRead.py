import cv2
import nanocamera as nano

def read_camera(resize_w=32, resize_h=32, frames_number=1):
    """
    Args:
        resize_w: the width of the images, default 32 
        resize_h: the height of the images, default 32
        frames_number: the number of sampled frames, default 1
    
    Returns:
        a list of sampled images, each images shape is (3,32,32) 
    """
    camera = nano.Camera(flip=0,width=640, height=480, fps=30)
    status = camera.isReady()
    if status:
        print("CSI camera is ready now!")
    else:
        error_status = camera.hasError()
        print(error_status)
        print("camera isn't ready!")
        exit(-1)

    frame_number = 0
    frames = []
    while frame_number<frames_number and camera.isReady():
        try:
            frame = camera.read()
            #print(frame.shape)
            frame=cv2.resize(frame,(resize_w,resize_h))
            frames.append(frame)
            #cv2.imwrite(str(frame_number)+".png", frame)
            frame_number += 1
        except KeyboardInterrupt:
            break
    
    camera.release()
    del camera
    return frames
    




if __name__=="__main__":
    read_camera()
            
            