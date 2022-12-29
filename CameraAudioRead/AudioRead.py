#arecord -D plughw:2,0 -f S16_LE -r 48000 -c 2 test1.wav

import pyaudio
import wave

import librosa
import numpy as np
# 得到的index是11
def find_mic_index():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

def read_wav(wav_path, frames_number, sample_rate, channels):
    x,sr=librosa.load(wav_path, sr=sample_rate, mono=channels)
    mfcc_features=librosa.feature.mfcc(x,sr=sr,n_mfcc=44)  # MFCC feature extraction from audios  

    mfccs = []
    for i in range(frames_number):
        mfcc_feature=mfcc_features[:,i*13:(i+1)*13]
        mfccs.append(np.expand_dims(mfcc_feature,axis=0))

    return mfccs

def read_audio(record_second = 1, frames_number=1):
    """
    Args:
        record_second: record time(s), default 1s
        frames_number: the number of sampled frames, default 1 
    
    Returns:
        the sample frames list, including several frames(default 1), each frame with shape (1, 44, 13) 
    """
    channels = 1
    rate = 48000
    chunk = 1024
    mic_index = 11
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=channels,
        input_device_index=mic_index,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    frames = []
    #print("audio record begin........")
    for i in range(0, int(rate/chunk*record_second)):
        data = stream.read(chunk, exception_on_overflow=False) # 这里得到的是字节类型的数据 \x05\x22 这种，长度是chunk*format的字节数，所以每个字节可以转为一个数，0-255之间
        frames.append(data) 

    #frames_bytes = list(b''.join(frames)) #这可以每个字节转成相应的数字，所以长度就是int(rate/chunk*record_second)*chunk*format的字节数
    

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save to temp file
    waveFile = wave.open("tmp.wav", 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


    mfcc_features = read_wav("tmp.wav", frames_number, rate, True)

    return mfcc_features
    
def save_audios(savepath):
    channels = 1
    rate = 48000
    chunk = 1024
    mic_index = 11
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=channels,
        input_device_index=mic_index,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    frames = []
    #print("audio record begin........")
    for i in range(0, int(rate/chunk*12)):
        data = stream.read(chunk, exception_on_overflow=False) # 这里得到的是字节类型的数据 \x05\x22 这种，长度是chunk*format的字节数，所以每个字节可以转为一个数，0-255之间
        frames.append(data) 

    #frames_bytes = list(b''.join(frames)) #这可以每个字节转成相应的数字，所以长度就是int(rate/chunk*record_second)*chunk*format的字节数
    

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save to temp file
    waveFile = wave.open(savepath, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



if __name__=="__main__":
    savepath = "/home/tangpeng/TinyM2NetNew/videos/mic_audios/dogs/dog_30_40_2.wav"
    save_audios(savepath)

    # mfccs = read_audio(record_second = 1, frames_number=2)
    # print(len(mfccs))
    # print(mfccs[0])
    # wav_test_path = "/home/tangpeng/tangpengtest/test.wav"
    # test_wav(wav_path=wav_test_path)
    # import noisereduce as nr
    # from scipy.io import wavfile
    # # load data
    # rate, data = wavfile.read("tmp.wav")
    # # perform noise reduction
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
