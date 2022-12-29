# TinyM2Net
## 目录结构
├── DataGeneration  
├── DataSet  
├── demo_real_quantized_cpu.ipynb  
├── main.py  
├── README.md  
├── train.py  
├── models  
├── read_sys_info.py  
├── Memory_LOG  
├── TestData  
├── LOG  
├── CameraAudioRead  
├── compare_compress_real.py  
├── compare_compress.py  
└── demo.ipynb  

`DataGeneration` 文件夹下为从视频生成训练数据的代码  
`DataSet` 是训练数据集  
`demo_real_quantized_cpu.ipynb`展示了使用摄像头和麦克风进行量化推理的代码  
`train.py` 是训练模型的代码  
`models` 下是训练得到的模型，共有五个，包括多模态未压缩模型（使用正常卷积）、多模态压缩未量化模型（使用Speratle卷积）、多模态量化模型、图像模态模型、语音模态模型  
`read_sys_info.py` 用于读取jestson平台的信息，主要包括内存和功耗  
`Memory_LOG` 文件夹下存放内存使用情况的log文件  
`TestData` 存放了使用麦克风和摄像头测试时的数据  
`LOG` 存放实际使用麦克风和摄像头测试环境时的结果输出  
`CameraAudioRead` 存放了使用摄像头和麦克风的代码  
`compare_compress_real.py` 用于实际测试模型，包括使用`TestData`的数据测试模型和直接使用麦克风和摄像头测试环境（实验3）  
`compare_compress.py` 使用`DataSet`中的验证数据集测试五个模型的精度（实验1和实验2）  
`demo.ipynb` 展示了模型的构建和使用  

