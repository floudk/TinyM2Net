import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
def draw_box(): #绘制箱线图
    basemodel_memory = []
    normal_convmodel_memory = []
    quantization_model_memory = []
    with open('basemodel.log','r') as fd:
        line = fd.readline().split(', ')
        basemodel_memory = [int(x) for x in line[1:-1]]
    with open('logg.log','r') as fd:
        line = fd.readline().split(', ')
        normal_convmodel_memory = [int(x) for x in line[1:-1]]
    with open('quantizationmodel.log','r') as fd:
        line = fd.readline().split(', ')
        quantization_model_memory = [int(x) for x in line[1:-1]]

    plt.figure(figsize=(4,3),dpi=200)
    plt.title("Memory Compare")
    print(len(basemodel_memory))
    #print(normal_convmodel_memory)
    #labels = ['非压缩模型','压缩非量化模型','量化模型']
    labels = ['1','2','3']
    #plt.boxplot([normal_convmodel_memory, basemodel_memory, quantization_model_memory],labels='1')
    plt.boxplot(normal_convmodel_memory, notch=True,sym='*',showmeans=True,meanline=True,labels='1')

    plt.savefig("./memory_boxx.png")
    plt.close()


def plot_violinplot(): #画小提琴图，展示效果比箱线图好
    basemodel_memory = []
    normal_convmodel_memory = []
    quantization_model_memory = []
    with open('basemodel.log', 'r') as fd:
        line = fd.readline().split(', ')
        basemodel_memory = [int(x) for x in line[1:-1]]
    with open('normalconvmodel.log', 'r') as fd:
        line = fd.readline().split(', ')
        normal_convmodel_memory = [int(x) for x in line[1:-1]]
    with open('quantizationmodel.log', 'r') as fd:
        line = fd.readline().split(', ')
        quantization_model_memory = [int(x) for x in line[1:-1]]


    data = {'memory':[], 'model_type':[]}
    data['memory'].extend(normal_convmodel_memory)
    data['model_type'].extend(['normal model' for i in range(len(normal_convmodel_memory))])
    data['memory'].extend(basemodel_memory)
    data['model_type'].extend(['base model' for i in range(len(basemodel_memory))])
    data['memory'].extend(quantization_model_memory)
    data['model_type'].extend(['quantization model' for i in range(len(quantization_model_memory))])

    data = pd.DataFrame(data)
    sns.violinplot(x='model_type',y='memory',data=data)
    plt.title('不同模型内存分布')
    plt.xlabel(' ')
    plt.ylabel('内存(MB)')
    plt.xticks([0,1,2],['未压缩','未量化','量化'])
    plt.savefig('./memory_violin.png')
    #plt.show()
    plt.close()

if __name__=="__main__":
    #draw_box()
    plot_violinplot()