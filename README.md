# ContinousConv-hl

## Introduction

本分支中的代码用于本地进行的数据处理和数据分析，在远程 GPU workstation 上不方便进行的操作均使用本分支中的代码完成，比如为 workstation 准备数据，以及模型预测结果的 visualization

## Usage

1. _dataprocess.py_

    主要使用的函数有：
    
    1. get_particle_data

        输入 _[apic2d生成的txt文件名, 保存的pkl文件名, num_lines]_ 

        本函数将原始数据保存为 _[step1, step2, step3, ..., ]_ 的形式，并通过 .pkl 文件保留数据结构。
        
        其中，每一个 step 为一个 list，其形式为 _[p_pos_matrix, p_vel_matrix, label1_matrix, label2_matrix]_
    
    2. _data_normalization_

        输入 _[origin_pkl, save_file]_
        
        将上一步产生的 .pkl 文件中的数据进行 normalization，数据格式不变，但会生成一个新的文件 _fluid_trian.pkl_ 进行保存
    
    3. _add_training_data_

        上一步生成的 _fluid_trian.pkl_ 文件将在最终用于 GPU workstation 的统一训练，为了避免覆盖，本函数用于后续增加数据到 _fluid_train.pkl_ 中，同样会对新加入的数据进行 normalization
    
    4. _boundary_generation_

        用 sample 的方式生成 normalized boundary particles, 其 normals 也会同步生成。最终，数据以 _[box_pos_matrix, box_normals_matrix]_ 的形式保存到 _box_train.pkl_ 文件

2. _rendering.py_

    用于读取预测过程中产生的数据，并进行动画可视化
    
    Note : 预测过程中，使用的是 _np.save_ 仅仅对 pos 进行存储，因此在读取时，只需要反复使用 _np.load_ 即可获得每个timestep的position数据。因此，如果在预测过程中，save 的方式不同，请不要使用本代码中的读取方法，可以依照 read_numpy_data 或 read_predict_data 的形式，新增适合自己的读取方法。

    读取方法要求能够返回 points : [step1_matrix, step2_matrix, ...]，其中，每个step是所有particle的坐标构成的矩阵