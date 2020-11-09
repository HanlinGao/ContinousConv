# Introduction
本分支中的代码用于GPU workstation。

由于远程连接GPU workstation后，主要依靠本分支进行git同步，因此，需要尽量避免代码修改、避免添加数据集文件到git中进行管理（GPU workstation使用的数据集会通过winSCP进行远程传输）

## 项目结构
- dataset目录
    - box_train.pkl
    - fluid_evaluation.pkl
    - fluid_train.pkl
- out目录
- Model.py
- run.py
- train.py
- Readme.md

## Usage
**Make sure dataset you are going to use is in the right form**

### Dataset
The _dataset_ directory contains data that will be used during train on GPU workstation

1. _box_train.pkl_ 

    数据形式为 _[box_pos_matrix, box_normals_matrix]_ ，其中，_box_pos_matrix_ 是一个 _n*3_ 大小的矩阵，每一行为 _[x, y, z]_ ，box_normals_matrix同理，每一行为3个normals特征。数据均进行过normalization，pos为[-1, 1], [0, 4]; normals为某一维度设置为碰撞速度的反方向

2. _fluid_evaluation.pkl_ 
    
    数据形式为 _[particle_pos_matrix, particle_vel_matrix, label1_matrix, label2_matrix]_ ，其中的每个matrix形状同理，表示的是1个timestep内的所有particle信息。本文件由训练数据中分离出来的某一个文件，如apic13.txt进行normalization后得到。

3. _fluid_train.pkl_ 
    
    数据形式为 _[particle_pos_matrix, particle_vel_matrix, label1_matrix, label2_matrix]_，本文件为训练数据，由apic2d生成的数据经过normalization得到。
    
    训练产生的训练曲线图片等文件，请按照需要，及时清理和保存。其中的图片文件按照 _训练日期+epoch数量+lr_ 的形式保存，以此避免覆盖，且方便区分

    **由于本文件经常进行数据量补充等相关修改，每次使用时，请确保使用的为最新版本训练集。相关训练集的操作详情请见hl branch**

4. _out_ 

    本文件夹保存预测过程中生成的 position 信息，由于在预测过程中反复调用 _np.save_ ，进行读取时，需要反复使用 _np.load_

5. _train.py_

    专门用于训练模型，训练过程中，每个epoch结束存储一次模型参数，所有epoch结束保存一次整个训练曲线

    使用时需要通过命令行指定 _num_epochs_ , _batch_size_ , _lr_ 参数的值，其他的可以不特别指定而使用default值。Note: 这样的前提是保证相关数据集的名称没有发生变化，请提前确认

6. _run.py_

    专门用于测试模型。测试过程中需确保 _model.requires_grad_ 设置为 _False_ . 

    测试过程中，每次预测出一个timestep的数据，会通过 _np.save_ 保存其中的 position 信息。因此，读取时，只需要反复使用 _np.load_ ，即可每次直接获得 position 的 numpy 矩阵，用于 visualization

    **run.py尚未完全通过测试，请谨慎使用**

