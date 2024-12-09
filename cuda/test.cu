// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
// 编译的命令为：nvcc test.cu -o test -g -G -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp 

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>

#define wbCheck(stmt)  do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "\n\nFailed to run stmt %d ", __LINE__);                       \
            printf( "Got CUDA error ...  %s \n\n", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
}




// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}



// 用于打印测试
void printVector(float* a,int length)
{

    for(int j=0;j<length;j++){
        std::cout <<a[j]<<std::endl;
    }

}


//矩阵转置
__global__ void matrixTranspose(float* input, float* output, int input_row, int input_col)
{
      int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程在矩阵范围内
    if (point_idx < input_row * input_col)
     {
        int row = point_idx / input_col; // 当前输入矩阵的行
        int col = point_idx % input_col; // 当前输入矩阵的列
        output[col * input_row + row] = input[point_idx]; // 转置操作
    }
}


//矩阵乘法
__global__ void matrixMultiply(float* A, float* B, float* output, int M, int N, int K) {
    // A: MxN matrix
    // B: NxK matrix
    // output: MxK result matrix
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(point_idx < M * K) 
    {
        // 计算当前行、列
        int row = point_idx / K;
        int col = point_idx % K;

        float value = 0.0f;
        for(int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * K + col];
        }
        output[point_idx] = value;
    }
}


//Convolution卷积操作
__global__ void Convolution(float* input, float* output, float* weight, float* bias, int num_points, int in_channels, int out_channels) 
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(point_idx < num_points * out_channels)
    {
        // 计算当前行、列
        int row = point_idx / num_points;
        int col = point_idx % num_points;
        output[point_idx] = bias[row];
        for (int i = 0; i < in_channels; i++) 
        {
            output[point_idx] += input[i * num_points + col] * weight[row * in_channels + i];
       }
         
    }
    
}


//BatchNorm批归一化操作
__global__ void BatchNorm(float* input, float* output, float* weight, float* bias, float* mean, float* var, int num_points, int feat_stn_bn1_var_channels)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //计算当前point_idx所在行
    int row = point_idx / num_points;
    if(point_idx < num_points * feat_stn_bn1_var_channels)
    {
         // 归一化
        output[point_idx] = weight[row] * (input[point_idx] - mean[row]) / sqrtf(var[row] + 1e-5f) + bias[row];
    }  

}

//Relu激活函数操作
__global__ void Relu(float* input, float* output,  int num_points, int feat_stn_relu_channels)
 {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < num_points * feat_stn_relu_channels)
    {
        if (input[point_idx] <= 0)
            output[point_idx] = 0;
        else
            output[point_idx] = input[point_idx];
    }
}

//MaxPool最大池化操作
__global__ void MaxPool(float* input, float* output,int num_points,int feat_stn_max_channels)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (point_idx < feat_stn_max_channels)
    {
        float max = input[point_idx * num_points];
        for (int i = 0; i < num_points;i+=100)
        {
            if(input[point_idx * num_points + i] > max)
            {
                max = input[point_idx * num_points + i];
            }
        }
        output[point_idx]=max;
    }
}


//feat_stn_fc
__global__ void FC(float* input, float* output,float* weight, float* bias,int in_channels, int out_channels) 
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(point_idx < out_channels)
    {
        //初始化为偏置
        output[point_idx] = bias[point_idx];

        for (int i = 0; i < in_channels; i++)
        {
            output[point_idx] += input[i] * weight[point_idx * in_channels + i];
        } 
    }
}

//Iden仿射变换
__global__ void Iden(float* input, int out_h,int out_w)
{
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(point_idx < out_h * out_w)
    {
        // 计算当前行、列
        int row = point_idx / out_w;
        int col = point_idx % out_w;
        if(row == col)
        {
            input[point_idx] += 1;
        }
    }
}



int main(int argc, char *argv[]) {
    
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // cout << dir;
    
    // 读取模型参数
    std::map<std::string, std::vector<float>> params = read_params(dir);

    std::vector<float> feat_stn_conv1_weight_host = params["feat.stn.conv1.weight"];
    std::vector<float> feat_stn_conv1_bias_host = params["feat.stn.conv1.bias"];
    std::vector<float> feat_stn_bn1_weight_host = params["feat.stn.bn1.weight"];
    std::vector<float> feat_stn_bn1_bias_host = params["feat.stn.bn1.bias"];
    std::vector<float> feat_stn_bn1_mean_host = params["feat.stn.bn1.running_mean"];
    std::vector<float> feat_stn_bn1_var_host = params["feat.stn.bn1.running_var"];

    std::vector<float> feat_stn_conv2_weight_host = params["feat.stn.conv2.weight"];
    std::vector<float> feat_stn_conv2_bias_host = params["feat.stn.conv2.bias"];
    std::vector<float> feat_stn_bn2_weight_host = params["feat.stn.bn2.weight"];
    std::vector<float> feat_stn_bn2_bias_host = params["feat.stn.bn2.bias"];
    std::vector<float> feat_stn_bn2_mean_host = params["feat.stn.bn2.running_mean"];
    std::vector<float> feat_stn_bn2_var_host = params["feat.stn.bn2.running_var"];

    std::vector<float> feat_stn_conv3_weight_host = params["feat.stn.conv3.weight"];
    std::vector<float> feat_stn_conv3_bias_host = params["feat.stn.conv3.bias"];
    std::vector<float> feat_stn_bn3_weight_host = params["feat.stn.bn3.weight"];
    std::vector<float> feat_stn_bn3_bias_host = params["feat.stn.bn3.bias"];
    std::vector<float> feat_stn_bn3_mean_host = params["feat.stn.bn3.running_mean"];
    std::vector<float> feat_stn_bn3_var_host = params["feat.stn.bn3.running_var"];

    std::vector<float> feat_stn_fc1_weight_host = params["feat.stn.fc1.weight"];
    std::vector<float> feat_stn_fc1_bias_host = params["feat.stn.fc1.bias"];
    std::vector<float> feat_stn_bn4_weight_host = params["feat.stn.bn4.weight"];
    std::vector<float> feat_stn_bn4_bias_host = params["feat.stn.bn4.bias"];
    std::vector<float> feat_stn_bn4_mean_host = params["feat.stn.bn4.running_mean"];
    std::vector<float> feat_stn_bn4_var_host = params["feat.stn.bn4.running_var"];

    std::vector<float> feat_stn_fc2_weight_host = params["feat.stn.fc2.weight"];
    std::vector<float> feat_stn_fc2_bias_host = params["feat.stn.fc2.bias"];
    std::vector<float> feat_stn_bn5_weight_host = params["feat.stn.bn5.weight"];
    std::vector<float> feat_stn_bn5_bias_host = params["feat.stn.bn5.bias"];
    std::vector<float> feat_stn_bn5_mean_host = params["feat.stn.bn5.running_mean"];
    std::vector<float> feat_stn_bn5_var_host = params["feat.stn.bn5.running_var"];

    std::vector<float> feat_stn_fc3_weight_host = params["feat.stn.fc3.weight"];
    std::vector<float> feat_stn_fc3_bias_host = params["feat.stn.fc3.bias"];

    std::vector<float> feat_conv1_weight_host = params["feat.conv1.weight"];
    std::vector<float> feat_conv1_bias_host = params["feat.conv1.bias"];
    std::vector<float> feat_bn1_weight_host = params["feat.bn1.weight"];
    std::vector<float> feat_bn1_bias_host = params["feat.bn1.bias"];
    std::vector<float> feat_bn1_mean_host = params["feat.bn1.running_mean"];
    std::vector<float> feat_bn1_var_host = params["feat.bn1.running_var"];




    std::vector<float> feat_fstn_conv1_weight_host = params["feat.fstn.conv1.weight"];
    std::vector<float> feat_fstn_conv1_bias_host = params["feat.fstn.conv1.bias"];
    std::vector<float> feat_fstn_bn1_weight_host = params["feat.fstn.bn1.weight"];
    std::vector<float> feat_fstn_bn1_bias_host = params["feat.fstn.bn1.bias"];
    std::vector<float> feat_fstn_bn1_mean_host = params["feat.fstn.bn1.running_mean"];
    std::vector<float> feat_fstn_bn1_var_host = params["feat.fstn.bn1.running_var"];

    std::vector<float> feat_fstn_conv2_weight_host = params["feat.fstn.conv2.weight"];
    std::vector<float> feat_fstn_conv2_bias_host = params["feat.fstn.conv2.bias"];
    std::vector<float> feat_fstn_bn2_weight_host = params["feat.fstn.bn2.weight"];
    std::vector<float> feat_fstn_bn2_bias_host = params["feat.fstn.bn2.bias"];
    std::vector<float> feat_fstn_bn2_mean_host = params["feat.fstn.bn2.running_mean"];
    std::vector<float> feat_fstn_bn2_var_host = params["feat.fstn.bn2.running_var"];

    std::vector<float> feat_fstn_conv3_weight_host = params["feat.fstn.conv3.weight"];
    std::vector<float> feat_fstn_conv3_bias_host = params["feat.fstn.conv3.bias"];
    std::vector<float> feat_fstn_bn3_weight_host = params["feat.fstn.bn3.weight"];
    std::vector<float> feat_fstn_bn3_bias_host = params["feat.fstn.bn3.bias"];
    std::vector<float> feat_fstn_bn3_mean_host = params["feat.fstn.bn3.running_mean"];
    std::vector<float> feat_fstn_bn3_var_host = params["feat.fstn.bn3.running_var"];

    std::vector<float> feat_fstn_fc1_weight_host = params["feat.fstn.fc1.weight"];
    std::vector<float> feat_fstn_fc1_bias_host = params["feat.fstn.fc1.bias"];
    std::vector<float> feat_fstn_bn4_weight_host = params["feat.fstn.bn4.weight"];
    std::vector<float> feat_fstn_bn4_bias_host = params["feat.fstn.bn4.bias"];
    std::vector<float> feat_fstn_bn4_mean_host = params["feat.fstn.bn4.running_mean"];
    std::vector<float> feat_fstn_bn4_var_host = params["feat.fstn.bn4.running_var"];

    std::vector<float> feat_fstn_fc2_weight_host = params["feat.fstn.fc2.weight"];
    std::vector<float> feat_fstn_fc2_bias_host = params["feat.fstn.fc2.bias"];
    std::vector<float> feat_fstn_bn5_weight_host = params["feat.fstn.bn5.weight"];
    std::vector<float> feat_fstn_bn5_bias_host = params["feat.fstn.bn5.bias"];
    std::vector<float> feat_fstn_bn5_mean_host = params["feat.fstn.bn5.running_mean"];
    std::vector<float> feat_fstn_bn5_var_host = params["feat.fstn.bn5.running_var"];

    std::vector<float> feat_fstn_fc3_weight_host = params["feat.fstn.fc3.weight"];
    std::vector<float> feat_fstn_fc3_bias_host = params["feat.fstn.fc3.bias"];
    
    std::vector<float> feat_conv2_weight_host = params["feat.conv2.weight"];
    std::vector<float> feat_conv2_bias_host = params["feat.conv2.bias"];
    std::vector<float> feat_bn2_weight_host = params["feat.bn2.weight"];
    std::vector<float> feat_bn2_bias_host = params["feat.bn2.bias"];
    std::vector<float> feat_bn2_mean_host = params["feat.bn2.running_mean"];
    std::vector<float> feat_bn2_var_host = params["feat.bn2.running_var"];

    std::vector<float> feat_conv3_weight_host = params["feat.conv3.weight"];
    std::vector<float> feat_conv3_bias_host = params["feat.conv3.bias"];
    std::vector<float> feat_bn3_weight_host = params["feat.bn3.weight"];
    std::vector<float> feat_bn3_bias_host = params["feat.bn3.bias"];
    std::vector<float> feat_bn3_mean_host = params["feat.bn3.running_mean"];
    std::vector<float> feat_bn3_var_host = params["feat.bn3.running_var"];

    std::vector<float> fc1_weight_host_host = params["fc1.weight"];
    std::vector<float> fc1_bias_host_host = params["fc1.bias"];
    std::vector<float> bn1_weight_host = params["bn1.weight"];
    std::vector<float> bn1_bias_host = params["bn1.bias"];
    std::vector<float> bn1_mean_host = params["bn1.running_mean"];
    std::vector<float> bn1_var_host = params["bn1.running_var"];

    std::vector<float> fc2_weight_host = params["fc2.weight"];
    std::vector<float> fc2_bias_host = params["fc2.bias"];
    std::vector<float> bn2_weight_host = params["bn2.weight"];
    std::vector<float> bn2_bias_host = params["bn2.bias"];
    std::vector<float> bn2_mean_host = params["bn2.running_mean"];
    std::vector<float> bn2_var_host = params["bn2.running_var"];

    std::vector<float> fc3_weight_host = params["fc3.weight"];
    std::vector<float> fc3_bias_host = params["fc3.bias"];




    //打印测试
    // for(const auto& value :params["bn1.weight"]) {
    //     std::cout<<value;
    // 

    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    
    int result = 0;//记录最后输出的结果
    int num = 0;   //记录正确识别数量

    int max_points = 34800;

    float * input;
	cudaMalloc((void**)& input, sizeof(float) * max_points*3);
	float * Tinput;
	cudaMalloc((void**)& Tinput, sizeof(float) * max_points*3);
    float * output64;
    cudaMalloc((void**)& output64, sizeof(float) * max_points*64);
    float * output64_2;
    cudaMalloc((void**)& output64_2, sizeof(float) * max_points*64);
    float * output64_3;
    cudaMalloc((void**)& output64_3, sizeof(float) * max_points*64);   
    float * Toutput64;
    cudaMalloc((void**)& Toutput64, sizeof(float) * max_points*64);    
    float * Toutput64_2;
    cudaMalloc((void**)& Toutput64_2, sizeof(float) * max_points*64);
    float * Toutput64_3;
    cudaMalloc((void**)& Toutput64_3, sizeof(float) * max_points*64);  
    float * output128;
    cudaMalloc((void**)& output128, sizeof(float) * max_points*128);
    float * output128_1;
    cudaMalloc((void**)& output128_1, sizeof(float) * max_points*128);
    float * output128_2;
    cudaMalloc((void**)& output128_2, sizeof(float) * max_points*128);
    float * output1024;
    cudaMalloc((void**)& output1024, sizeof(float) * max_points*1024);
    float * output1024_1;
    cudaMalloc((void**)& output1024_1, sizeof(float) * max_points*1024);
    float * output1024_2;
    cudaMalloc((void**)& output1024_2, sizeof(float) * max_points*1024);

	float *vector1024;
    cudaMalloc((void**)& vector1024, sizeof(float) * 1024);
	float *vector512;
    cudaMalloc((void**)& vector512, sizeof(float) * 512);
    float *vector512_1;
    cudaMalloc((void**)& vector512_1, sizeof(float) * 512);
    float *vector512_2;
    cudaMalloc((void**)& vector512_2, sizeof(float) * 512);
	float *vector256;
    cudaMalloc((void**)& vector256, sizeof(float) * 256);
    float *vector256_1;
    cudaMalloc((void**)& vector256_1, sizeof(float) * 256);
    float *vector256_2;
    cudaMalloc((void**)& vector256_2, sizeof(float) * 256);
	float *vector9;
    cudaMalloc((void**)& vector9, sizeof(float) * 9);
	float *vector64X64;
    cudaMalloc((void**)& vector64X64, sizeof(float) * 64 * 64);
    float *vector10;
    cudaMalloc((void**)& vector10, sizeof(float) * 10);

    //******************************************************************STN3d************************************************************************//
    //******************************************************************matrixTranspose*******************************************************************//

    
    //******************************************************************feat_stn_conv1*******************************************************************//
    int feat_stn_conv1_in_channels = 3;     // 输入通道数量
    int feat_stn_conv1_out_channels = 64;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_stn_conv1_weight;
    float* feat_stn_conv1_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_stn_conv1_weight, feat_stn_conv1_in_channels * feat_stn_conv1_out_channels * sizeof(float)); // 3 x 64 = 192
    cudaMalloc((void**)&feat_stn_conv1_bias, feat_stn_conv1_out_channels * sizeof(float));                 // 64

    // 将这些数据从 host 拷贝到 device

    wbCheck(cudaMemcpy(feat_stn_conv1_weight, &feat_stn_conv1_weight_host[0], feat_stn_conv1_in_channels * feat_stn_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_conv1_bias, &feat_stn_conv1_bias_host[0], feat_stn_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_stn_bn1*******************************************************************//
    int feat_stn_bn1_channels = 64;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_bn1_weight, *feat_stn_bn1_bias, *feat_stn_bn1_mean, *feat_stn_bn1_var;
    cudaMalloc((void**)&feat_stn_bn1_weight, feat_stn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn1_bias, feat_stn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn1_mean, feat_stn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn1_var, feat_stn_bn1_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_bn1_weight, &feat_stn_bn1_weight_host[0], feat_stn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn1_bias, &feat_stn_bn1_bias_host[0], feat_stn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn1_mean, &feat_stn_bn1_mean_host[0], feat_stn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn1_var, &feat_stn_bn1_var_host[0], feat_stn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));




    // //******************************************************************feat_stn_relu1*******************************************************************//
    int feat_stn_relu1_channels = 64;  // 输出通道数量


    //******************************************************************feat_stn_conv2*******************************************************************//
    int feat_stn_conv2_in_channels = 64;     // 输入通道数量
    int feat_stn_conv2_out_channels = 128;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_stn_conv2_weight;
    float* feat_stn_conv2_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_stn_conv2_weight, feat_stn_conv2_in_channels * feat_stn_conv2_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_stn_conv2_bias, feat_stn_conv2_out_channels * sizeof(float));                 

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_stn_conv2_weight, &feat_stn_conv2_weight_host[0], feat_stn_conv2_in_channels * feat_stn_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_conv2_bias, &feat_stn_conv2_bias_host[0], feat_stn_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_stn_bn2*******************************************************************//
    int feat_stn_bn2_channels = 128;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_bn2_weight, *feat_stn_bn2_bias, *feat_stn_bn2_mean, *feat_stn_bn2_var;
    cudaMalloc((void**)&feat_stn_bn2_weight, feat_stn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn2_bias, feat_stn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn2_mean, feat_stn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn2_var, feat_stn_bn2_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_bn2_weight, &feat_stn_bn2_weight_host[0], feat_stn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn2_bias, &feat_stn_bn2_bias_host[0], feat_stn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn2_mean, &feat_stn_bn2_mean_host[0], feat_stn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn2_var, &feat_stn_bn2_var_host[0], feat_stn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_stn_relu2*******************************************************************//
    int feat_stn_relu2_channels = 128;  // 输出通道数量

//******************************************************************feat_stn_conv3*******************************************************************//
    int feat_stn_conv3_in_channels = 128;     // 输入通道数量
    int feat_stn_conv3_out_channels = 1024;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_stn_conv3_weight;
    float* feat_stn_conv3_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_stn_conv3_weight, feat_stn_conv3_in_channels * feat_stn_conv3_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_stn_conv3_bias, feat_stn_conv3_out_channels * sizeof(float));                 

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_stn_conv3_weight, &feat_stn_conv3_weight_host[0], feat_stn_conv3_in_channels * feat_stn_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_conv3_bias, &feat_stn_conv3_bias_host[0], feat_stn_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************feat_stn_bn3*******************************************************************//
    int feat_stn_bn3_channels = 1024;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_bn3_weight, *feat_stn_bn3_bias, *feat_stn_bn3_mean, *feat_stn_bn3_var;
    cudaMalloc((void**)&feat_stn_bn3_weight, feat_stn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn3_bias, feat_stn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn3_mean, feat_stn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn3_var, feat_stn_bn3_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_bn3_weight, &feat_stn_bn3_weight_host[0], feat_stn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn3_bias, &feat_stn_bn3_bias_host[0], feat_stn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn3_mean, &feat_stn_bn3_mean_host[0], feat_stn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn3_var, &feat_stn_bn3_var_host[0], feat_stn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_stn_relu3*******************************************************************//
    int feat_stn_relu3_channels = 1024;  // 输出通道数量


    // //******************************************************************feat_stn_max*******************************************************************//
    int feat_stn_max_channels = 1024;  // 输出通道数量



    // //******************************************************************feat_stn_fc1*******************************************************************//
    int feat_stn_fc1_in_channels = 1024;  // 输入通道数量
    int feat_stn_fc1_out_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_fc1_weight;
    float *feat_stn_fc1_bias;
    cudaMalloc((void**)&feat_stn_fc1_weight, feat_stn_fc1_out_channels * feat_stn_fc1_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_fc1_bias, feat_stn_fc1_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_fc1_weight, &feat_stn_fc1_weight_host[0], feat_stn_fc1_out_channels * feat_stn_fc1_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_fc1_bias, &feat_stn_fc1_bias_host[0], feat_stn_fc1_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_stn_bn4*******************************************************************//
    int feat_stn_bn4_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_bn4_weight, *feat_stn_bn4_bias, *feat_stn_bn4_mean, *feat_stn_bn4_var;
    cudaMalloc((void**)&feat_stn_bn4_weight, feat_stn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn4_bias, feat_stn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn4_mean, feat_stn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn4_var, feat_stn_bn4_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_bn4_weight, &feat_stn_bn4_weight_host[0], feat_stn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn4_bias, &feat_stn_bn4_bias_host[0], feat_stn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn4_mean, &feat_stn_bn4_mean_host[0], feat_stn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn4_var, &feat_stn_bn4_var_host[0], feat_stn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_stn_relu4*******************************************************************//
    int feat_stn_relu4_channels = 512;  // 输出通道数量




    //******************************************************************feat_stn_fc2*******************************************************************//
    int feat_stn_fc2_in_channels = 512;  // 输入通道数量
    int feat_stn_fc2_out_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_fc2_weight;
    float *feat_stn_fc2_bias;
    cudaMalloc((void**)&feat_stn_fc2_weight, feat_stn_fc2_out_channels * feat_stn_fc2_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_fc2_bias, feat_stn_fc2_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_fc2_weight, &feat_stn_fc2_weight_host[0], feat_stn_fc2_out_channels * feat_stn_fc2_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_fc2_bias, &feat_stn_fc2_bias_host[0], feat_stn_fc2_out_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_stn_bn5*******************************************************************//
    int feat_stn_bn5_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_bn5_weight, *feat_stn_bn5_bias, *feat_stn_bn5_mean, *feat_stn_bn5_var;
    cudaMalloc((void**)&feat_stn_bn5_weight, feat_stn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn5_bias, feat_stn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn5_mean, feat_stn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_bn5_var, feat_stn_bn5_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_bn5_weight, &feat_stn_bn5_weight_host[0], feat_stn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn5_bias, &feat_stn_bn5_bias_host[0], feat_stn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn5_mean, &feat_stn_bn5_mean_host[0], feat_stn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_bn5_var, &feat_stn_bn5_var_host[0], feat_stn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_stn_relu5*******************************************************************//
    int feat_stn_relu5_channels = 256;  // 输出通道数量



    //******************************************************************feat_stn_fc3*******************************************************************//
    int feat_stn_fc3_in_channels = 256;  // 输入通道数量
    int feat_stn_fc3_out_channels = 9;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_stn_fc3_weight;
    float *feat_stn_fc3_bias;
    cudaMalloc((void**)&feat_stn_fc3_weight, feat_stn_fc3_out_channels * feat_stn_fc3_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_stn_fc3_bias, feat_stn_fc3_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_stn_fc3_weight, &feat_stn_fc3_weight_host[0], feat_stn_fc3_out_channels * feat_stn_fc3_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_stn_fc3_bias, &feat_stn_fc3_bias_host[0], feat_stn_fc3_out_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_stn_iden*******************************************************************//
    int feat_stn_iden_h = 3;  // 输出矩阵高度
    int feat_stn_iden_w = 3;  // 输出矩阵宽度


    
    //******************************************************************end STNkd*******************************************************************//


    // //******************************************************************matrixMultiply1*******************************************************************//




    //******************************************************************matrixTranspose2*******************************************************************//



//******************************************************************feat_conv1*******************************************************************//
    int feat_conv1_in_channels = 3;     // 输入通道数量
    int feat_conv1_out_channels = 64;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_conv1_weight;
    float* feat_conv1_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_conv1_weight, feat_conv1_in_channels * feat_conv1_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_conv1_bias, feat_conv1_out_channels * sizeof(float));                

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_conv1_weight, &feat_conv1_weight_host[0], feat_conv1_in_channels * feat_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_conv1_bias, &feat_conv1_bias_host[0], feat_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_bn1*******************************************************************//
    int feat_bn1_channels = 64;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_bn1_weight, *feat_bn1_bias, *feat_bn1_mean, *feat_bn1_var;
    cudaMalloc((void**)&feat_bn1_weight, feat_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn1_bias, feat_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn1_mean, feat_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn1_var, feat_bn1_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_bn1_weight, &feat_bn1_weight_host[0], feat_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn1_bias, &feat_bn1_bias_host[0], feat_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn1_mean, &feat_bn1_mean_host[0], feat_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn1_var, &feat_bn1_var_host[0], feat_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_relu1*******************************************************************//
    int feat_relu1_channels = 64;  // 输出通道数量


    //******************************************************************STNkd************************************************************************//

//******************************************************************feat_fstn_conv1*******************************************************************//
    int feat_fstn_conv1_in_channels = 64;     // 输入通道数量
    int feat_fstn_conv1_out_channels = 64;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_fstn_conv1_weight;
    float* feat_fstn_conv1_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_fstn_conv1_weight, feat_fstn_conv1_in_channels * feat_fstn_conv1_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_fstn_conv1_bias, feat_fstn_conv1_out_channels * sizeof(float));                

    // 将这些数据从 host 拷贝到 device

    wbCheck(cudaMemcpy(feat_fstn_conv1_weight, &feat_fstn_conv1_weight_host[0], feat_fstn_conv1_in_channels * feat_fstn_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_conv1_bias, &feat_fstn_conv1_bias_host[0], feat_fstn_conv1_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_bn1*******************************************************************//
    int feat_fstn_bn1_channels = 64;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_bn1_weight, *feat_fstn_bn1_bias, *feat_fstn_bn1_mean, *feat_fstn_bn1_var;
    cudaMalloc((void**)&feat_fstn_bn1_weight, feat_fstn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn1_bias, feat_fstn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn1_mean, feat_fstn_bn1_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn1_var, feat_fstn_bn1_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_bn1_weight, &feat_fstn_bn1_weight_host[0], feat_fstn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn1_bias, &feat_fstn_bn1_bias_host[0], feat_fstn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn1_mean, &feat_fstn_bn1_mean_host[0], feat_fstn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn1_var, &feat_fstn_bn1_var_host[0], feat_fstn_bn1_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_relu1*******************************************************************//
    int feat_fstn_relu1_channels = 64;  // 输出通道数量


    //******************************************************************feat_fstn_conv2*******************************************************************//
    int feat_fstn_conv2_in_channels = 64;     // 输入通道数量
    int feat_fstn_conv2_out_channels = 128;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_fstn_conv2_weight;
    float* feat_fstn_conv2_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_fstn_conv2_weight, feat_fstn_conv2_in_channels * feat_fstn_conv2_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_fstn_conv2_bias, feat_fstn_conv2_out_channels * sizeof(float));                 

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_fstn_conv2_weight, &feat_fstn_conv2_weight_host[0], feat_fstn_conv2_in_channels * feat_fstn_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_conv2_bias, &feat_fstn_conv2_bias_host[0], feat_fstn_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_bn2*******************************************************************//
    int feat_fstn_bn2_channels = 128;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_bn2_weight, *feat_fstn_bn2_bias, *feat_fstn_bn2_mean, *feat_fstn_bn2_var;
    cudaMalloc((void**)&feat_fstn_bn2_weight, feat_fstn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn2_bias, feat_fstn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn2_mean, feat_fstn_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn2_var, feat_fstn_bn2_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_bn2_weight, &feat_fstn_bn2_weight_host[0], feat_fstn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn2_bias, &feat_fstn_bn2_bias_host[0], feat_fstn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn2_mean, &feat_fstn_bn2_mean_host[0], feat_fstn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn2_var, &feat_fstn_bn2_var_host[0], feat_fstn_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_relu2*******************************************************************//
    int feat_fstn_relu2_channels = 128;  // 输出通道数量


//******************************************************************feat_fstn_conv3*******************************************************************//
    int feat_fstn_conv3_in_channels = 128;     // 输入通道数量
    int feat_fstn_conv3_out_channels = 1024;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_fstn_conv3_weight;
    float* feat_fstn_conv3_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_fstn_conv3_weight, feat_fstn_conv3_in_channels * feat_fstn_conv3_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_fstn_conv3_bias, feat_fstn_conv3_out_channels * sizeof(float));                 

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_fstn_conv3_weight, &feat_fstn_conv3_weight_host[0], feat_fstn_conv3_in_channels * feat_fstn_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_conv3_bias, &feat_fstn_conv3_bias_host[0], feat_fstn_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************feat_fstn_bn3*******************************************************************//
    int feat_fstn_bn3_channels = 1024;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_bn3_weight, *feat_fstn_bn3_bias, *feat_fstn_bn3_mean, *feat_fstn_bn3_var;
    cudaMalloc((void**)&feat_fstn_bn3_weight, feat_fstn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn3_bias, feat_fstn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn3_mean, feat_fstn_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn3_var, feat_fstn_bn3_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_bn3_weight, &feat_fstn_bn3_weight_host[0], feat_fstn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn3_bias, &feat_fstn_bn3_bias_host[0], feat_fstn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn3_mean, &feat_fstn_bn3_mean_host[0], feat_fstn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn3_var, &feat_fstn_bn3_var_host[0], feat_fstn_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************feat_fstn_relu3*******************************************************************//
    int feat_fstn_relu3_channels = 1024;  // 输出通道数量

    // //******************************************************************feat_fstn_max*******************************************************************//
    int feat_fstn_max_channels = 1024;  // 输出通道数量



    // //******************************************************************feat_fstn_fc1*******************************************************************//
    int feat_fstn_fc1_in_channels = 1024;  // 输入通道数量
    int feat_fstn_fc1_out_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_fc1_weight;
    float *feat_fstn_fc1_bias;
    cudaMalloc((void**)&feat_fstn_fc1_weight, feat_fstn_fc1_out_channels * feat_fstn_fc1_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_fc1_bias, feat_fstn_fc1_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_fc1_weight, &feat_fstn_fc1_weight_host[0], feat_fstn_fc1_out_channels * feat_fstn_fc1_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_fc1_bias, &feat_fstn_fc1_bias_host[0], feat_fstn_fc1_out_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_fstn_bn4*******************************************************************//
    int feat_fstn_bn4_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_bn4_weight, *feat_fstn_bn4_bias, *feat_fstn_bn4_mean, *feat_fstn_bn4_var;
    cudaMalloc((void**)&feat_fstn_bn4_weight, feat_fstn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn4_bias, feat_fstn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn4_mean, feat_fstn_bn4_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn4_var, feat_fstn_bn4_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_bn4_weight, &feat_fstn_bn4_weight_host[0], feat_fstn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn4_bias, &feat_fstn_bn4_bias_host[0], feat_fstn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn4_mean, &feat_fstn_bn4_mean_host[0], feat_fstn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn4_var, &feat_fstn_bn4_var_host[0], feat_fstn_bn4_channels * sizeof(float), cudaMemcpyHostToDevice));



    // //******************************************************************feat_fstn_relu4*******************************************************************//
    int feat_fstn_relu4_channels = 512;  // 输出通道数量


    //******************************************************************feat_fstn_fc2*******************************************************************//
    int feat_fstn_fc2_in_channels = 512;  // 输入通道数量
    int feat_fstn_fc2_out_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_fc2_weight;
    float *feat_fstn_fc2_bias;
    cudaMalloc((void**)&feat_fstn_fc2_weight, feat_fstn_fc2_out_channels * feat_fstn_fc2_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_fc2_bias, feat_fstn_fc2_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_fc2_weight, &feat_fstn_fc2_weight_host[0], feat_fstn_fc2_out_channels * feat_fstn_fc2_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_fc2_bias, &feat_fstn_fc2_bias_host[0], feat_fstn_fc2_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_bn5*******************************************************************//
    int feat_fstn_bn5_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_bn5_weight, *feat_fstn_bn5_bias, *feat_fstn_bn5_mean, *feat_fstn_bn5_var;
    cudaMalloc((void**)&feat_fstn_bn5_weight, feat_fstn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn5_bias, feat_fstn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn5_mean, feat_fstn_bn5_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_bn5_var, feat_fstn_bn5_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_bn5_weight, &feat_fstn_bn5_weight_host[0], feat_fstn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn5_bias, &feat_fstn_bn5_bias_host[0], feat_fstn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn5_mean, &feat_fstn_bn5_mean_host[0], feat_fstn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_bn5_var, &feat_fstn_bn5_var_host[0], feat_fstn_bn5_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_fstn_relu5*******************************************************************//
    int feat_fstn_relu5_channels = 256;  // 输出通道数量



    //******************************************************************feat_fstn_fc3*******************************************************************//
    int feat_fstn_fc3_in_channels = 256;  // 输入通道数量
    int feat_fstn_fc3_out_channels = 64*64;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_fstn_fc3_weight;
    float *feat_fstn_fc3_bias;
    cudaMalloc((void**)&feat_fstn_fc3_weight, feat_fstn_fc3_out_channels * feat_fstn_fc3_in_channels * sizeof(float));
    cudaMalloc((void**)&feat_fstn_fc3_bias, feat_fstn_fc3_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_fstn_fc3_weight, &feat_fstn_fc3_weight_host[0], feat_fstn_fc3_out_channels * feat_fstn_fc3_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_fstn_fc3_bias, &feat_fstn_fc3_bias_host[0], feat_fstn_fc3_out_channels * sizeof(float), cudaMemcpyHostToDevice));

    //******************************************************************feat_fstn_iden*******************************************************************//
    int feat_fstn_iden_h = 64;  // 输出矩阵高度
    int feat_fstn_iden_w = 64;  // 输出矩阵宽度



    //******************************************************************end STNkd*******************************************************************//
    
    //******************************************************************matrixTranspose3*******************************************************************//


    //******************************************************************matrixMultiply2*******************************************************************//



    //******************************************************************matrixTranspose4*******************************************************************//



    //******************************************************************feat_conv2*******************************************************************//
    int feat_conv2_in_channels = 64;     // 输入通道数量
    int feat_conv2_out_channels = 128;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_conv2_weight;
    float* feat_conv2_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_conv2_weight, feat_conv2_in_channels * feat_conv2_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_conv2_bias, feat_conv2_out_channels * sizeof(float));                

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_conv2_weight, &feat_conv2_weight_host[0], feat_conv2_in_channels * feat_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_conv2_bias, &feat_conv2_bias_host[0], feat_conv2_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_bn2*******************************************************************//
    int feat_bn2_channels = 128;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_bn2_weight, *feat_bn2_bias, *feat_bn2_mean, *feat_bn2_var;
    cudaMalloc((void**)&feat_bn2_weight, feat_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn2_bias, feat_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn2_mean, feat_bn2_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn2_var, feat_bn2_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_bn2_weight, &feat_bn2_weight_host[0], feat_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn2_bias, &feat_bn2_bias_host[0], feat_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn2_mean, &feat_bn2_mean_host[0], feat_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn2_var, &feat_bn2_var_host[0], feat_bn2_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************feat_relu2*******************************************************************//
    int feat_relu2_channels = 128;  // 输出通道数量

//******************************************************************feat_conv3*******************************************************************//
    int feat_conv3_in_channels = 128;     // 输入通道数量
    int feat_conv3_out_channels = 1024;   // 输出通道数量

    // 为输入、输出、权重和偏置分配内存
    float* feat_conv3_weight;
    float* feat_conv3_bias;

    // CUDA 设备内存分配
    cudaMalloc((void**)&feat_conv3_weight, feat_conv3_in_channels * feat_conv3_out_channels * sizeof(float)); 
    cudaMalloc((void**)&feat_conv3_bias, feat_conv3_out_channels * sizeof(float));                

    // 将这些数据从 host 拷贝到 device
    wbCheck(cudaMemcpy(feat_conv3_weight, &feat_conv3_weight_host[0], feat_conv3_in_channels * feat_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_conv3_bias, &feat_conv3_bias_host[0], feat_conv3_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************feat_bn3*******************************************************************//
    int feat_bn3_channels = 1024;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *feat_bn3_weight, *feat_bn3_bias, *feat_bn3_mean, *feat_bn3_var;
    cudaMalloc((void**)&feat_bn3_weight, feat_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn3_bias, feat_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn3_mean, feat_bn3_channels * sizeof(float));
    cudaMalloc((void**)&feat_bn3_var, feat_bn3_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(feat_bn3_weight, &feat_bn3_weight_host[0], feat_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn3_bias, &feat_bn3_bias_host[0], feat_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn3_mean, &feat_bn3_mean_host[0], feat_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(feat_bn3_var, &feat_bn3_var_host[0], feat_bn3_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************feat_max*******************************************************************//
    int feat_max_channels = 1024;  // 输出通道数量


    //******************************************************************fc1*******************************************************************//
    int fc1_in_channels = 1024;  // 输入通道数量
    int fc1_out_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *fc1_weight;
    float *fc1_bias;
    cudaMalloc((void**)&fc1_weight, fc1_out_channels * fc1_in_channels * sizeof(float));
    cudaMalloc((void**)&fc1_bias, fc1_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(fc1_weight, &fc1_weight_host_host[0], fc1_out_channels * fc1_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(fc1_bias, &fc1_bias_host_host[0], fc1_out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // //******************************************************************bn1*******************************************************************//
    int bn1_channels = 512;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *bn1_weight, *bn1_bias, *bn1_mean, *bn1_var;
    cudaMalloc((void**)&bn1_weight, bn1_channels * sizeof(float));
    cudaMalloc((void**)&bn1_bias, bn1_channels * sizeof(float));
    cudaMalloc((void**)&bn1_mean, bn1_channels * sizeof(float));
    cudaMalloc((void**)&bn1_var, bn1_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(bn1_weight, &bn1_weight_host[0], bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn1_bias, &bn1_bias_host[0], bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn1_mean, &bn1_mean_host[0], bn1_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn1_var, &bn1_var_host[0], bn1_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************relu1*******************************************************************//
    int relu1_channels = 512;  // 输出通道数量


    //******************************************************************fc2*******************************************************************//
    int fc2_in_channels = 512;  // 输入通道数量
    int fc2_out_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *fc2_weight;
    float *fc2_bias;
    cudaMalloc((void**)&fc2_weight, fc2_out_channels * fc2_in_channels * sizeof(float));
    cudaMalloc((void**)&fc2_bias, fc2_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(fc2_weight, &fc2_weight_host[0], fc2_out_channels * fc2_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(fc2_bias, &fc2_bias_host[0], fc2_out_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************bn2*******************************************************************//
    int bn2_channels = 256;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *bn2_weight, *bn2_bias, *bn2_mean, *bn2_var;
    cudaMalloc((void**)&bn2_weight, bn2_channels * sizeof(float));
    cudaMalloc((void**)&bn2_bias, bn2_channels * sizeof(float));
    cudaMalloc((void**)&bn2_mean, bn2_channels * sizeof(float));
    cudaMalloc((void**)&bn2_var, bn2_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(bn2_weight, &bn2_weight_host[0], bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn2_bias, &bn2_bias_host[0], bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn2_mean, &bn2_mean_host[0], bn2_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(bn2_var, &bn2_var_host[0], bn2_channels * sizeof(float), cudaMemcpyHostToDevice));


    // //******************************************************************relu2*******************************************************************//
    int relu2_channels = 256;  // 输出通道数量



    //******************************************************************fc3*******************************************************************//
    int fc3_in_channels = 256;  // 输入通道数量
    int fc3_out_channels = 10;  // 输出通道数量

    // 分配 CUDA 设备内存
    float *fc3_weight;
    float *fc3_bias;
    cudaMalloc((void**)&fc3_weight, fc3_out_channels * fc3_in_channels * sizeof(float));
    cudaMalloc((void**)&fc3_bias, fc3_out_channels * sizeof(float));

    // 从主机拷贝数据到设备
    wbCheck(cudaMemcpy(fc3_weight, &fc3_weight_host[0], fc3_out_channels * fc3_in_channels * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(fc3_bias, &fc3_bias_host[0], fc3_out_channels * sizeof(float), cudaMemcpyHostToDevice));




    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < list_of_points.size(); i++)
    {
        // TODO ...在这里实现利用CUDA对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
        // 打印每一帧的数据，仅用于调试！
    
        // std::cout << "Points " << i << ": ";
        // // for (const auto& point : list_of_points[i]) {
        // //     std::cout << point << " ";
        // // }
        // std::cout << "\nLabel: " << list_of_labels[i] << std::endl;



        int num_points=list_of_points[i].size()/3;
        
        const int Grid=512;
        const int Block3=(num_points *3 +Grid -1 )/Grid; //3*num_points
        const int Block64=(num_points *64 +Grid -1 )/Grid;
        const int Block128=(num_points *128 +Grid -1 )/Grid;
        const int Block1024=(num_points *1024 +Grid -1 )/Grid;

        size_t sharedMemSize = (num_points + 1024) * sizeof(float);

        // 调用 CUDA核函数
        //matrixTranspose1
        wbCheck(cudaMemcpy(input, list_of_points[i].data(), num_points * 3 * sizeof(float), cudaMemcpyHostToDevice));

        matrixTranspose<<<Block3, Grid>>>(input, Tinput,num_points,3);
        //wbCheck(cudaGetLastError());
        //wbCheck(cudaDeviceSynchronize());
    

        //******************************************************************STN3d************************************************************************//
        //feat_stn_conv1
        Convolution<<<Block64, Grid, sharedMemSize>>>(Tinput, output64, feat_stn_conv1_weight, feat_stn_conv1_bias, num_points, feat_stn_conv1_in_channels, feat_stn_conv1_out_channels);

        //feat_stn_bn1
        BatchNorm<<<Block64, Grid>>>(output64, output64_2, feat_stn_bn1_weight,  feat_stn_bn1_bias, feat_stn_bn1_mean, feat_stn_bn1_var,num_points, feat_stn_bn1_channels);

        //feat_stn_relu1
        Relu<<<Block64, Grid>>>(output64_2, output64_3, num_points, feat_stn_relu1_channels);

        //feat_stn_conv2
        Convolution<<<Block128, Grid, sharedMemSize>>>(output64_3, output128, feat_stn_conv2_weight, feat_stn_conv2_bias, num_points, feat_stn_conv2_in_channels, feat_stn_conv2_out_channels);
    
        //feat_stn_bn2
        BatchNorm<<<Block128, Grid>>>(output128, output128_1, feat_stn_bn2_weight,  feat_stn_bn2_bias, feat_stn_bn2_mean, feat_stn_bn2_var,num_points, feat_stn_bn2_channels);

        //feat_stn_relu2
        Relu<<<Block128, Grid>>>(output128_1, output128_2, num_points, feat_stn_relu2_channels);
    
        //feat_stn_conv3
        Convolution<<<Block1024, Grid, sharedMemSize>>>(output128_2, output1024, feat_stn_conv3_weight, feat_stn_conv3_bias, num_points, feat_stn_conv3_in_channels, feat_stn_conv3_out_channels);
    
        //feat_stn_bn3
        BatchNorm<<<Block1024, Grid>>>(output1024, output1024_1, feat_stn_bn3_weight,  feat_stn_bn3_bias, feat_stn_bn3_mean, feat_stn_bn3_var,num_points, feat_stn_bn3_channels);

        //feat_stn_relu3
        Relu<<<Block1024, Grid>>>(output1024_1, output1024_2, num_points, feat_stn_relu3_channels);
  
        //feat_stn_max
        MaxPool<<<1, 1024>>>(output1024_2, vector1024, num_points, feat_stn_max_channels);
                
        // float* host_x;
        // host_x = (float*)malloc(1024* num_points*sizeof(float));
        // cudaMemcpy(host_x, output1024_2,  1024* num_points*sizeof(float), cudaMemcpyDeviceToHost);
        // for (int x = 0; x < num_points; x++){
        //     std::cout << host_x[x] << ",";

        // }
        //feat_stn_fc1
        FC<<<1, 512>>>(vector1024, vector512, feat_stn_fc1_weight, feat_stn_fc1_bias,feat_stn_fc1_in_channels,feat_stn_fc1_out_channels);

        //feat_stn_bn4
        BatchNorm<<<1, 512>>>(vector512, vector512_1, feat_stn_bn4_weight,  feat_stn_bn4_bias, feat_stn_bn4_mean, feat_stn_bn4_var,1, feat_stn_bn4_channels);

        //feat_stn_relu4
        Relu<<<1, 512>>>(vector512_1, vector512_2, 1, feat_stn_relu4_channels);

        //feat_stn_fc2
        FC<<<1, 256>>>(vector512_2, vector256, feat_stn_fc2_weight, feat_stn_fc2_bias,feat_stn_fc2_in_channels,feat_stn_fc2_out_channels);

        //feat_stn_bn5
        BatchNorm<<<1, 256>>>(vector256, vector256_1, feat_stn_bn5_weight,  feat_stn_bn5_bias, feat_stn_bn5_mean, feat_stn_bn5_var,1, feat_stn_bn5_channels);

        //feat_stn_relu5
        Relu<<<1, 256>>>(vector256_1, vector256_2, 1, feat_stn_relu5_channels);

        //feat_stn_fc3
        FC<<<1, 9>>>(vector256_2, vector9, feat_stn_fc3_weight, feat_stn_fc3_bias,feat_stn_fc3_in_channels,feat_stn_fc3_out_channels);

        //feat_stn_iden
        Iden<<<1, 9>>>(vector9, feat_stn_iden_h, feat_stn_iden_w);

        //matrixMultiply1
        matrixMultiply<<<Block3, Grid>>>(input, vector9, Tinput, num_points, 3, 3);

        //matrixTranspose2
 
        matrixTranspose<<<Block3, Grid>>>(Tinput, input,num_points,3);

        //feat_conv1
        Convolution<<<Block64, Grid, sharedMemSize>>>(input, Toutput64, feat_conv1_weight, feat_conv1_bias, num_points, feat_conv1_in_channels, feat_conv1_out_channels);
        
        //feat_bn1
        BatchNorm<<<Block64, Grid>>>(Toutput64, Toutput64_2, feat_bn1_weight,  feat_bn1_bias, feat_bn1_mean, feat_bn1_var,num_points, feat_bn1_channels);
    
        //feat_relu1
        Relu<<<Block64, Grid>>>(Toutput64_2, Toutput64_3, num_points, feat_relu1_channels);

        cudaMemcpy(output64,Toutput64_3,num_points*64*sizeof(float),cudaMemcpyDeviceToDevice);

        //******************************************************************STNkd************************************************************************//
        //feat_fstn_conv1
        Convolution<<<Block64, Grid, sharedMemSize>>>(Toutput64_3, Toutput64_2, feat_fstn_conv1_weight, feat_fstn_conv1_bias, num_points, feat_fstn_conv1_in_channels, feat_fstn_conv1_out_channels);
        
        //feat_fstn_bn1
        BatchNorm<<<Block64, Grid>>>(Toutput64_2, Toutput64, feat_fstn_bn1_weight,  feat_fstn_bn1_bias, feat_fstn_bn1_mean, feat_fstn_bn1_var,num_points, feat_fstn_bn1_channels);
    
        //feat_fstn_relu1
        Relu<<<Block64, Grid>>>(Toutput64, Toutput64_3, num_points, feat_fstn_relu1_channels);
    
        //feat_fstn_conv2
        Convolution<<<Block128, Grid, sharedMemSize>>>(Toutput64_3, output128, feat_fstn_conv2_weight, feat_fstn_conv2_bias, num_points, feat_fstn_conv2_in_channels, feat_fstn_conv2_out_channels);
    
        //feat_fstn_bn2
        BatchNorm<<<Block128, Grid>>>(output128, output128_1, feat_fstn_bn2_weight,  feat_fstn_bn2_bias, feat_fstn_bn2_mean, feat_fstn_bn2_var,num_points, feat_fstn_bn2_channels);

        //feat_fstn_relu2
        Relu<<<Block128, Grid>>>(output128_1, output128_2, num_points, feat_fstn_relu2_channels);
    
        //feat_fstn_conv3
        Convolution<<<Block1024, Grid, sharedMemSize>>>(output128_2, output1024, feat_fstn_conv3_weight, feat_fstn_conv3_bias, num_points, feat_fstn_conv3_in_channels, feat_fstn_conv3_out_channels);
    
        //feat_fstn_bn3
        BatchNorm<<<Block1024, Grid>>>(output1024, output1024_1, feat_fstn_bn3_weight,  feat_fstn_bn3_bias, feat_fstn_bn3_mean, feat_fstn_bn3_var,num_points, feat_fstn_bn3_channels);

        //feat_fstn_relu3
        Relu<<<Block1024, Grid>>>(output1024_1, output1024_2, num_points, feat_fstn_relu3_channels);

        //feat_fstn_max
        MaxPool<<<1, 1024>>>(output1024_2, vector1024, num_points, feat_fstn_max_channels);

        //feat_fstn_fc1
        FC<<<1, 512>>>(vector1024, vector512, feat_fstn_fc1_weight, feat_fstn_fc1_bias,feat_fstn_fc1_in_channels,feat_fstn_fc1_out_channels);

        //feat_fstn_bn4
        BatchNorm<<<1, 512>>>(vector512, vector512_1, feat_fstn_bn4_weight,  feat_fstn_bn4_bias, feat_fstn_bn4_mean, feat_fstn_bn4_var,1, feat_fstn_bn4_channels);

        //feat_fstn_relu4
        Relu<<<1, 512>>>(vector512_1, vector512_2, 1, feat_fstn_relu4_channels);

        //feat_fstn_fc2
        FC<<<1, 256>>>(vector512_2, vector256, feat_fstn_fc2_weight, feat_fstn_fc2_bias,feat_fstn_fc2_in_channels,feat_fstn_fc2_out_channels);

        //feat_fstn_bn5
        BatchNorm<<<1, 256>>>(vector256, vector256_1, feat_fstn_bn5_weight,  feat_fstn_bn5_bias, feat_fstn_bn5_mean, feat_fstn_bn5_var,1, feat_fstn_bn5_channels);

        //feat_fstn_relu5
        Relu<<<1, 256>>>(vector256_1, vector256_2, 1, feat_fstn_relu5_channels);

        //feat_fstn_fc3
        FC<<<4, 1024>>>(vector256_2, vector64X64, feat_fstn_fc3_weight, feat_fstn_fc3_bias,feat_fstn_fc3_in_channels,feat_fstn_fc3_out_channels);

        //feat_fstn_iden
        Iden<<<4, 1024>>>(vector64X64, feat_fstn_iden_h, feat_fstn_iden_w);

        //******************************************************************end STNkd************************************************************************//
        //matrixTranspose3
        matrixTranspose<<<Block64, Grid>>>(output64, Toutput64,64,num_points);
        
        //matrixMultiply2
        matrixMultiply<<<Block64, Grid>>>(Toutput64, vector64X64, output64, num_points, 64, 64);

        //matrixTranspose4
        matrixTranspose<<<Block64, Grid>>>(output64, Toutput64,num_points,64);

        //feat_conv2
        Convolution<<<Block128, Grid, sharedMemSize>>>(Toutput64, output128, feat_conv2_weight, feat_conv2_bias, num_points, feat_conv2_in_channels, feat_conv2_out_channels);
        
        //feat_bn2
        BatchNorm<<<Block128, Grid>>>(output128, output128_1, feat_bn2_weight,  feat_bn2_bias, feat_bn2_mean, feat_bn2_var,num_points, feat_bn2_channels);
    
        //feat_relu2
        Relu<<<Block128, Grid>>>(output128_1, output128_2, num_points, feat_relu2_channels);

        //feat_conv3
        Convolution<<<Block1024, Grid, sharedMemSize>>>(output128_2, output1024, feat_conv3_weight, feat_conv3_bias, num_points, feat_conv3_in_channels, feat_conv3_out_channels);
        
        //feat_bn3
        BatchNorm<<<Block1024, Grid>>>(output1024, output1024_1, feat_bn3_weight,  feat_bn3_bias, feat_bn3_mean, feat_bn3_var,num_points, feat_bn3_channels);
    
        //feat_max
        MaxPool<<<1, 1024>>>(output1024_1, vector1024, num_points, feat_max_channels);

        //******************************************************************end PointNetEncoder************************************************************************//

        //fc1
        FC<<<1, 512>>>(vector1024, vector512, fc1_weight, fc1_bias, fc1_in_channels, fc1_out_channels);

        //bn1
        BatchNorm<<<1, 512>>>(vector512, vector512_1, bn1_weight,  bn1_bias, bn1_mean, bn1_var,1, bn1_channels);

        //relu1
        Relu<<<1, 512>>>(vector512_1, vector512_2, 1, relu1_channels);


        //fc2
        FC<<<1, 256>>>(vector512_2, vector256, fc2_weight, fc2_bias, fc2_in_channels, fc2_out_channels);

        //bn2
        BatchNorm<<<1, 256>>>(vector256, vector256_1, bn2_weight,  bn2_bias, bn2_mean, bn2_var,1, bn2_channels);

        //relu2
        Relu<<<1, 256>>>(vector256_1, vector256_2, 1, relu2_channels);

        //fc3
        FC<<<1, 10>>>(vector256_2, vector10, fc3_weight, fc3_bias, fc3_in_channels, fc3_out_channels);





        //获取推理结果
        float max = 0;
        float* host_output;
        host_output = (float*)malloc(10 * sizeof(float));
        cudaMemcpy(host_output, vector10, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    
    
        for (int j = 0; j < 10; j++){
            std::cout << host_output[j] << ",";
            if(host_output[j] > max)
            {
                max = host_output[j];
                result = j;
            }
        }

        std::cout << std::endl;


        if(result == list_of_labels[i])
        {
            num++;
        }


        
    }
    
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":"<< num*1.0/list_of_points.size();




    // 清理分配的内存
    

    cudaFree(feat_stn_conv1_weight);
    cudaFree(feat_stn_conv1_bias);
    cudaFree(feat_stn_bn1_weight);
    cudaFree(feat_stn_bn1_bias);
    cudaFree(feat_stn_bn1_mean);
    cudaFree(feat_stn_bn1_var);

    cudaFree(feat_stn_conv2_weight);
    cudaFree(feat_stn_conv2_bias);
    cudaFree(feat_stn_bn2_weight);
    cudaFree(feat_stn_bn2_bias);
    cudaFree(feat_stn_bn2_mean);
    cudaFree(feat_stn_bn2_var);

    cudaFree(feat_stn_conv3_weight);
    cudaFree(feat_stn_conv3_bias);
    cudaFree(feat_stn_bn3_weight);
    cudaFree(feat_stn_bn3_bias);
    cudaFree(feat_stn_bn3_mean);
    cudaFree(feat_stn_bn3_var);

 
    cudaFree(feat_stn_fc1_weight);
    cudaFree(feat_stn_fc1_bias);

    cudaFree(feat_stn_bn4_weight);
    cudaFree(feat_stn_bn4_bias);
    cudaFree(feat_stn_bn4_mean);
    cudaFree(feat_stn_bn4_var);

    cudaFree(feat_stn_fc2_weight);
    cudaFree(feat_stn_fc2_bias);

    cudaFree(feat_stn_bn5_weight);
    cudaFree(feat_stn_bn5_bias);
    cudaFree(feat_stn_bn5_mean);
    cudaFree(feat_stn_bn5_var);



    cudaFree(feat_stn_fc3_weight);
    cudaFree(feat_stn_fc3_bias);


    

    cudaFree(feat_conv1_weight);
    cudaFree(feat_conv1_bias);



    cudaFree(feat_fstn_conv1_weight);
    cudaFree(feat_fstn_conv1_bias);

    cudaFree(feat_fstn_bn1_weight);
    cudaFree(feat_fstn_bn1_bias);
    cudaFree(feat_fstn_bn1_mean);
    cudaFree(feat_fstn_bn1_var);


    cudaFree(feat_fstn_conv2_weight);
    cudaFree(feat_fstn_conv2_bias);

    cudaFree(feat_fstn_bn2_weight);
    cudaFree(feat_fstn_bn2_bias);
    cudaFree(feat_fstn_bn2_mean);
    cudaFree(feat_fstn_bn2_var);



    cudaFree(feat_fstn_conv3_weight);
    cudaFree(feat_fstn_conv3_bias);

    cudaFree(feat_fstn_bn3_weight);
    cudaFree(feat_fstn_bn3_bias);
    cudaFree(feat_fstn_bn3_mean);
    cudaFree(feat_fstn_bn3_var);




    cudaFree(feat_fstn_fc1_weight);
    cudaFree(feat_fstn_fc1_bias);

    cudaFree(feat_fstn_bn4_weight);
    cudaFree(feat_fstn_bn4_bias);
    cudaFree(feat_fstn_bn4_mean);
    cudaFree(feat_fstn_bn4_var);



    cudaFree(feat_fstn_fc2_weight);
    cudaFree(feat_fstn_fc2_bias);

    cudaFree(feat_fstn_bn5_weight);
    cudaFree(feat_fstn_bn5_bias);
    cudaFree(feat_fstn_bn5_mean);
    cudaFree(feat_fstn_bn5_var);


 
    cudaFree(feat_fstn_fc3_weight);
    cudaFree(feat_fstn_fc3_bias);




    cudaFree(feat_conv2_weight);
    cudaFree(feat_conv2_bias);

    cudaFree(feat_bn2_weight);
    cudaFree(feat_bn2_bias);
    cudaFree(feat_bn2_mean);
    cudaFree(feat_bn2_var);


    cudaFree(feat_conv3_weight);
    cudaFree(feat_conv3_bias);

    cudaFree(feat_bn3_weight);
    cudaFree(feat_bn3_bias);
    cudaFree(feat_bn3_mean);
    cudaFree(feat_bn3_var);


    cudaFree(fc1_weight);
    cudaFree(fc1_bias);

    cudaFree(bn1_weight);
    cudaFree(bn1_bias);
    cudaFree(bn1_mean);
    cudaFree(bn1_var);

    cudaFree(fc2_weight);
    cudaFree(fc2_bias);

    cudaFree(bn2_weight);
    cudaFree(bn2_bias);
    cudaFree(bn2_mean);
    cudaFree(bn2_var);


    cudaFree(fc3_weight);
    cudaFree(fc3_bias);

	cudaFree(input);
	cudaFree(Tinput);
    cudaFree(output64);
    cudaFree(output64_2);
    cudaFree(output64_3);   
    cudaFree(Toutput64);    
    cudaFree(Toutput64_2);
    cudaFree(Toutput64_3);  
    cudaFree(output128);
    cudaFree(output128_1);
    cudaFree(output128_2);
    cudaFree(output1024);
    cudaFree(output1024_1);
    cudaFree(output1024_2);
    cudaFree(vector1024);
    cudaFree(vector512);
    cudaFree(vector512_1);
    cudaFree(vector512_2);
    cudaFree(vector256);
    cudaFree(vector256_1);
    cudaFree(vector256_2);
    cudaFree(vector9);
    cudaFree(vector64X64);







    return 0;
}