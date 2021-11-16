#include <Windows.h>
#include <iostream>
#include <set>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <vector>

template<typename _Type>
std::allocator<_Type> allocator = std::allocator<_Type>();

cudaError_t addWithCuda(int* valueBuffer);


constexpr static size_t thrCudaNum = 512;
constexpr static size_t currentStepValuesSize = thrCudaNum * ((0xFFFFFFFFU >> 1) / thrCudaNum + 1);
constexpr size_t arraySize = 32;

constexpr unsigned int globalValueLimit = 0xFFFFFFFFU >> 1;

__forceinline__ __device__ char getBitValueAt(const unsigned& step, const unsigned char nIdx)
{
    return (((char*)(&step))[nIdx / static_cast<const unsigned char>(8)] >> nIdx % static_cast<const unsigned char>(8)) & 1;
}



__global__ void addKernel(int* valueBuffer, unsigned* currentStepValues, unsigned char* valueBufferSize)
{
    //curand()
    int value = 4;

    for (unsigned char j = 0; j < 31; j++)
    {
        switch (getBitValueAt(currentStepValues[threadIdx.x], j))
        {
        case 0:
            value += 5;
            break;
        case 1:
            value -= 3;
            break;

        }
    }

    
    currentStepValues[threadIdx.x]++;

    for (size_t j = 0; j < valueBufferSize[threadIdx.x]; j++)
    {
        if (value == valueBuffer[arraySize * threadIdx.x + j])
        {
            return;
        }
    }
    valueBuffer[arraySize * threadIdx.x + valueBufferSize[threadIdx.x]] = value;
    valueBufferSize[threadIdx.x]++;

    

}

inline int getSPcores(const cudaDeviceProp& devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

cudaDeviceProp dev_prop;

int main()
{

    cudaGetDeviceProperties(&dev_prop, 0);

    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_FONT_INFOEX cfi;
    cfi.cbSize = sizeof(cfi);
    cfi.nFont = 0;
    cfi.dwFontSize.X = 0;                   // Width of each character in the font
    cfi.dwFontSize.Y = 24;                  // Height
    cfi.FontFamily = FF_DONTCARE;
    cfi.FontWeight = FW_NORMAL;
    std::wcscpy(cfi.FaceName, L"Consolas"); // Choose your font
    SetCurrentConsoleFontEx(hConsole, FALSE, &cfi);

    setlocale(LC_ALL, "RU");
    SetConsoleTextAttribute(hConsole, 10);
    std::cout << "Сейчас ваша видеокарта пройдёт 10 тестов. Будет вычилсяться " << currentStepValuesSize << " чисел; \n\n";

    SetConsoleTextAttribute(hConsole, 12);
    std::cout << "Информация о видеокарте: \n";

    std::cout << "Ваша видеокарта: " << dev_prop.name << std::endl;
    std::cout << "Ваша видеокарта имеет: " << getSPcores(dev_prop) << " ядер;" << std::endl;
    std::cout << "Всего: " << (float)total / 1024.0F / 1024.0F / 1024.0F << "ГБ оперативной видеопамяти;" << std::endl;
    std::cout << "Свободно: " << (float)free / 1024.0F / 1024.0F / 1024.0F << "ГБ оперативной видеопамяти;" << std::endl;
    std::cout << "Занято: " << (float)(total - free) / 1024.0F / 1024.0F / 1024.0F << "ГБ оперативной видеопамяти;" << std::endl;
    std::cout << "Частота вашей видеокарты: " << (float)dev_prop.clockRate / 1000.0F << "МГц;" << std::endl;
    std::cout << "Пиковая тактовая частота вашей оперативной видеопамяти: " << (float)dev_prop.memoryClockRate / 1000 << "МГц;" << std::endl;
    std::cout << "Размер L2-кэша: " << (float)dev_prop.l2CacheSize / 1024.0F << "КБ;" << std::endl;

    std::cout << "Ваша видеокарта" << (dev_prop.integrated ? " " : " не ") << "встроена в материнскую плату;" << std::endl;
    std::cout << "Ваша видеокарта" << (dev_prop.concurrentKernels ? " " : " не ") << "поддерживает мультиядерные вычисления;" << std::endl;
    std::cout << "Ваша видеокарта" << (dev_prop.ECCEnabled ? " " : " не ") << "серверная;" << std::endl;
    std::cout << "Ваша видеокарта" << (dev_prop.tccDriver ? " " : " не ") << "использует TCC драйвер;" << std::endl;
    std::cout << "Эта видеокарта" << (dev_prop.isMultiGpuBoard ? " " : " не ") << "единственная на вашей материнской плате;" << std::endl;

    std::cout << "\n\n";

    int* c = allocator<int>.allocate(arraySize * thrCudaNum);
    SetConsoleTextAttribute(hConsole, 11);
    cudaError_t cudaStatus;
    // Add vectors in parallel.
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Тест №" << i << ": ";
        cudaStatus = addWithCuda(c);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }
    }

    std::vector<int> vector;

    for (size_t i = 0; i < arraySize * thrCudaNum; i++)
    {
        vector.push_back(c[i]);
    }

    for (decltype(auto) i : std::set< int >{ vector.begin(), vector.end() })
    {
        if(i != 0)
			std::cout << i << " ";
    }

    std::cout << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    allocator<int>.deallocate(c, arraySize * thrCudaNum);

    while (true){}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* valueBuffer)
{
    
    int* dev_valueBuffer = nullptr;
    unsigned int* dev_currentStepValues = nullptr;
    unsigned char* dev_valueBufferSize = nullptr;

    unsigned int* currentStepValues = allocator<unsigned int>.allocate(thrCudaNum);

    for (size_t i = 0; i < thrCudaNum; i++)
    {
        currentStepValues[i] = i * (unsigned)(currentStepValuesSize / 1024ULL);
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_valueBuffer, arraySize * thrCudaNum * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_currentStepValues, thrCudaNum * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_valueBufferSize, thrCudaNum * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }




    cudaStatus = cudaMemcpy(dev_currentStepValues, currentStepValues, thrCudaNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //allocator<unsigned int>.deallocate(currentStepValues, currentStepValuesSize);


    int t1 = clock();

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <currentStepValuesSize / 1024, 1024 >> > (dev_valueBuffer, dev_currentStepValues, dev_valueBufferSize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    int t2 = clock() - t1;
    std::cout << "Время: " << (double)t2 / 1000. << " секунд" << std::endl;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(valueBuffer, dev_valueBuffer, arraySize * thrCudaNum * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(dev_valueBuffer);
    cudaFree(dev_currentStepValues);

    return cudaStatus;
}
