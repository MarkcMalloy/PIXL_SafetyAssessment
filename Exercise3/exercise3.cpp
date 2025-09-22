#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>

using namespace cv;
using std::cout;
using std::endl;

// C++ Compile commands:
//# Linux/macOS with pkg-config: g++ -std=c++17 manual_filter.cpp -o manual_filter `pkg-config --cflags --libs opencv4`
// Windows:
// filter2DManual
Mat convolutionFiltering2D(const Mat& src, const Mat& kernel, bool normalize = false,
                   int borderType = BORDER_REFLECT101)
{
    if (kernel.empty())
        throw invalid_argument("Kernel is empty.");
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0)
        throw invalid_argument("Kernel must have odd dimensions (2n+1 x 2m+1).");

    // Convert image to float for calculation
    Mat srcFloat;
    src.convertTo(srcFloat, CV_MAKETYPE(CV_32F, src.channels()));

    // Kernel to float
    Mat k32;
    kernel.convertTo(k32, CV_32F);

    // Normalize kernel if requested
    if (normalize) {
        double s = sum(k32)[0];
        if (fabs(s) > 1e-12) k32 /= static_cast<float>(s);
    }

    // Flip kernel (true convolution)
    Mat k;
    flip(k32, k, -1);

    const int kr = k.rows / 2;
    const int kc = k.cols / 2;

    // Pad source
    Mat padded;
    copyMakeBorder(srcFloat, padded, kr, kr, kc, kc, borderType);

    Mat dst(src.size(), srcFloat.type());

    const int C = src.channels();
    for (int y = 0; y < src.rows; ++y) {
        float* drow = dst.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            vector<float> acc(C, 0.f);
            for (int ky = 0; ky < k.rows; ++ky) {
                const float* prow = padded.ptr<float>(y + ky);
                const float* krow = k.ptr<float>(ky);
                for (int kx = 0; kx < k.cols; ++kx) {
                    float w = krow[kx];
                    const float* spx = prow + (x + kx) * C;
                    for (int c = 0; c < C; ++c) {
                        acc[c] += w * spx[c];
                    }
                }
            }
            for (int c = 0; c < C; ++c) {
                drow[x * C + c] = acc[c];
            }
        }
    }

    Mat out;
    dst.convertTo(out, src.type());
    return out;
}

// Some example kernels
static Mat makeBoxKernel(int k) {
    if (k % 2 == 0 || k < 1) throw invalid_argument("Kernel size must be odd and >=1");
    return Mat::ones(k, k, CV_32F);
}

static Mat makeSharpenKernel() {
    float data[9] = { 0, -1,  0,
                     -1,  5, -1,
                      0, -1,  0 };
    return Mat(3, 3, CV_32F, data).clone();
}

int main(int argc, char** argv)
{
    string imagePath = "Exercise3/Images/ariane5_1b.jpg";
    Mat img = imread(imagePath, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Failed to read image: " << imagePath << endl;
        return 1;
    }

    cout << "Loaded: " << imagePath << " (" << img.cols << "x" << img.rows << ")" << endl;

    // Example 1: Box blur 3x3
    Mat kernelBox = makeBoxKernel(3);
    Mat blurred = convolutionFiltering2D(img, kernelBox, true);
    imwrite("Exercise3/filtered_box3.png", blurred);

    // Example 2: Sharpen
    Mat kernelSharp = makeSharpenKernel();
    Mat sharpened = convolutionFiltering2D(img, kernelSharp, false);
    imwrite("Exercise3/filtered_sharpen.png", sharpened);

    // Show results
    imshow("Original", img);
    imshow("Box Blur 3x3", blurred);
    imshow("Sharpen", sharpened);
    waitKey(0);

    return 0;
}