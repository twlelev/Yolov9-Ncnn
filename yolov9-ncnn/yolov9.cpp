#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


static void generate_proposals(const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{   

    float* data = (float*)feat_blob.data;
    for(int i=0;i<feat_blob.h;i++)
    {   
        int class_index = -1;
        float confidence = 0;

        for(int j=4;j<feat_blob.w;j++)
        {
            if(data[i*feat_blob.w+j]>=prob_threshold)
            {   
                class_index = j-4;
                confidence=data[i*feat_blob.w+j];
                float x0 = data[i*feat_blob.w] - data[i*feat_blob.w+2] * 0.5f;
                float y0 = data[i*feat_blob.w+1] - data[i*feat_blob.w+3] * 0.5f;
                float x1 = data[i*feat_blob.w] + data[i*feat_blob.w+2] * 0.5f;
                float y1 = data[i*feat_blob.w+1] + data[i*feat_blob.w+3] * 0.5f;
            
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = confidence;
                objects.push_back(obj);
            }
        }
    }
}


static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}



ncnn::Mat transpose(const ncnn::Mat& in) {
    ncnn::Mat out(in.h, in.w, in.c);
    for (int q=0; q<in.c; q++) {
        const float* ptr = in.channel(q);
        float* outptr = out.channel(q);
        for (int y=0; y<in.h; y++) {
            for (int x=0; x<in.w; x++) {
                outptr[x * in.h + y] = ptr[y * in.w + x];
            }
        }
    }

    return out;
}


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("image.jpg", image);
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread failed\n");
        return -1;
    }

    ncnn::Net yolov9;

    yolov9.opt.use_vulkan_compute = true;

    if (yolov9.load_param("../yolov9-c-converted-sim.param"))
        exit(-1);
    if (yolov9.load_model("../yolov9-c-converted-sim.bin"))
        exit(-1);

    int width = bgr.cols;   
    int height = bgr.rows;  

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale= 0.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);

    int dw = target_size - w;
    int dh = target_size - h;
    dw = dw / 2;
    dh = dh / 2;

    // pad to target_size rectangle left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));
    
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    //in = divide_by_255(in) ;

    ncnn::Extractor ex = yolov9.create_extractor();

    // output0
    ex.input("images", in_pad);
    ncnn::Mat out;
    ex.extract("output0", out);

    //CHW->CWH  (1,84,8400)->(1,8400,84)
    ncnn::Mat out_t = transpose(out);

    std::vector<Object> objects;
    generate_proposals(out_t, prob_threshold, objects);


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(objects);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, nms_threshold);

    int count = picked.size();
    std::vector<Object> newobjects;
    newobjects.resize(count);
    for (int i = 0; i < count; i++)
    {
        newobjects[i] = objects[picked[i]];
        // adjust offset to original unpadded
        float x0 = (newobjects[i].rect.x - dw) / scale;
        float y0 = (newobjects[i].rect.y - dh) / scale;
        float x1 = (newobjects[i].rect.x + newobjects[i].rect.width - dw) / scale;
        float y1 = (newobjects[i].rect.y + newobjects[i].rect.height - dh) / scale;

        //clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        newobjects[i].rect.x = x0;
        newobjects[i].rect.y = y0;
        newobjects[i].rect.width = x1 - x0;
        newobjects[i].rect.height = y1 - y0;
    }


    draw_objects(bgr,newobjects);

    // return 0;
}