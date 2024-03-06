
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cmath>
#include <jni.h>
#include <iostream>
#include <string>
#include <vector>

// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net yolov9;

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
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

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

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

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
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
                obj.x = x0;
                obj.y = y0;
                obj.w = x1 - x0;
                obj.h = y1 - y0;
                obj.label = class_index;
                obj.prob = confidence;
                objects.push_back(obj);
            }
        }
    }
}

static ncnn::Mat transpose(const ncnn::Mat& in) {
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

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV9Ncnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov9ncnn_YoloV9Ncnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolov9.opt = opt;

    // init param
    {
        int ret = yolov9.load_param(mgr, "yolov9-c-converted-sim.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV9Ncnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = yolov9.load_model(mgr, "yolov9-c-converted-sim.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV9Ncnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yolov9ncnn/YoloV9Ncnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolov9ncnn/YoloV9Ncnn;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}

// public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov9ncnn_YoloV9Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    const int target_size = 640;

    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
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

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    int dw = target_size - w;
    int dh = target_size - h;
    dw = dw / 2;
    dh = dh / 2;

    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));


    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

    // yolov5
    std::vector<Object> objects;
    {
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov9.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        ex.input("images", in_pad);

        {
            ncnn::Mat out;
            ex.extract("output0", out);

            ncnn::Mat out_t = transpose(out);
            generate_proposals(out_t, prob_threshold, objects);

        }



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
            float x0 = (newobjects[i].x - dw) / scale;
            float y0 = (newobjects[i].y - dh) / scale;
            float x1 = (newobjects[i].x + newobjects[i].w - dw) / scale;
            float y1 = (newobjects[i].y + newobjects[i].h - dh) / scale;

            //clip
            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

            newobjects[i].x = x0;
            newobjects[i].y = y0;
            newobjects[i].w = x1 - x0;
            newobjects[i].h = y1 - y0;
        }

        objects = newobjects;
    }

    // objects to Obj[]
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

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].w);
        env->SetFloatField(jObj, hId, objects[i].h);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV9Ncnn", "%.2fms   detect", elasped);

    return jObjArray;
}

}
