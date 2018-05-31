#import <opencv2/opencv.hpp>
#include "YoloTinyV1Classifier.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <fstream>
#include <AVFoundation/AVFoundation.h>
#include <CoreText/CoreText.h>
#include "YOLOKit.h"
#import <YOLOKit/YOLOKit-Swift.h>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <list>

//#import <opencv2/opencv.hpp>
using tensorflow::uint8;
namespace tf = tensorflow;
UIImageView* _imgView;
CGSize imgSize;
tensorflow::SessionOptions options;
tensorflow::Session* session_pointer = nullptr;
tensorflow::GraphDef tensorflow_graph;
//const std::string inputTensorName = "image_input:0";
//const std::string outputTensorName = "prediction/BiasAdd:0";
const std::string inputTensorName = "input";
const std::string outputTensorName = "output";
const NSUInteger yoloImageDimension = 416;
const NSUInteger numClasses = 6;
//const NSUInteger gridSize = 7;
const NSUInteger gridSize = 13;
const NSUInteger numBoxesPerCell = 5;
CGImageRef cgImageRef;
//const NSArray<NSString *> *labels = @[@"logo", @"foot", @"knee", @"face", @"hand", @"elbow"];
const char* labels[] = {
    "logo", "foot", "knee", "face", "hand", "elbow"
};

//http://www.mattrajca.com/2016/11/25/getting-started-with-deep-mnist-and-tensorflow-on-ios.html
@implementation YoloTinyV1Classifier
namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace
#pragma MARK - initialization

- (id) init {
    self = [super init];
    
    if(self) {
        boxes = [[NSMutableArray alloc] initWithCapacity:gridSize * gridSize * numBoxesPerCell];
    }
    
    return self;
}
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float vals[], int count) {
    float max = -FLT_MAX;
    for (int i=0; i<count; i++) {
        max = fmax(max, vals[i]);
    }
    float sum = 0.0;
    for (int i=0; i<count; i++) {
        vals[i] = exp(vals[i] - max);
        sum += vals[i];
    }
    for (int i=0; i<count; i++) {
        vals[i] /= sum;
    }
}
#pragma MARK - implementation
static void YoloPostProcess(const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                            Eigen::Aligned>& output, std::vector<std::pair<float, int> >* top_results) {
    const int NUM_CLASSES = 6;
    const int NUM_BOXES_PER_BLOCK = 5;
    double ANCHORS[] = {
        // for tiny-yolo-voc yolov2
        1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52
    };
    
    // 13 for tiny-yolo-voc, 19 for yolo
    const int gridHeight = 13;
    const int gridWidth = 13;
    const int blockSize = 32;
    
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_result_pq;
    
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_rect_pq;
    
    NSMutableDictionary *idxRect = [NSMutableDictionary dictionary];
    NSMutableDictionary *idxDetectedClass = [NSMutableDictionary dictionary];
    int i=0;
    for (int y = 0; y < gridHeight; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
            for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                int offset = (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                + (NUM_CLASSES + 5) * b;
                
                // implementation based on the TF Android TFYoloDetector.java
                // also in http://machinethink.net/blog/object-detection-with-yolo/
                float xPos = (x + sigmoid(output(offset + 0))) * blockSize;
                float yPos = (y + sigmoid(output(offset + 1))) * blockSize;
                
                float w = (float) (exp(output(offset + 2)) * ANCHORS[2 * b + 0]) * blockSize;
                float h = (float) (exp(output(offset + 3)) * ANCHORS[2 * b + 1]) * blockSize;
                
                // Now xPos and yPos represent the center of the bounding box in the 416×416 image that we used as input to the neural network; w and h are the width and height of the box in that same image space.
                CGRect rect = CGRectMake(
                                         fmax(0, (xPos - w / 2) * imgSize.width / yoloImageDimension),
                                         fmax(0, (yPos - h / 2) * imgSize.height / yoloImageDimension),
                                         w* imgSize.width / yoloImageDimension, h* imgSize.height / yoloImageDimension);
                
                float confidence = sigmoid(output(offset + 4));
                
                float classes[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    classes[c] = output(offset + 5 + c);
                }
                softmax(classes, NUM_CLASSES);
                
                int detectedClass = -1;
                float maxClass = 0;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    if (classes[c] > maxClass) {
                        detectedClass = c;
                        maxClass = classes[c];
                    }
                }
                
                float confidenceInClass = maxClass * confidence;
                if (confidenceInClass > 0.25) {
                    NSLog(@"%s (%d) %f %d, %d, %d, %@", labels[detectedClass], detectedClass, confidenceInClass, y, x, b, NSStringFromCGRect(rect));
                    top_result_pq.push(std::pair<float, int>(confidenceInClass, detectedClass));
                    top_rect_pq.push(std::pair<float, int>(confidenceInClass, i));
                    [idxRect setObject:NSStringFromCGRect(rect) forKey:[NSNumber numberWithInt:i]];
                    [idxDetectedClass setObject:[NSNumber numberWithInt:detectedClass] forKey:[NSNumber numberWithInt:i++]];
                }
            }
        }
    }
    
    
    std::vector<std::pair<float, int> > top_rects;
    while (!top_rect_pq.empty()) {
        top_rects.push_back(top_rect_pq.top());
        top_rect_pq.pop();
    }
    std::reverse(top_rects.begin(), top_rects.end());
    
    
    // Start with the box that has the highest score.
    // Remove any remaining boxes - with the same class? - that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    std::vector<std::pair<float, int> > nms_rects;
    while (!top_rects.empty()) {
        auto& first = top_rects.front();
        CGRect rect_first = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:first.second]]);
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:first.second]] intValue];
        NSLog(@"first class: %s", labels[detectedClass]);
        
        for (unsigned long i = top_rects.size()-1; i>=1; i--) {
            auto& item = top_rects.at(i);
            int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:item.second]] intValue];
            
            CGRect rect_item = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:item.second]]);
            CGRect rectIntersection = CGRectIntersection(rect_first, rect_item);
            if (CGRectIsNull(rectIntersection)) {
                //NSLog(@"no intesection");
                NSLog(@"no intesection - class: %s", labels[detectedClass]);
            }
            else {
                float areai = rect_first.size.width * rect_first.size.height;
                float ratio = rectIntersection.size.width * rectIntersection.size.height / areai;
                NSLog(@"found intesection - class: %s", labels[detectedClass]);
                
                if (ratio > 0.23) {
                    top_rects.erase(top_rects.begin() + i);
                }
            }
        }
        nms_rects.push_back(first);
        top_rects.erase(top_rects.begin());
    }
    
    //    AppDelegate *appDelegate = (AppDelegate *)[[UIApplication sharedApplication] delegate];
    //  [self drawImageWithRects2:nms_rects idxRect:idxRect idxDetectedClass:idxDetectedClass];
    
    //[_imgView setImage:[UIImage imageNamed:[input_image stringByAppendingString:@".jpg"]]];
    _imgView.contentMode = UIViewContentModeScaleAspectFit; //UIViewContentModeTopLeft; //UIViewContentModeScaleAspectFit;
    cv::Mat src=cvMatFromUIImage(_imgView.image);
    cv::Mat dst;
    dst=src;
    NSLog(@"nms_rects size=%lu", nms_rects.size());
    while (!nms_rects.empty()) {
        auto& front = nms_rects.front();
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
        //        NSLog(@"%f: %s %d %@", front.first, LABELS[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
        
        //        cv::Point pt1(bboxes[i][0], bboxes[i][1]);
        //        cv::Point pt2(bboxes[i][2], bboxes[i][3]);
        //        cv::rectangle(dst, pt1, pt2, cv::Scalar(255, 0, 0), 4);
        //        // also works - use rect instead of two points
        CGRect rect = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
        
        cv::Rect cvrect(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height);
        cv::rectangle(dst, cvrect, detectedClass==0?cv::Scalar(255, 0, 0):cv::Scalar(0, 255,0), 4);
        
        
        nms_rects.erase(nms_rects.begin());
    }
    
    [_imgView setImage: UIImageFromCVMat(dst)];
    
    /////////
    while (!nms_rects.empty()) {
        auto& front = nms_rects.front();
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
        top_results->push_back(std::pair<float, int>(front.first, detectedClass));
        
        NSLog(@"%f: %s %d %@", front.first, labels[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
        nms_rects.erase(nms_rects.begin());
    }
    
}
cv::Mat cvMatFromUIImage(UIImage * image)
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
//- (void)drawImageWithRects:(std::vector<std::pair<float, int> >)nms_rects idxRect:(NSDictionary*)idxRect idxDetectedClass:(NSDictionary*)idxDetectedClass {
//    [_imgView setImage:[UIImage imageNamed:[input_image stringByAppendingString:@".jpg"]]];
//    _imgView.contentMode = UIViewContentModeScaleAspectFit; //UIViewContentModeTopLeft; //UIViewContentModeScaleAspectFit;
//    cv::Mat src=[self cvMatFromUIImage:_imgView.image];
//    cv::Mat dst;
//    dst=src;
//    NSLog(@"nms_rects size=%lu", nms_rects.size());
//    while (!nms_rects.empty()) {
//        auto& front = nms_rects.front();
//        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
//        //        NSLog(@"%f: %s %d %@", front.first, LABELS[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
//
//        //        cv::Point pt1(bboxes[i][0], bboxes[i][1]);
//        //        cv::Point pt2(bboxes[i][2], bboxes[i][3]);
//        //        cv::rectangle(dst, pt1, pt2, cv::Scalar(255, 0, 0), 4);
//        //        // also works - use rect instead of two points
//        CGRect rect = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
//
//        cv::Rect cvrect(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height);
//        cv::rectangle(dst, cvrect, detectedClass==0?cv::Scalar(255, 0, 0):cv::Scalar(0, 255,0), 4);
//
//
//        nms_rects.erase(nms_rects.begin());
//    }
//
//    [_imgView setImage:[self UIImageFromCVMat:dst]];
//}
- (void)loadModelV2 {
    //tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!tensorflow::NewSession(options, &self->session).ok()) {
        return;
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    LOG(INFO) << "Graph created.";
    NSString* network_path = FilePathForResourceName(@"tf-tiny-yolo-voc-6c", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    LOG(INFO) << "Creating session.";
    if (!self->session->Create(tensorflow_graph).ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: ";
    }
}
NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}
bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}
- (void)loadModel {
    
    tf::Status status = tf::NewSession(tf::SessionOptions(), &self->session);
    if (!status.ok()) {
        LOG(INFO) << "create session failed.";
        return;
    }
    LOG(INFO) << "Session created.";
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    NSString* network_path = FilePathForResourceName(@"tf-tiny-yolo-voc-6c", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = self->session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return;
    }else{
        LOG(INFO) << "TensorFlow Graph created";
    }
}

std::vector<uint8> LoadImageFromFile2(const char* file_name,
                                      int* out_width, int* out_height,
                                      int* out_channels) {
    FILE* file_handle = fopen(file_name, "rb");
    fseek(file_handle, 0, SEEK_END);
    const size_t bytes_in_file = ftell(file_handle);
    fseek(file_handle, 0, SEEK_SET);
    std::vector<uint8> file_data(bytes_in_file);
    fread(file_data.data(), 1, bytes_in_file, file_handle);
    fclose(file_handle);
    CFDataRef file_data_ref = CFDataCreateWithBytesNoCopy(NULL, file_data.data(),
                                                          bytes_in_file,
                                                          kCFAllocatorNull);
    CGDataProviderRef image_provider =
    CGDataProviderCreateWithCFData(file_data_ref);
    
    const char* suffix = strrchr(file_name, '.');
    if (!suffix || suffix == file_name) {
        suffix = "";
    }
    CGImageRef image;
    if (strcasecmp(suffix, ".png") == 0) {
        image = CGImageCreateWithPNGDataProvider(image_provider, NULL, true,
                                                 kCGRenderingIntentDefault);
    } else if ((strcasecmp(suffix, ".jpg") == 0) ||
               (strcasecmp(suffix, ".jpeg") == 0)) {
        image = CGImageCreateWithJPEGDataProvider(image_provider, NULL, true,
                                                  kCGRenderingIntentDefault);
    } else {
        CFRelease(image_provider);
        CFRelease(file_data_ref);
        fprintf(stderr, "Unknown suffix for file '%s'\n", file_name);
        *out_width = 0;
        *out_height = 0;
        *out_channels = 0;
        return std::vector<uint8>();
    }
    
    const int width = (int)CGImageGetWidth(image);
    const int height = (int)CGImageGetHeight(image);
    const int channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (width * channels);
    const int bytes_in_image = (bytes_per_row * height);
    std::vector<uint8> result(bytes_in_image);
    const int bits_per_component = 8;
    CGContextRef context = CGBitmapContextCreate(result.data(), width, height,
                                                 bits_per_component, bytes_per_row, color_space,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CFRelease(image);
    CFRelease(image_provider);
    CFRelease(file_data_ref);
    
    *out_width = width;
    *out_height = height;
    *out_channels = channels;
    return result;
}

std::vector<uint8> LoadImageFromFile(CGImageRef image,
                                     int* out_width, int* out_height,
                                     int* out_channels) {
    auto image_provider = CGImageGetDataProvider(image);
    auto file_data_ref = CGDataProviderCopyData(image_provider);
    
    const int width = (int)CGImageGetWidth(image);
    const int height = (int)CGImageGetHeight(image);
    const int channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (width * channels);
    const int bytes_in_image = (bytes_per_row * height);
    std::vector<uint8> result(bytes_in_image);
    const int bits_per_component = 8;
    CGContextRef context = CGBitmapContextCreate(result.data(), width, height,
                                                 bits_per_component, bytes_per_row, color_space,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CFRelease(file_data_ref);
    
    *out_width = width;
    *out_height = height;
    *out_channels = channels;
    return result;
}
- (void)classifyImage:(UIImage*) image {
    [boxes removeAllObjects];
    // cgImageRef = image;
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(image.CGImage, &image_width, &image_height, &image_channels);
    const int wanted_width = yoloImageDimension; //416;
    const int wanted_height = yoloImageDimension; //416;
    const int wanted_channels = 3;
    
    // YOLO’s convolutional layers downsample the image by a factor of 32 so by using an input image of 416 we get an output feature map of 13x13.
    assert(image_channels >= wanted_channels);
    
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, wanted_width, wanted_height, wanted_channels}));
    
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    
    tensorflow::uint8* in = image_data.data();
    tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                //out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                out_pixel[c] = in_pixel[c] / 255.0f; // in Android's TensorFlowYoloDetector.java, no std and mean is used for input values - "We also need to scale the pixel values from integers that are between 0 and 255 to the floating point values that the graph operates on. We control the scaling with the input_mean and input_std flags: we first subtract input_mean from each pixel value, then divide it by input_std." https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
            }
        }
    }
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{inputTensorName, image_tensor}},
                                                 {outputTensorName}, {}, &outputs);
    
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
    }
    tensorflow::Tensor* output = &outputs[0];
    std::vector<std::pair<float, int> > top_results;
    
    imgSize = image.size;
    // YoloPostProcess(output->flat<float>(), &top_results);
    
    const int NUM_CLASSES = 6;
    const int NUM_BOXES_PER_BLOCK = 5;
    double ANCHORS[] = {
        // for tiny-yolo-voc.pb: 20 classes
        1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52
        // for tiny-yolo(coco).pb: 80 classes
        //0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741
    };
    // 13 for tiny-yolo-voc, 19 for yolo
    const int gridHeight = 13;///19;
    const int gridWidth = 13;//19;
    const int blockSize = 32;
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_result_pq;
    
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_rect_pq;
    
    NSMutableDictionary *idxRect = [NSMutableDictionary dictionary];
    NSMutableDictionary *idxDetectedClass = [NSMutableDictionary dictionary];
    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 13x13x125 elements (actually x128 instead of x125 because in
    // Metal the number of channels must be a multiple of 4).
    int i=0;
    for (int y = 0; y < gridHeight; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
            for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                int offset = (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                + (NUM_CLASSES + 5) * b;
                
                // implementation based on the TF Android TFYoloDetector.java
                // also in http://machinethink.net/blog/object-detection-with-yolo/
                //results co 125 chanel trong do 5 chanel dunng de luu giu thong tin x,y,w,h,confident
                
                float xPos = (x + sigmoid(output->flat<float>()(offset + 0))) * blockSize;
                float yPos = (y + sigmoid(output->flat<float>()(offset + 1))) * blockSize;
                
                float w = (float) (exp(output->flat<float>()(offset + 2)) * ANCHORS[2 * b + 0]) * blockSize;
                float h = (float) (exp(output->flat<float>()(offset + 3)) * ANCHORS[2 * b + 1]) * blockSize;
                //Softmax and Sigmoid Helping to predict the target class
                // the logistic sigmoid to turn this into a percentage.
                float confidence = sigmoid(output->flat<float>()(offset + 4));
                
                // Now xPos and yPos represent the center of the bounding box in the 416×416 image that we used as input to the neural network; w and h are the width and height of the box in that same image space.
                CGRect rect = CGRectMake(
                                         fmax(0, (xPos - w / 2) * imgSize.width / yoloImageDimension),
                                         fmax(0, (yPos - h / 2) * imgSize.height / yoloImageDimension),
                                         w* imgSize.width / yoloImageDimension, h* imgSize.height / yoloImageDimension);
                
                
                
                // Gather the predicted classes for this anchor box and softmax them,
                // so we can interpret these numbers as percentages.
                float classes[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    classes[c] = output->flat<float>()(offset + 5 + c);
                }
                softmax(classes, NUM_CLASSES);
                
                // Find the index of the class with the largest score.(detectedClass)
                int detectedClass = -1;
                float maxClass = 0;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    if (classes[c] > maxClass) {
                        detectedClass = c;
                        maxClass = classes[c];
                    }
                }
                
                float confidenceInClass = maxClass * confidence;
                if (confidenceInClass > 0.2) {
                    NSLog(@"%s (%d) %f %d, %d, %d, %@", labels[detectedClass], detectedClass, confidenceInClass, y, x, b, NSStringFromCGRect(rect));
                    top_result_pq.push(std::pair<float, int>(confidenceInClass, detectedClass));
                    top_rect_pq.push(std::pair<float, int>(confidenceInClass, i));
                    [idxRect setObject:NSStringFromCGRect(rect) forKey:[NSNumber numberWithInt:i]];
                    [idxDetectedClass setObject:[NSNumber numberWithInt:detectedClass] forKey:[NSNumber numberWithInt:i++]];
                    //                    [boxes addObject:
                    //                     [[YoloV1Box alloc]
                    //                      initWithX:rect.origin.x y:rect.origin.y
                    //                      width:rect.size.width height:rect.size.height
                    //                      confidence:confidenceInClass classIndex:detectedClass
                    //                      label:[NSString stringWithUTF8String:labels[detectedClass]]]];
                }
                
            }
        }
    }
    std::vector<std::pair<float, int> > top_rects;
    while (!top_rect_pq.empty()) {
        top_rects.push_back(top_rect_pq.top());
        top_rect_pq.pop();
    }
    std::reverse(top_rects.begin(), top_rects.end());
    
    
    // Start with the box that has the highest score.
    // Remove any remaining boxes - with the same class? - that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    std::vector<std::pair<float, int> > nms_rects;
    while (!top_rects.empty()) {
        auto& first = top_rects.front();
        CGRect rect_first = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:first.second]]);
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:first.second]] intValue];
        NSLog(@"first class: %s", labels[detectedClass]);
        
        for (unsigned long i = top_rects.size()-1; i>=1; i--) {
            auto& item = top_rects.at(i);
            int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:item.second]] intValue];
            
            CGRect rect_item = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:item.second]]);
            CGRect rectIntersection = CGRectIntersection(rect_first, rect_item);
            if (CGRectIsNull(rectIntersection)) {
                //NSLog(@"no intesection");
                NSLog(@"no intesection - class: %s", labels[detectedClass]);
            }
            else {
                float areai = rect_first.size.width * rect_first.size.height;
                float ratio = rectIntersection.size.width * rectIntersection.size.height / areai;
                NSLog(@"found intesection - class: %s", labels[detectedClass]);
                
                if (ratio > 0.23) {
                    top_rects.erase(top_rects.begin() + i);
                }
            }
        }
        nms_rects.push_back(first);
        top_rects.erase(top_rects.begin());
    }
    while (!nms_rects.empty()) {
        auto& front = nms_rects.front();
        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
        CGRect rect = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
        
        [boxes addObject:
         [[YoloV1Box alloc]
          initWithX:rect.origin.x y:rect.origin.y
          width:rect.size.width height:rect.size.height
          confidence:0.11 classIndex:detectedClass
          label:[NSString stringWithUTF8String:labels[detectedClass]]]];
        nms_rects.erase(nms_rects.begin());
    }
}
- (void)drawImageWithRects:(UIImageView*) imgView {
    _imgView = imgView;
    imgSize = _imgView.image.size;
    [self RunInferenceOnImage];
}
UIImage * UIImageFromCVMat(cv::Mat cvMat)
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}
- (NSArray<YoloV1Box *> *) result {
    return boxes;
}

- (void)close {
    if(self->session) {
        tf::Status status = self->session->Close();
        
        if (!status.ok()) {
            LOG(ERROR) << "Error while closing session: " << status.ToString() << "\n";
            return;
        }
        
        self->session = nil;
    }
}
- (void)drawImageWithRects2:(std::vector<std::pair<float, int> >)nms_rects idxRect:(NSDictionary*)idxRect idxDetectedClass:(NSDictionary*)idxDetectedClass{
    
    //    //nms_rects: numbers of rects
    //    [_imgView setImage:[UIImage imageNamed:[input_image stringByAppendingString:@".jpg"]]];
    //    _imgView.contentMode = UIViewContentModeScaleAspectFit; //UIViewContentModeTopLeft; //UIViewContentModeScaleAspectFit;
    //    cv::Mat src=[self cvMatFromUIImage:_imgView.image];
    //    cv::Mat dst;
    //    dst=src;
    //    NSLog(@"nms_rects size=%lu", nms_rects.size());
    //    while (!nms_rects.empty()) {
    //        auto& front = nms_rects.front();
    //        int detectedClass = [[idxDetectedClass objectForKey:[NSNumber numberWithInt:front.second]] intValue];
    //        //        NSLog(@"%f: %s %d %@", front.first, LABELS[detectedClass], detectedClass, [idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
    //
    //        //        cv::Point pt1(bboxes[i][0], bboxes[i][1]);
    //        //        cv::Point pt2(bboxes[i][2], bboxes[i][3]);
    //        //        cv::rectangle(dst, pt1, pt2, cv::Scalar(255, 0, 0), 4);
    //        //        // also works - use rect instead of two points
    //        CGRect rect = CGRectFromString([idxRect objectForKey:[NSNumber numberWithInt:front.second]]);
    //
    //        cv::Rect cvrect(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height);
    //        cv::rectangle(dst, cvrect, detectedClass==0?cv::Scalar(255, 0, 0):cv::Scalar(0, 255,0), 4);
    //
    //
    //        nms_rects.erase(nms_rects.begin());
    //    }
    //
    //    [_imgView setImage:[self UIImageFromCVMat:dst]];
}
- (void) RunInferenceOnImage {
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(_imgView.image.CGImage, &image_width, &image_height, &image_channels);;
    
    
    const int wanted_width = yoloImageDimension; //416;
    const int wanted_height = yoloImageDimension; //416;
    const int wanted_channels = 3;
    
    
    
    // YOLO’s convolutional layers downsample the image by a factor of 32 so by using an input image of 416 we get an output feature map of 13x13.
    
    assert(image_channels >= wanted_channels);
    
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    
    tensorflow::uint8* in = image_data.data();
    tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                //out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                out_pixel[c] = in_pixel[c] / 255.0f; // in Android's TensorFlowYoloDetector.java, no std and mean is used for input values - "We also need to scale the pixel values from integers that are between 0 and 255 to the floating point values that the graph operates on. We control the scaling with the input_mean and input_std flags: we first subtract input_mean from each pixel value, then divide it by input_std." https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
            }
        }
    }
    
    
    
    std::string input_layer = "input";
    std::string output_layer = "output";
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                                                 {output_layer}, {}, &outputs);
    
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
    }
    tensorflow::Tensor* output = &outputs[0];
    std::vector<std::pair<float, int> > top_results;
    YoloPostProcess(output->flat<float>(), &top_results);
}

@end


