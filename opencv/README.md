# Learning C++ with the OpenCV Library

## Building OpenCV library and Installing 

<details><summary><b>Clone the OpenCV library</b></summary>

Clone the OpenCV library from their official GitHub link:

```console 
user@DESKTOP:~$ cd Downloads
user@DESKTOP:Downloads$ git clone git@github.com:opencv/opencv.git
```

</details>

<details><summary><b>Build the OpenCV library</b></summary>

Clone the OpenCV library from their official GitHub link:

```console 
user@DESKTOP:Downloads$ cd opencv
user@DESKTOP:opencv$ mkdir build
user@DESKTOP:opencv$ cd build
```

Generate the Makefile for compiling the C++ project:
    
```console 
user@DESKTOP:build$ cmake ..
```

The command above utilizes cmake to build the Makefile for our project. Cmake is a "build system" and is supported by almost all modern IDEs and compilers. In order to really start digging into C++, its important to familiarize ourselves with cmake. Here are a few interesting pages on cmake: [modern cmake](https://cliutils.gitlab.io/modern-cmake/), [build with cmake (MIT)](https://vnav.mit.edu/labs/lab1/cmake.html).

Build project (generate binaries) using make:
```console 
user@DESKTOP:opencv$ make . -j4
```

Above, the Makefile was used to generate binaries (executables) for the project. The Makefile is really the important part of building a C++ project. It can be seen as a preprocessor for the projects source code, which then compiles C (or C++) source files, and then finally links the code. For more information on Makefiles, [checkout this page](https://www.gnu.org/software/make/manual/html_node/Introduction.html).
</details>

<details><summary><b>Install the OpenCV library binaries</b></summary>

Install newly generated binaries on system:

```console 
user@DESKTOP:opencv$ make install
```

</details>

## Testing the OpenCV library

All source files mentioned below are included in this repository, so recreating them is not necessary. Building the Makefile and compiling it will still need to be done. 

<details><summary><b>Using OpenCV to view webcam feed</b></summary>

This will be our first time testing the installed OpenCV library using C++. We can begin by making an example script to open a webcam on our computers.

1. <details><summary><b>Create the source file for viewing a webcam stream</b></summary>

    Make the source file:
    ```console 
    user@DESKTOP:~$ vim DisplayWebcam.cpp
    ```

    Add the following code:
    ```cpp
    #include<opencv2/opencv.hpp> //OpenCV header to use VideoCapture class//
    #include<iostream>
    using namespace std;
    using namespace cv;

    int main() {
    Mat myImage; //Declaring a matrix to load the frames//
    namedWindow("Video Player"); //Declaring the video to show the video//
    VideoCapture cap(0); //Declaring an object to capture stream of frames from default camera//
    
    if (!cap.isOpened()){ //This section prompt an error message if no video stream is found//
        cout << "No video stream detected" << endl;
        system("pause");
        return-1;
    }

    while (true){ //Taking an everlasting loop to show the video//
        cap >> myImage;

        if (myImage.empty()){ //Breaking the loop if no video frame is detected//
            break;
        }

        imshow("Video Player", myImage);//Showing the video//
        char c = (char)waitKey(25);//Allowing 25 milliseconds frame processing time and initiating break condition//
        
        if (c == 27){ //If 'Esc' is entered break the loop//
            break;
        }
    }

    cap.release(); //Releasing the buffer memory//
    return 0;
    }
    ```
2. <details><summary><b>Create a CMakeLists.txt File</b></summary>

    ```console 
    user@DESKTOP:~$ vim CMakeLists.txt
    ```

    ```cmake
    cmake_minimum_required(VERSION 2.8)

    project( OpenCVTesting )
    find_package( OpenCV REQUIRED )

    include_directories( ${OpenCV_INCLUDE_DIRS} )

    add_executable( DisplayWebcam DisplayWebcam.cpp )

    target_link_libraries( DisplayWebcam ${OpenCV_LIBS} )
    ```

3. <details><summary><b>Compile the workspace</b></summary>

    ```console 
    user@DESKTOP:~$ cmake .
    user@DESKTOP:~$ make
    ```

4. <details><summary><b>Run the webcam viewer binary</b></summary>
    
    ```console 
    user@DESKTOP:~$ ./DisplayWebcam
    ```
</details>

<details><summary><b>Using Traditional ML for Face Detection</b></summary>

1. <details><summary><b>Download HaarCascade Classifier XML files</b></summary>

    The XML files containing trained Haar Cascade Classifiers are included in this repository, [though they can also be found here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

2. <details><summary><b>Create the source file for object detections</b></summary>

    Make the source file:
    ```console 
    user@DESKTOP:~$ vim FaceDetect.cpp
    ```

    Add the following code:
    ```cpp
    #include <iostream>
    #include <opencv2/opencv.hpp>
    /*
    *  Could remove above and include specific header files
    *
    *   #include <opencv2/objdetect.hpp>
    *   #include <opencv2/highgui.hpp>
    *   #include <opencv2/imgproc.hpp>    
    */
    
    using namespace std;
    using namespace cv;
    
    // Function for Face Detection
    void detectAndDraw( Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye, CascadeClassifier& smile_face, double scale );
    string cascadeName, nestedCascadeName;
    
    int main( int argc, const char** argv )
    {
        // VideoCapture class for playing video for which faces to be detected
        VideoCapture capture;
        Mat frame, image;
    
        // PreDefined trained XML classifiers with facial features
        CascadeClassifier cascade, eye, smile_face;
        double scale=1;
    
        // Load classifiers from "haarcascades" directory
        eye.load("haarcascades/haarcascade_lefteye_2splits.xml");
        smile_face.load("haarcascades/haarcascade_smile.xml");

        // Change path before execution
        cascade.load("haarcascades/haarcascade_frontalface_default.xml");
    
        // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
        capture.open(0);
        if(capture.isOpened())
        {
            // Capture frames from video and detect faces
            cout << "Face Detection Started...." << endl;
            while(1)
            {
                capture >> frame;
                if( frame.empty() )
                    break;
                Mat frame1 = frame.clone();
                detectAndDraw(frame1, cascade, eye, smile_face, scale);
                char c = (char)waitKey(10);
            
                // Press q to exit from window
                if( c == 27 || c == 'q' || c == 'Q' )
                    break;
            }
        }
        else
            cout<<"Could not Open Camera";
        return 0;
    }
    
    void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye, CascadeClassifier& smile_face, double scale)
    {
        vector<Rect> faces, faces2;
        Mat gray, smallImg;
    
        cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
        double fx = 1 / scale;
    
        // Resize the Grayscale Image
        resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);
    
        // Detect faces of different sizes using cascade classifier
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

        // Draw circles around the faces
        for (size_t i = 0; i < faces.size(); i++)
        {
            Rect r = faces[i];
            Mat smallImgROI;
            vector<Rect> nestedObjects;
            Point center;
            Scalar color = Scalar(100, 200, 200); // Color for Drawing tool
            Scalar color2 = Scalar(100, 10, 200);  // Color 2 for Drawing tool
            Scalar color3 = Scalar(250, 10, 250);  // Color 2 for Drawing tool
            int radius;
    
            double aspect_ratio = (double)r.width/r.height;

            if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
            {
                center.x = cvRound((r.x + r.width*0.5)*scale);
                center.y = cvRound((r.y + r.height*0.5)*scale);
                radius = cvRound((r.width + r.height)*0.25*scale);
                circle( img, center, radius, color, 3, 8, 0 );
            }
            else
                rectangle(img, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)), cv::Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
            if(eye.empty() && smile_face.empty())
                continue;

            smallImgROI = smallImg(r);
            
            // Detection of eyes in the input image
            eye.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
            // Draw circles around eyes
            for (size_t j = 0; j < nestedObjects.size(); j++)
            {
                Rect nr = nestedObjects[j];
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
                radius = cvRound((nr.width + nr.height)*0.25*scale);
                circle(img, center, radius, color2, 3, 8, 0);
            }

            // Detection of eyes in the input image
            smile_face.detectMultiScale(smallImg, nestedObjects, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
            
            // Draw circles around mouth
            int biggest_radius = 0;
            Point center_biggest;

            for (size_t j = 0; j < nestedObjects.size(); j++)
            {
                Rect nr = nestedObjects[j];
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
                radius = cvRound((nr.width + nr.height)*0.25*scale);

                if (radius > biggest_radius){
                    center_biggest.x = center.x;
                    center_biggest.y = center.y;
                    biggest_radius = radius;
                }
                //circle(img, center, radius, color3, 3, 8, 0);
            }
            circle(img, center_biggest, biggest_radius, color3, 3, 8, 0);   
        }
    
        // Show Processed Image with detected faces
        imshow( "Face Detection", img );
    }
    ```
3. <details><summary><b>Update the CMakeLists.txt File</b></summary>

    ```console 
    user@DESKTOP:~$ vim CMakeLists.txt
    ```

    ```cmake
    cmake_minimum_required(VERSION 2.8)

    project( OpenCVTesting )
    find_package( OpenCV REQUIRED )

    include_directories( ${OpenCV_INCLUDE_DIRS} )

    add_executable( DisplayWebcam camera.cpp )
    add_executable( FaceDetect FaceDetect.cpp )

    target_link_libraries( DisplayWebcam ${OpenCV_LIBS} )
    target_link_libraries( FaceDetect ${OpenCV_LIBS} )
    ```

4. <details><summary><b>Recompile the workspace</b></summary>

    ```console 
    user@DESKTOP:~$ cmake .
    user@DESKTOP:~$ make
    ```

5. <details><summary><b>Run the deep-learning object detection binary</b></summary>
    
    ```console 
    user@DESKTOP:~$ ./DnnObjectDetect --config=yolov3.cfg --model=yolov3.weights --classes=object_detection_classes_yolov3.txt --width=608 --height=608 --scale=0.00392 --rgb
    ```

</details>

<details><summary><b>Using Deep-Learning for Object Detection</b></summary>

1. <details><summary><b>Download DL model configuration and pre-trained weight files</b></summary>
    
    The model configuration file (yolov3.cfg) is provided in the repository within the 'yolov3' folder. However, the pre-trained weights file (yolov3.weights) is too large of a file to push to GitHub without large file storage, so you will need to download it from [this link](https://pjreddie.com/media/files/yolov3.weights) and place it into the 'yolov3' folder. These files as well as more can be found [here](https://pjreddie.com/darknet/yolo/).

2. <details><summary><b>Create the source file for object detection</b></summary>

    Make the source file:
    ```console 
    user@DESKTOP:~$ vim DnnObjectDetect.cpp
    ```

    Add the following code:
    ```cpp
    #include <fstream>
    #include <sstream>

    #include <opencv2/dnn.hpp>
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>

    #if defined(CV_CXX11) && defined(HAVE_THREADS)
    #define USE_THREADS 1
    #endif

    #ifdef USE_THREADS
    #include <mutex>
    #include <thread>
    #include <queue>
    #endif

    #include "common.hpp"

    std::string keys =
        "{ help  h     | | Print help message. }"
        "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
        "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
        "{ device      |  0 | camera device number. }"
        "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
        "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
        "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
        "{ thr         | .5 | Confidence threshold. }"
        "{ nms         | .4 | Non-maximum suppression threshold. }"
        "{ backend     |  0 | Choose one of computation backends: "
                            "0: automatically (by default), "
                            "1: Halide language (http://halide-lang.org/), "
                            "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "3: OpenCV implementation }"
        "{ target      | 0 | Choose one of target computation devices: "
                            "0: CPU target (by default), "
                            "1: OpenCL, "
                            "2: OpenCL fp16 (half-float precision), "
                            "3: VPU }"
        "{ async       | 0 | Number of asynchronous forwards at the same time. "
                            "Choose 0 for synchronous mode }";

    using namespace cv;
    using namespace dnn;

    float confThreshold, nmsThreshold;
    std::vector<std::string> classes;

    inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                        const Scalar& mean, bool swapRB);

    void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int backend);

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    void callback(int pos, void* userdata);

    #ifdef USE_THREADS
    template <typename T>
    class QueueFPS : public std::queue<T>
    {
    public:
        QueueFPS() : counter(0) {}

        void push(const T& entry)
        {
            std::lock_guard<std::mutex> lock(mutex);

            std::queue<T>::push(entry);
            counter += 1;
            if (counter == 1)
            {
                // Start counting from a second frame (warmup).
                tm.reset();
                tm.start();
            }
        }

        T get()
        {
            std::lock_guard<std::mutex> lock(mutex);
            T entry = this->front();
            this->pop();
            return entry;
        }

        float getFPS()
        {
            tm.stop();
            double fps = counter / tm.getTimeSec();
            tm.start();
            return static_cast<float>(fps);
        }

        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (!this->empty())
                this->pop();
        }

        unsigned int counter;

    private:
        TickMeter tm;
        std::mutex mutex;
    };
    #endif  // USE_THREADS

    int main(int argc, char** argv)
    {
        CommandLineParser parser(argc, argv, keys);

        const std::string modelName = parser.get<String>("@alias");
        const std::string zooFile = parser.get<String>("zoo");

        keys += genPreprocArguments(modelName, zooFile);

        parser = CommandLineParser(argc, argv, keys);
        parser.about("Use this script to run object detection deep learning networks using OpenCV.");
        if (argc == 1 || parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        confThreshold = parser.get<float>("thr");
        nmsThreshold = parser.get<float>("nms");
        float scale = parser.get<float>("scale");
        Scalar mean = parser.get<Scalar>("mean");
        bool swapRB = parser.get<bool>("rgb");
        int inpWidth = parser.get<int>("width");
        int inpHeight = parser.get<int>("height");
        size_t async = parser.get<int>("async");
        CV_Assert(parser.has("model"));
        std::string modelPath = findFile(parser.get<String>("model"));
        std::string configPath = findFile(parser.get<String>("config"));

        // Open file with classes names.
        if (parser.has("classes"))
        {
            std::string file = parser.get<String>("classes");
            std::ifstream ifs(file.c_str());
            if (!ifs.is_open())
                CV_Error(Error::StsError, "File " + file + " not found");
            std::string line;
            while (std::getline(ifs, line))
            {
                classes.push_back(line);
            }
        }

        // Load a model.
        Net net = readNet(modelPath, configPath, parser.get<String>("framework"));
        int backend = parser.get<int>("backend");
        net.setPreferableBackend(backend);
        net.setPreferableTarget(parser.get<int>("target"));
        std::vector<String> outNames = net.getUnconnectedOutLayersNames();

        // Create a window
        static const std::string kWinName = "Deep learning object detection in OpenCV";
        namedWindow(kWinName, WINDOW_NORMAL);
        int initialConf = (int)(confThreshold * 100);
        createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

        // Open a video file or an image file or a camera stream.
        VideoCapture cap;
        if (parser.has("input"))
            cap.open(parser.get<String>("input"));
        else
            cap.open(parser.get<int>("device"));

    #ifdef USE_THREADS
        bool process = true;

        // Frames capturing thread
        QueueFPS<Mat> framesQueue;
        std::thread framesThread([&](){
            Mat frame;
            while (process)
            {
                cap >> frame;
                if (!frame.empty())
                    framesQueue.push(frame.clone());
                else
                    break;
            }
        });

        // Frames processing thread
        QueueFPS<Mat> processedFramesQueue;
        QueueFPS<std::vector<Mat> > predictionsQueue;
        std::thread processingThread([&](){
            std::queue<AsyncArray> futureOutputs;
            Mat blob;
            while (process)
            {
                // Get a next frame
                Mat frame;
                {
                    if (!framesQueue.empty())
                    {
                        frame = framesQueue.get();
                        if (async)
                        {
                            if (futureOutputs.size() == async)
                                frame = Mat();
                        }
                        else
                            framesQueue.clear();  // Skip the rest of frames
                    }
                }

                // Process the frame
                if (!frame.empty())
                {
                    preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
                    processedFramesQueue.push(frame);

                    if (async)
                    {
                        futureOutputs.push(net.forwardAsync());
                    }
                    else
                    {
                        std::vector<Mat> outs;
                        net.forward(outs, outNames);
                        predictionsQueue.push(outs);
                    }
                }

                while (!futureOutputs.empty() &&
                    futureOutputs.front().wait_for(std::chrono::seconds(0)))
                {
                    AsyncArray async_out = futureOutputs.front();
                    futureOutputs.pop();
                    Mat out;
                    async_out.get(out);
                    predictionsQueue.push({out});
                }
            }
        });

        // Postprocessing and rendering loop
        while (waitKey(1) < 0)
        {
            if (predictionsQueue.empty())
                continue;

            std::vector<Mat> outs = predictionsQueue.get();
            Mat frame = processedFramesQueue.get();

            postprocess(frame, outs, net, backend);

            if (predictionsQueue.counter > 1)
            {
                std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
                putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

                label = format("Network: %.2f FPS", predictionsQueue.getFPS());
                putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

                label = format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
                putText(frame, label, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            }
            imshow(kWinName, frame);
        }

        process = false;
        framesThread.join();
        processingThread.join();

    #else  // USE_THREADS
        if (async)
            CV_Error(Error::StsNotImplemented, "Asynchronous forward is supported only with Inference Engine backend.");

        // Process frames.
        Mat frame, blob;
        while (waitKey(1) < 0)
        {
            cap >> frame;
            if (frame.empty())
            {
                waitKey();
                break;
            }

            preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);

            std::vector<Mat> outs;
            net.forward(outs, outNames);

            postprocess(frame, outs, net, backend);

            // Put efficiency information.
            std::vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            std::string label = format("Inference time: %.2f ms", t);
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 5);

            imshow(kWinName, frame);
        }
    #endif  // USE_THREADS
        return 0;
    }

    inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                        const Scalar& mean, bool swapRB)
    {
        static Mat blob;
        // Create a 4D blob from a frame.
        if (inpSize.width <= 0) inpSize.width = frame.cols;
        if (inpSize.height <= 0) inpSize.height = frame.rows;
        blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

        // Run a model.
        net.setInput(blob, "", scale, mean);
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
        {
            resize(frame, frame, inpSize);
            Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
    }

    void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend)
    {
        static std::vector<int> outLayers = net.getUnconnectedOutLayers();
        static std::string outLayerType = net.getLayer(outLayers[0])->type;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;
        if (outLayerType == "DetectionOutput")
        {
            // Network produces output blob with a shape 1x1xNx7 where N is a number of
            // detections and an every detection is a vector of values
            // [batchId, classId, confidence, left, top, right, bottom]
            CV_Assert(outs.size() > 0);
            for (size_t k = 0; k < outs.size(); k++)
            {
                float* data = (float*)outs[k].data;
                for (size_t i = 0; i < outs[k].total(); i += 7)
                {
                    float confidence = data[i + 2];
                    if (confidence > confThreshold)
                    {
                        int left   = (int)data[i + 3];
                        int top    = (int)data[i + 4];
                        int right  = (int)data[i + 5];
                        int bottom = (int)data[i + 6];
                        int width  = right - left + 1;
                        int height = bottom - top + 1;
                        if (width <= 2 || height <= 2)
                        {
                            left   = (int)(data[i + 3] * frame.cols);
                            top    = (int)(data[i + 4] * frame.rows);
                            right  = (int)(data[i + 5] * frame.cols);
                            bottom = (int)(data[i + 6] * frame.rows);
                            width  = right - left + 1;
                            height = bottom - top + 1;
                        }
                        classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                        boxes.push_back(Rect(left, top, width, height));
                        confidences.push_back(confidence);
                    }
                }
            }
        }
        else if (outLayerType == "Region")
        {
            for (size_t i = 0; i < outs.size(); ++i)
            {
                // Network produces output blob with a shape NxC where N is a number of
                // detected objects and C is a number of classes + 4 where the first 4
                // numbers are [center_x, center_y, width, height]
                float* data = (float*)outs[i].data;
                for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
                {
                    Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                    Point classIdPoint;
                    double confidence;
                    minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

        // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
        // or NMS is required if number of outputs > 1
        if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
        {
            std::map<int, std::vector<size_t> > class2indices;
            for (size_t i = 0; i < classIds.size(); i++)
            {
                if (confidences[i] >= confThreshold)
                {
                    class2indices[classIds[i]].push_back(i);
                }
            }
            std::vector<Rect> nmsBoxes;
            std::vector<float> nmsConfidences;
            std::vector<int> nmsClassIds;
            for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
            {
                std::vector<Rect> localBoxes;
                std::vector<float> localConfidences;
                std::vector<size_t> classIndices = it->second;
                for (size_t i = 0; i < classIndices.size(); i++)
                {
                    localBoxes.push_back(boxes[classIndices[i]]);
                    localConfidences.push_back(confidences[classIndices[i]]);
                }
                std::vector<int> nmsIndices;
                NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
                for (size_t i = 0; i < nmsIndices.size(); i++)
                {
                    size_t idx = nmsIndices[i];
                    nmsBoxes.push_back(localBoxes[idx]);
                    nmsConfidences.push_back(localConfidences[idx]);
                    nmsClassIds.push_back(it->first);
                }
            }
            boxes = nmsBoxes;
            classIds = nmsClassIds;
            confidences = nmsConfidences;
        }

        for (size_t idx = 0; idx < boxes.size(); ++idx)
        {
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                    box.x + box.width, box.y + box.height, frame);
        }
    }

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
    {
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 10);

        std::string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ": " + label;
        }

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);

        top = max(top, labelSize.height);
        rectangle(frame, Point(left, top - labelSize.height),
                Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(), 5);
    }

    void callback(int pos, void*)
    {
        confThreshold = pos * 0.01f;
    }

    ```

3. <details><summary><b>Update the CMakeLists.txt File</b></summary>

    ```console 
    user@DESKTOP:~$ vim CMakeLists.txt
    ```

    ```cmake
    cmake_minimum_required(VERSION 2.8)

    project( OpenCVTesting )
    find_package( OpenCV REQUIRED )

    include_directories( ${OpenCV_INCLUDE_DIRS} )

    add_executable( DisplayWebcam camera.cpp )
    add_executable( FaceDetect FaceDetect.cpp )
    add_executable( DnnObjectDetect DnnObjectDetect.cpp )

    target_link_libraries( DisplayWebcam ${OpenCV_LIBS} )
    target_link_libraries( FaceDetect ${OpenCV_LIBS} )
    target_link_libraries( DnnObjectDetect ${OpenCV_LIBS} )
    ```

4. <details><summary><b>Recompile the workspace</b></summary>

    ```console 
    user@DESKTOP:~$ cmake .
    user@DESKTOP:~$ make
    ```

5. <details><summary><b>Run the deep-learning object detection binary</b></summary>
    
    ```console 
    user@DESKTOP:~$ ./DnnObjectDetect --config=yolov3/yolov3.cfg --model=yolov3/yolov3.weights --classes=yolov3/object_detection_classes_yolov3.txt --width=608 --height=608 --scale=0.00392 --rgb
    ```

</details>
