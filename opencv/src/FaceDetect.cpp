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
    eye.load("../haarcascades/haarcascade_lefteye_2splits.xml");
    smile_face.load("../haarcascades/haarcascade_smile.xml");

    // Change path before execution
    cascade.load("../haarcascades/haarcascade_frontalface_default.xml");
 
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
