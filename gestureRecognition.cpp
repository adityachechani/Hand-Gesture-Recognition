//Copyright 2018 Aditya Chechani adityac@bu.edu
//Copyright 2018 Nidhi Tiwari nidhit@bu.edu

//Multiple gesture and hand tracking using Muti-scale Template Matching and Convex defects.

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void SkinDetect(Mat& src, Mat& dst);
int countFingers(Mat& src, Mat& dst);
bool MultipleTemplateMatching(Mat Img_input, Mat Img_template, float thresh, float closeness, vector<Point3f> &List_Matches, int num);

//Calculates the angle between fingers fount count
float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{

	float dist1 = std::sqrt((px1 - cx1)*(px1 - cx1) + (py1 - cy1)*(py1 - cy1));
	float dist2 = std::sqrt((px2 - cx1)*(px2 - cx1) + (py2 - cy1)*(py2 - cy1));

	float Ax, Ay;
	float Bx, By;
	float Cx, Cy;

	//find closest point to C  
	//printf("dist = %lf %lf\n", dist1, dist2);  

	Cx = cx1;
	Cy = cy1;
	if (dist1 < dist2)
	{
		Bx = px1;
		By = py1;
		Ax = px2;
		Ay = py2;


	}
	else {
		Bx = px2;
		By = py2;
		Ax = px1;
		Ay = py1;
	}


	float Q1 = Cx - Ax;
	float Q2 = Cy - Ay;
	float P1 = Bx - Ax;
	float P2 = By - Ay;


	float A = std::acos((P1*Q1 + P2 * Q2) / (std::sqrt(P1*P1 + P2 * P2) * std::sqrt(Q1*Q1 + Q2 * Q2)));

	A = A * 180 / CV_PI;

	return A;
}

int main()
{
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}
	namedWindow("MyVideo", WINDOW_AUTOSIZE);
	Mat frame, template_img;

	//-------------------------------------------------------
	vector<string> template_files{ "close_palm.jpg", "thumbs_up.jpg", "Fist.jpg", "out.png" };
	vector<string> template_names{ "Palm", "Thumbs Up", "Fist", "Out" };
	vector<Mat> Templates;
	int tH, tW;
	
	//Reading Template images and storing in a vector
	for (int i = 0; i < template_files.size(); i++)
	{
		template_img = imread(template_files[i], 0);
		resize(template_img, template_img, Size(200, 200));
		threshold(template_img, template_img, 10, 255, CV_THRESH_BINARY);
		Templates.push_back(template_img);
		tH = template_img.cols;
		tW = template_img.rows;
	}

	namedWindow("Gestures", WINDOW_AUTOSIZE);
	namedWindow("Finger Count", WINDOW_AUTOSIZE);

	double SCALE_START = 1.0;
	double SCALE_END = 0.2;
	double SCALE_POINTS = 8.0;
	double INTERVAL = (SCALE_START - SCALE_END) / SCALE_POINTS;

	while (true)
	{
		// read a new frame from video
		Mat resized;
		Mat frame(Size(640, 420), CV_8UC3);
		Mat fingerFrame(Size(640, 420), CV_8UC3);
		Mat skinMask(Size(640, 420), CV_8UC3);
		bool bSuccess = cap.read(frame);

		fingerFrame = frame.clone();

		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		//show the frame in "MyVideo" window
		//imshow("MyVideo", frame);
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

		//Create a skin Mask using lower and upper bound hsv values
		SkinDetect(frame, skinMask);

		vector<Point3f> List_Matches;
		countFingers(skinMask, fingerFrame);
		vector<Mat> scaled_images;
		int rows = skinMask.rows;
		int cols = skinMask.cols;

		for (float scale = SCALE_START; scale >= SCALE_END; scale -= INTERVAL)
		{
			//resize the image to different scles for accurate template matching
			resize(skinMask, resized, Size(0, 0), scale, scale);

			Mat temp(rows, cols, CV_8UC1, Scalar(0));
			resized.copyTo(temp(Rect(0, 0, resized.cols, resized.rows)));
			resize(temp, temp, Size(200, 200));
			scaled_images.push_back(temp);

			//store the ratio of the new image to the original image for graphical display bounding box
			float ratio = skinMask.cols / resized.cols;

			//exit the while loop if scaled image is smaller than template image
			if (tW > resized.cols || tH > resized.rows) break;

			//threshold of 0.8 for matching
			float thresh = 0.8;
			//This variable eliminates the points closer to each other
			float closeness = 1.0;

			List_Matches.clear();
			for (int i = 0; i < Templates.size(); i++)
			{
				//Matches multiple gestures in a frame
				MultipleTemplateMatching(resized, Templates[i], thresh, closeness, List_Matches, i);
			}

			int startX, startY, endX, endY;
			if (List_Matches.size() > 0)
			{
				//This for loop is used to draw the bounding box and display the type of gesture
				for (int i = 0; i < List_Matches.size(); i++)
				{
					startX = List_Matches[i].x / scale;
					startY = List_Matches[i].y / scale;
					endX = (List_Matches[i].x + tW) / scale;
					endY = (List_Matches[i].y + tH) / scale;
					rectangle(frame, Point(startX, startY), Point(endX, endY), Scalar(0, 0, 255), 2);
					if (List_Matches[i].z == 0)
						startX = 100;
					if (List_Matches[i].z == 1)
						startX = 200;
					if (List_Matches[i].z == 2)
						startX = 400;
					if (List_Matches[i].z == 3)
						startX = 500;
					putText(frame, template_names[List_Matches[i].z], Point(startX, 50),
						FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(255, 0, 0), 2, CV_AA);
				}
			}
		}

		//To concat images together for display
		Mat img_concat;
		hconcat(frame, fingerFrame, img_concat);
		imshow("Warped images comparison", img_concat);
		Mat skin_images;
		hconcat(scaled_images, skin_images);
		resize(skin_images, skin_images, Size(img_concat.cols, skin_images.rows));
		imshow("SKIN IMAges", skin_images);

	}
	cap.release();
	return 0;
}

/*
*@params: binary image of hand, destination image
*Uses Convex hull and its defects to count the number of fingers
*/
int countFingers(Mat& skinMask, Mat& fingerFrame)
{
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	//find all contours in the image and select the one with maximum pixel area
	findContours(skinMask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CLOCKWISE, Point(0, 0));
	size_t largestContour = 0;
	for (size_t i = 1; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > contourArea(contours[largestContour]))
			largestContour = i;
	}
	//draw the contour with maximum area
	drawContours(fingerFrame, contours, largestContour, Scalar(0, 0, 255), 1);
	int count;

	if (!contours.empty())
	{
		vector<vector<Point>> hull(1);
		//calculate hull points for the max area contour and draw a bounding box around it
		convexHull(Mat(contours[largestContour]), hull[0], false);
		drawContours(fingerFrame, hull, 0, Scalar(0, 255, 0), 3);
		if (hull[0].size() > 2)
		{
			vector<int> hullIndexes;
			convexHull(Mat(contours[largestContour]), hullIndexes, true);
			vector<Vec4i> defects;
			//calculate defects from the hull points for finger count
			//Defects gives us three points: initial, middle and end
			convexityDefects(Mat(contours[largestContour]), hullIndexes, defects);
			Rect boundingBox = boundingRect(hull[0]);
			rectangle(fingerFrame, boundingBox, Scalar(255, 0, 0));
			Point center = Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
			std::vector<Point> validPoints;
			for (size_t i = 0; i < defects.size(); i++)
			{
				Point p1 = contours[largestContour][defects[i][0]];
				Point p2 = contours[largestContour][defects[i][1]];
				Point p3 = contours[largestContour][defects[i][2]];
				//calculate angle between initial point and center of bounding box
				double angle = atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
				//calculate angle between first and last defect point
				double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
				//calculate the distance between initial and end defect point
				double length = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2));
				if (angle > -30 && angle < 190 && abs(inAngle) > 10 && abs(inAngle) < 120 && length > 0.2 * boundingBox.height)
				{
					validPoints.push_back(p1);
				}
			}
			count = validPoints.size();
			for (size_t i = 0; i < validPoints.size(); i++)
			{
				cv::circle(fingerFrame, validPoints[i], 9, cv::Scalar(0, 255, 0), 2);
			}
		}
	}
	else
		return 0;
	putText(fingerFrame, "Number of Fingers: " + to_string(count), Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 2, CV_AA);
	return count;
}

/*
*@params: bgr image, destination image
*Detects skin color based on the lower and upper bound in the hsv color space
*/
void SkinDetect(Mat& frame, Mat& skinMask)
{
	Mat converted(Size(640, 420), CV_8UC3);
	Scalar lower = Scalar(0, 50, 0);
	Scalar upper = Scalar(70, 170, 255);

	Size kSize;
	kSize.height = 3;
	kSize.width = 3;
	double sigma = 0.3*(3 / 2 - 1) + 0.8;
	GaussianBlur(frame, frame, kSize, sigma, 0.0, 4);
	cvtColor(frame, converted, COLOR_BGR2HSV);
	inRange(converted, lower, upper, skinMask);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(skinMask, skinMask, kernel, Point(-1, -1), 2, 1, 1);
	erode(skinMask, skinMask, kernel, Point(-1, -1), 3, 1, 1);
	GaussianBlur(skinMask, skinMask, Size(3, 3), 0, 0);
}


/*
*@params: input img, template, threshold, closeness factor, destination vector for matched points, template index
* Implements multi label template matching and stores the highest threshold points above threshold
*@returns boolean
*/
bool MultipleTemplateMatching(Mat Img_input, Mat Img_template, float thresh, float closeness, vector<Point3f> &List_Matches, int num)
{
	Mat result;
	Size Template_size = Img_template.size();
	Size ClosenessRadius((Template_size.width / 2) * closeness, (Template_size.height / 2) * closeness);

	matchTemplate(Img_input, Img_template, result, TM_CCORR_NORMED);
	threshold(result, result, thresh, 1.0, THRESH_TOZERO);
	while (true)
	{
		double minval, maxval;
		Point minloc, maxloc;
		minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
		if (maxval >= thresh)
		{
			List_Matches.push_back(Point3f(maxloc.x, maxloc.y, num));
			rectangle(result, Point2f(maxloc.x - ClosenessRadius.width, maxloc.y - ClosenessRadius.height),
				Point2f(maxloc.x + ClosenessRadius.width, maxloc.y + ClosenessRadius.height), Scalar(0), -1);
		}
		else
			break;
	}
	return true;
}
