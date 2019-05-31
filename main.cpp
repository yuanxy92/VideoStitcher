#include <opencv2/opencv.hpp>
#include "src/stitching.hpp"

int main(int argc, char* argv[]) {
	//std::vector<int> inds = {1, 15, 16};
	//std::vector<cv::Mat> imgs;
	//for (int i = 0; i < inds.size(); i++) {
	//	imgs.push_back(cv::imread(cv::format("E:/data/giga_paper/66/small/local_%02d.jpg", inds[i])));
	//	cv::resize(imgs[i], imgs[i], cv::Size(2000, 1500));
	//}

	std::string dir = "E:/data/giga_paper/360/5+9_G+L_0530";
	std::vector<cv::VideoCapture> readers(5);
	cv::VideoWriter writer;
	readers[0].open(cv::format("%s/MIX_CUCAU1731012_CUCAU1725012.mp4", dir.c_str()));
	readers[1].open(cv::format("%s/MIX_CUCAU1731016_CUCAU1829016.mp4", dir.c_str()));
	readers[2].open(cv::format("%s/MIX_CUCAU1731017_CUCAU1724014.mp4", dir.c_str()));
	readers[3].open(cv::format("%s/MIX_CUCAU1829012_CUCAU1731034.mp4", dir.c_str()));
	readers[4].open(cv::format("%s/MIX_CUCAU1829020_CUCAU1731023.mp4", dir.c_str()));

	cv::Ptr<mycv::VideoStitcher> stitcher = mycv::VideoStitcher::create();
	cv::Mat pano;
	stitcher->stitch(readers, writer);

	return 0;
}