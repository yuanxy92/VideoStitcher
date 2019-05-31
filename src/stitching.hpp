/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef MY_STITCHING_STITCHER_HPP
#define MY_STITCHING_STITCHER_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"

#include <opencv2/opencv.hpp>


#if defined(Status)
#  warning Detected X11 'Status' macro definition, it can cause build conflicts. Please, include this header before any X11 headers.
#endif


/**
@defgroup stitching Images stitching

This figure illustrates the stitching module pipeline implemented in the Stitcher class. Using that
class it's possible to configure/remove some steps, i.e. adjust the stitching pipeline according to
the particular needs. All building blocks from the pipeline are available in the detail namespace,
one can combine and use them separately.

The implemented stitching pipeline is very similar to the one proposed in @cite BL07 .

![stitching pipeline](StitchingPipeline.jpg)

Camera models
-------------

There are currently 2 camera models implemented in stitching pipeline.

- _Homography model_ expecting perspective transformations between images
  implemented in @ref cv::detail::BestOf2NearestMatcher cv::detail::HomographyBasedEstimator
  cv::detail::BundleAdjusterReproj cv::detail::BundleAdjusterRay
- _Affine model_ expecting affine transformation with 6 DOF or 4 DOF implemented in
  @ref cv::detail::AffineBestOf2NearestMatcher cv::detail::AffineBasedEstimator
  cv::detail::BundleAdjusterAffine cv::detail::BundleAdjusterAffinePartial cv::AffineWarper

Homography model is useful for creating photo panoramas captured by camera,
while affine-based model can be used to stitch scans and object captured by
specialized devices. Use @ref cv::Stitcher::create to get preconfigured pipeline for one
of those models.

@note
Certain detailed settings of @ref cv::Stitcher might not make sense. Especially
you should not mix classes implementing affine model and classes implementing
Homography model, as they work with different transformations.

@{
    @defgroup stitching_match Features Finding and Images Matching
    @defgroup stitching_rotation Rotation Estimation
    @defgroup stitching_autocalib Autocalibration
    @defgroup stitching_warp Images Warping
    @defgroup stitching_seam Seam Estimation
    @defgroup stitching_exposure Exposure Compensation
    @defgroup stitching_blend Image Blenders
@}
  */

namespace mycv {

//! @addtogroup stitching
//! @{

/** @example samples/cpp/stitching.cpp
A basic example on image stitching
*/

/** @example samples/python/stitching.py
A basic example on image stitching in Python.
*/

/** @example samples/cpp/stitching_detailed.cpp
A detailed example on image stitching
*/

/** @brief High level image stitcher.

It's possible to use this class without being aware of the entire stitching pipeline. However, to
be able to achieve higher stitching stability and quality of the final images at least being
familiar with the theory is recommended.

@note
-   A basic example on image stitching can be found at
    opencv_source_code/samples/cpp/stitching.cpp
-   A basic example on image stitching in Python can be found at
    opencv_source_code/samples/python/stitching.py
-   A detailed example on image stitching can be found at
    opencv_source_code/samples/cpp/stitching_detailed.cpp
 */
class VideoStitcher
{
public:
    /**
     * When setting a resolution for stitching, this values is a placeholder
     * for preserving the original resolution.
     */
    static constexpr const double ORIG_RESOL = -1.0;

    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };

    enum Mode
    {
        /** Mode for creating photo panoramas. Expects images under perspective
        transformation and projects resulting pano to sphere.

        @sa detail::BestOf2NearestMatcher SphericalWarper
        */
        PANORAMA = 0,
        /** Mode for composing scans. Expects images under affine transformation does
        not compensate exposure by default.

        @sa detail::AffineBestOf2NearestMatcher AffineWarper
        */
        SCANS = 1,

    };

    /** @brief Creates a Stitcher configured in one of the stitching modes.

    @param mode Scenario for stitcher operation. This is usually determined by source of images
    to stitch and their transformation. Default parameters will be chosen for operation in given
    scenario.
    @return Stitcher class instance.
     */
    static cv::Ptr<VideoStitcher> create(Mode mode = VideoStitcher::PANORAMA);

    double registrationResol() const { return registr_resol_; }
    void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }

    double seamEstimationResol() const { return seam_est_resol_; }
    void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }

    double compositingResol() const { return compose_resol_; }
    void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }

    double panoConfidenceThresh() const { return conf_thresh_; }
    void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    bool waveCorrection() const { return do_wave_correct_; }
    void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }

    cv::detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
    void setWaveCorrectKind(cv::detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }

    cv::Ptr<cv::Feature2D> featuresFinder() { return features_finder_; }
    const cv::Ptr<cv::Feature2D> featuresFinder() const { return features_finder_; }
    void setFeaturesFinder(cv::Ptr<cv::Feature2D> features_finder)
        { features_finder_ = features_finder; }

    cv::Ptr<cv::detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
    const cv::Ptr<cv::detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
    void setFeaturesMatcher(cv::Ptr<cv::detail::FeaturesMatcher> features_matcher)
        { features_matcher_ = features_matcher; }

    const cv::UMat& matchingMask() const { return matching_mask_; }
    void setMatchingMask(const cv::UMat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        matching_mask_ = mask.clone();
    }

    cv::Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
    const cv::Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
    void setBundleAdjuster(cv::Ptr<cv::detail::BundleAdjusterBase> bundle_adjuster)
        { bundle_adjuster_ = bundle_adjuster; }

    cv::Ptr<cv::detail::Estimator> estimator() { return estimator_; }
    const cv::Ptr<cv::detail::Estimator> estimator() const { return estimator_; }
    void setEstimator(cv::Ptr<cv::detail::Estimator> estimator)
        { estimator_ = estimator; }

    cv::Ptr<cv::WarperCreator> warper() { return warper_; }
    const cv::Ptr<cv::WarperCreator> warper() const { return warper_; }
    void setWarper(cv::Ptr<cv::WarperCreator> creator) { warper_ = creator; }

    cv::Ptr<cv::detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
    const cv::Ptr<cv::detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
    void setExposureCompensator(cv::Ptr<cv::detail::ExposureCompensator> exposure_comp)
        { exposure_comp_ = exposure_comp; }

	cv::Ptr<cv::detail::SeamFinder> seamFinder() { return seam_finder_; }
    const cv::Ptr<cv::detail::SeamFinder> seamFinder() const { return seam_finder_; }
    void setSeamFinder(cv::Ptr<cv::detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

	cv::Ptr<cv::detail::Blender> blender() { return blender_; }
    const cv::Ptr<cv::detail::Blender> blender() const { return blender_; }
    void setBlender(cv::Ptr<cv::detail::Blender> b) { blender_ = b; }

    /** @brief These functions try to match the given images and to estimate rotations of each camera.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    @return Status code.
     */
    Status estimateTransform(cv::InputArrayOfArrays images, 
		cv::InputArrayOfArrays masks = cv::noArray());

    /** @overload */
    Status composePanorama(cv::OutputArray pano);
    /** @brief These functions try to compose the given images (or images stored internally from the other function
    calls) into the final pano under the assumption that the image transformations were estimated
    before.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param pano Final pano.
    @return Status code.
     */
    Status composePanorama(cv::InputArrayOfArrays images, cv::OutputArray pano);
	Status composePanorama(std::vector<cv::VideoCapture> readers, std::string outname);


    /** @overload */
    Status stitch(cv::InputArrayOfArrays images, cv::OutputArray pano);
    /** @brief These functions try to stitch the given images.

    @param images Input images.
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    @param pano Final pano.
    @return Status code.
     */
    Status stitch(cv::InputArrayOfArrays images, cv::InputArrayOfArrays masks, 
		cv::OutputArray pano);

	Status stitch(std::vector<cv::VideoCapture> readers, cv::VideoWriter writer);


    std::vector<int> component() const { return indices_; }
    std::vector<cv::detail::CameraParams> cameras() const { return cameras_; }
    double workScale() const { return work_scale_; }

private:
    Status matchImages();
    Status estimateCameraParams();

    double registr_resol_;
    double seam_est_resol_;
    double compose_resol_;
    double conf_thresh_;
    cv::Ptr<cv::Feature2D> features_finder_;
    cv::Ptr<cv::detail::FeaturesMatcher> features_matcher_;
    cv::UMat matching_mask_;
    cv::Ptr<cv::detail::BundleAdjusterBase> bundle_adjuster_;
    cv::Ptr<cv::detail::Estimator> estimator_;
    bool do_wave_correct_;
	cv::detail::WaveCorrectKind wave_correct_kind_;
	cv::Ptr<cv::WarperCreator> warper_;
	cv::Ptr<cv::detail::ExposureCompensator> exposure_comp_;
	cv::Ptr<cv::detail::SeamFinder> seam_finder_;
	cv::Ptr<cv::detail::Blender> blender_;

    std::vector<cv::UMat> imgs_;
    std::vector<cv::UMat> masks_;
    std::vector<cv::Size> full_img_sizes_;
    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::UMat> seam_est_imgs_;
    std::vector<int> indices_;
    std::vector<cv::detail::CameraParams> cameras_;
    double work_scale_;
    double seam_scale_;
    double seam_work_aspect_;
    double warped_image_scale_;
};

/**
 * @deprecated use Stitcher::create
 */
CV_DEPRECATED cv::Ptr<VideoStitcher> createStitcher(bool try_use_gpu = false);

/**
 * @deprecated use Stitcher::create
 */
CV_DEPRECATED cv::Ptr<VideoStitcher> createStitcherScans(bool try_use_gpu = false);

//! @} stitching

} // namespace cv

#endif // OPENCV_STITCHING_STITCHER_HPP
