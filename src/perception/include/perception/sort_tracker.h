#ifndef __SORT_TRACKER_H__
#define __SORT_TRACKER_H__

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <cmath>

struct Detection {
    int cls;
    double x;
    double y;
    double L;
    double W;
    double yaw;
};

struct TrackOutput {
    int track_id;
    int cls;
    double x;
    double y;
    double L;
    double W;
    double yaw;
};

class SortTracker {
public:
    SortTracker(int max_age=3, int min_hits=3, double iou_threshold=0.3);
    std::vector<TrackOutput> update(const std::vector<Detection>& detections);

private:
    struct Track;
    std::vector<std::unique_ptr<Track>> tracks_;
    int max_age_;
    int min_hits_;
    double iou_threshold_;

    static std::array<double,4> carla_to_aabb(const Detection& d);
    static double iou_bbox(const std::array<double,4>& A, const std::array<double,4>& B);
    static std::vector<std::vector<double>> compute_cost_matrix(const std::vector<Detection>& dets,
                                                                 const std::vector<std::unique_ptr<Track>>& tracks);
    static std::vector<std::pair<int,int>> hungarian_assignment(const std::vector<std::vector<double>>& cost);
};

#endif
