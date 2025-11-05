#include "sort_tracker.hpp"
#include <Eigen/Dense>
#include <limits>
#include <algorithm>

struct SimpleKF {
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;

    SimpleKF() {}
    void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }
    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd Si = S.inverse();
        Eigen::MatrixXd K = P * H.transpose() * Si;
        x = x + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.size(), x.size());
        P = (I - K * H) * P;
    }
};

struct SortTracker::Track {
    int id;
    int hits;
    int age;
    int time_since_update;
    int state; // 1 tentative, 2 confirmed, 3 lost, 4 deleted
    int cls;

    // kfs
    SimpleKF kf_pos; // 4x1 [x,y,vx,vy]
    SimpleKF kf_yaw; // 2x1 [yaw, dyaw]
    SimpleKF kf_length; // 2x1
    SimpleKF kf_width; // 2x1

    double car_length;
    double car_width;
    double car_yaw;

    Track(int id_, const Detection& d, int confirm_hits)
        : id(id_), hits(1), age(1), time_since_update(0), state(1), cls(d.cls)
    {
        // init pos kf (4-state: x,y,vx,vy)
        kf_pos.x = Eigen::VectorXd(4);
        kf_pos.x << d.x, d.y, 0.0, 0.0;
        kf_pos.P = Eigen::MatrixXd::Identity(4,4) * 1000.0;
        kf_pos.F = Eigen::MatrixXd(4,4);
        kf_pos.F << 1,0,1,0,
                    0,1,0,1,
                    0,0,1,0,
                    0,0,0,1;
        kf_pos.H = Eigen::MatrixXd(2,4);
        kf_pos.H << 1,0,0,0,
                  0,1,0,0;
        kf_pos.Q = Eigen::MatrixXd::Identity(4,4) * 0.1;
        kf_pos.R = Eigen::MatrixXd::Identity(2,2) * 10.0;

        // yaw/size KFs (2 state: value, rate)
        auto init_2d = [](double v, double Qscale, double Rscale) {
            SimpleKF kf;
            kf.x = Eigen::VectorXd(2); kf.x << v, 0.0;
            kf.P = Eigen::MatrixXd::Identity(2,2) * 10.0;
            kf.F = Eigen::MatrixXd(2,2); kf.F << 1,1, 0,1;
            kf.H = Eigen::MatrixXd(1,2); kf.H << 1,0;
            kf.Q = Eigen::MatrixXd::Identity(2,2) * Qscale;
            kf.R = Eigen::MatrixXd::Identity(1,1) * Rscale;
            return kf;
        };
        kf_yaw = init_2d(d.yaw, 0.01, 1.0);
        kf_length = init_2d(d.L, 0.001, 1.0);
        kf_width = init_2d(d.W, 0.001, 1.0);

        car_length = d.L; car_width = d.W; car_yaw = d.yaw;
    }

    void predict() {
        kf_pos.predict();
        kf_yaw.predict();
        kf_length.predict();
        kf_width.predict();
        // normalize yaw
        car_yaw = wrap_deg(kf_yaw.x(0));
        kf_yaw.x(0) = car_yaw;
        car_length = kf_length.x(0);
        car_width = kf_width.x(0);
        age += 1;
        time_since_update += 1;
    }

    void update(const Detection& d) {
        // pos
        Eigen::Vector2d zpos; zpos << d.x, d.y;
        kf_pos.update(zpos);
        // length/width
        Eigen::VectorXd zl(1); zl << d.L;
        Eigen::VectorXd zw(1); zw << d.W;
        kf_length.update(zl);
        kf_width.update(zw);
        // yaw: adjust to nearest equivalent (period 180)
        double ref = kf_yaw.x(0);
        double adj = nearest_equivalent_deg(d.yaw, ref, 180.0);
        Eigen::VectorXd zy(1); zy << adj;
        kf_yaw.update(zy);

        car_length = kf_length.x(0);
        car_width = kf_width.x(0);
        car_yaw = wrap_deg(kf_yaw.x(0));
        kf_yaw.x(0) = car_yaw;

        time_since_update = 0;
        hits += 1;
        if(state==1 && hits>=3) state = 2;
    }

    std::array<double,6> get_state() const {
        double x = kf_pos.x(0);
        double y = kf_pos.x(1);
        return {0.0, x, y, car_length, car_width, car_yaw};
    }
};

// Utility functions
static double wrap_deg(double angle) {
    double a = fmod(angle + 180.0, 360.0);
    if (a < 0) a += 360.0;
    return a - 180.0;
}

static double nearest_equivalent_deg(double meas, double ref, double period=360.0) {
    double d = meas - ref;
    d = fmod(d + period/2.0, period) - period/2.0;
    return ref + d;
}

// Static helper implementations
std::array<double,4> SortTracker::carla_to_aabb(const Detection& d) {
    double x_c = d.x, y_c = d.y, L = d.L, W = d.W, yaw = d.yaw * M_PI/180.0;
    double dx = L/2.0, dy = W/2.0;
    double c = cos(yaw), s = sin(yaw);
    std::array<std::array<double,2>,4> corners = {{{dx,dy},{dx,-dy},{-dx,-dy},{-dx,dy}}};
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();
    for(auto &pt: corners) {
        double X = pt[0]*c - pt[1]*s + x_c;
        double Y = pt[0]*s + pt[1]*c + y_c;
        x_min = std::min(x_min, X);
        x_max = std::max(x_max, X);
        y_min = std::min(y_min, Y);
        y_max = std::max(y_max, Y);
    }
    return {x_min, y_min, x_max - x_min, y_max - y_min};
}

double SortTracker::iou_bbox(const std::array<double,4>& A, const std::array<double,4>& B) {
    double xA = std::max(A[0], B[0]);
    double yA = std::max(A[1], B[1]);
    double xB = std::min(A[0]+A[2], B[0]+B[2]);
    double yB = std::min(A[1]+A[3], B[1]+B[3]);
    double iw = std::max(0.0, xB - xA);
    double ih = std::max(0.0, yB - yA);
    double inter = iw*ih;
    double areaA = A[2]*A[3];
    double areaB = B[2]*B[3];
    double denom = areaA + areaB - inter;
    if (denom <= 0) return 0.0;
    return inter / denom;
}

std::vector<std::vector<double>> SortTracker::compute_cost_matrix(const std::vector<Detection>& dets,
                                                                  const std::vector<std::unique_ptr<Track>>& tracks) {
    size_t M = dets.size();
    size_t N = tracks.size();
    std::vector<std::vector<double>> cost(M, std::vector<double>(N, 1.0));
    for(size_t i=0;i<M;++i) {
        auto A = carla_to_aabb(dets[i]);
        for(size_t j=0;j<N;++j) {
            auto tr = tracks[j].get();
            double px = tr->kf_pos.x(0);
            double py = tr->kf_pos.x(1);
            Detection tmp{tr->cls, px, py, tr->car_length, tr->car_width, tr->car_yaw};
            auto B = carla_to_aabb(tmp);
            cost[i][j] = 1.0 - iou_bbox(A, B);
        }
    }
    return cost;
}

std::vector<std::pair<int,int>> SortTracker::hungarian_assignment(const std::vector<std::vector<double>>& cost) {
    int nrows = (int)cost.size();
    int ncols = cost.empty() ? 0 : (int)cost[0].size();
    std::vector<std::pair<int,int>> pairs;
    if(nrows==0 || ncols==0) return pairs;
    struct Item{double c; int r; int col;};
    std::vector<Item> items;
    items.reserve(nrows*ncols);
    for(int i=0;i<nrows;++i) for(int j=0;j<ncols;++j) items.push_back({cost[i][j], i, j});
    std::sort(items.begin(), items.end(), [](const Item&a,const Item&b){return a.c<b.c;});
    std::vector<char> row_used(nrows,0), col_used(ncols,0);
    for(auto &it: items) {
        if(!row_used[it.r] && !col_used[it.col]) {
            pairs.emplace_back(it.r, it.col);
            row_used[it.r]=1; col_used[it.col]=1;
        }
    }
    return pairs;
}

SortTracker::SortTracker(int max_age, int min_hits, double iou_threshold)
    : max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold) {}

std::vector<TrackOutput> SortTracker::update(const std::vector<Detection>& detections) {
    // 1) predict
    for(auto &t : tracks_) {
        t->predict();
    }
    // active tracks
    std::vector<std::unique_ptr<Track>> active;
    for(auto &t: tracks_) if(t->state != 4) active.push_back(std::move(t));
    tracks_.clear();
    tracks_.swap(active);

    // 2) matching
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_tracks;
    std::vector<std::pair<int,int>> matched_pairs;

    int M = (int)detections.size();
    int N = (int)tracks_.size();
    for(int i=0;i<M;++i) unmatched_dets.push_back(i);
    for(int j=0;j<N;++j) unmatched_tracks.push_back(j);

    std::vector<std::vector<double>> cost = compute_cost_matrix(detections, tracks_);
    if(M>0 && N>0) {
        auto pairs = hungarian_assignment(cost);
        for(auto &pr: pairs) {
            int r = pr.first, c = pr.second;
            double iou = 1.0 - cost[r][c];
            if(iou >= iou_threshold_) {
                matched_pairs.push_back(pr);
                // remove from unmatched lists
                unmatched_dets.erase(std::remove(unmatched_dets.begin(), unmatched_dets.end(), r), unmatched_dets.end());
                unmatched_tracks.erase(std::remove(unmatched_tracks.begin(), unmatched_tracks.end(), c), unmatched_tracks.end());
            }
        }
    }

    // 3) update matched
    for(auto &pr: matched_pairs) {
        int di = pr.first, ti = pr.second;
        tracks_[ti]->update(detections[di]);
        // push history could be added
    }

    // 4) unmatched tracks state changes
    for(int ti: unmatched_tracks) {
        auto &tr = tracks_[ti];
        if(tr->state == 2) tr->state = 3; // confirmed -> lost
        else if(tr->state == 3) {
            if(tr->time_since_update > max_age_) tr->state = 4; // deleted
        } else if(tr->state == 1) tr->state = 4;
    }

    // 5) create new tracks for unmatched detections
    static int next_id = 0;
    for(int di: unmatched_dets) {
        auto t = std::make_unique<Track>(next_id++, detections[di], min_hits_);
        tracks_.push_back(std::move(t));
    }

    // remove deleted
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const std::unique_ptr<Track>& t){ return t->state==4; }), tracks_.end());

    // 6) prepare outputs
    std::vector<TrackOutput> outs;
    for(auto &tr: tracks_) {
        if(tr->state==2 || tr->state==3) {
            auto st = tr->get_state();
            TrackOutput o;
            o.track_id = tr->id;
            o.cls = tr->cls;
            o.x = st[1]; o.y = st[2]; o.L = st[3]; o.W = st[4]; o.yaw = st[5];
            outs.push_back(o);
        }
    }
    return outs;
}
