#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <omp.h>
#include <filesystem>
#include <intrin.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '{'; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << '}'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << '{' << p.first << ", " << p.second << '}'; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << '{'; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << '}'; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << '{'; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << '}'; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<']'<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    static constexpr uint64_t M = INT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next(); }
    inline uint64_t next() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    inline int next_int() { return next() & M; }
    inline int next_int(int mod) { return next() % mod; }
    inline int next_int(int l, int r) { return l + next_int(r - l + 1); }
    inline double next_double() { return next_int() * e; }
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_int(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

/* fast queue */
class FastQueue {
    int front = 0;
    int back = 0;
    int v[4096];
public:
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

class RandomQueue {
    int sz = 0;
    int v[4096];
public:
    inline bool empty() const { return !sz; }
    inline int size() const { return sz; }
    inline void push(int x) { v[sz++] = x; }
    inline void reset() { sz = 0; }
    inline int pop(int i) {
        std::swap(v[i], v[sz - 1]);
        return v[--sz];
    }
    inline int pop(Xorshift& rnd) {
        return pop(rnd.next_int(sz));
    }
};

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif



struct Pos {
    int y, x;
    Pos(int y = -1, int x = -1) : y(y), x(x) {}
    std::string stringify() const {
        return "(" + std::to_string(y) + ", " + std::to_string(x) + ")";
    }
    bool operator<(const Pos& rhs) const {
        return y == rhs.y ? x < rhs.x : y < rhs.y;
    }
    bool operator==(const Pos& rhs) const {
        return y == rhs.y && x == rhs.x;
    }
};
std::istream& operator>>(std::istream& in, Pos& pos) {
    in >> pos.y >> pos.x;
    return in;
}

std::vector<Pos> choose_positions(int L, int N, int seed = 0) {
    std::mt19937_64 engine(seed);
    std::vector<Pos> cands;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            cands.emplace_back(y, x);
        }
    }
    std::shuffle(cands.begin(), cands.end(), engine);
    cands.erase(cands.begin() + N, cands.end());
    std::sort(cands.begin(), cands.end());
    return cands;
}

struct Input {

    int L, N, S;
    std::vector<Pos> landing_pos;

    void load(std::istream& in) {
        in >> L >> N >> S;
        landing_pos.resize(N);
        in >> landing_pos;
    }

    void generate(std::mt19937_64& engine) {
        std::uniform_int_distribution<> dist_int;
        if (L == -1) L = dist_int(engine) % 41 + 10;
        if (N == -1) N = dist_int(engine) % 41 + 60;
        if (S == -1) {
            S = dist_int(engine) % 30 + 1;
            S *= S;
        }
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                landing_pos.emplace_back(y, x);
            }
        }
        std::shuffle(landing_pos.begin(), landing_pos.end(), engine);
        landing_pos.erase(landing_pos.begin() + N, landing_pos.end());
    }

    Input(std::istream& in) { load(in); }

    Input(const std::string& input_file) {
        std::ifstream in(input_file);
        load(in);
    }

    Input(std::mt19937_64& engine, int L, int N, int S) : L(L), N(N), S(S) {
        generate(engine);
    }

};

struct Quantizer {

    const int rate;
    const int intercept;
    const int slope;

    Quantizer(int rate_, int intercept_, int slope_) : rate(rate_), intercept(intercept_), slope(slope_) {}

    int quantize(double value) const {
        return std::clamp((int)round((value - intercept) / slope), 0, rate - 1) * slope + intercept;
    }

    int to_value(int index) const {
        assert(0 <= index && index < rate);
        return intercept + index * slope;
    }

    int to_index(int value) const {
        value -= intercept;
        assert(value % slope == 0);
        int index = value / slope;
        assert(0 <= index && index < rate);
        return index;
    }

    int to_index(double value) const {
        return to_index(quantize(value));
    }

};

template<typename T>
using Grid = std::vector<std::vector<T>>;

struct EncodedGrid;
using EncodedGridPtr = std::shared_ptr<EncodedGrid>;
struct EncodedGrid {

    const int num_neighbors;
    const bool align_parity;
    const Grid<int> grid;
    const std::vector<Pos> displacements;
    const std::vector<int> codes;
    const std::vector<bool> parities;

    EncodedGrid(
        int num_neighbors_,
        bool align_parity_,
        const Grid<int>& grid_,
        const std::vector<Pos> displacements_,
        const std::vector<int>& codes_,
        const std::vector<bool>& parities_
    ) :
        num_neighbors(num_neighbors_),
        align_parity(align_parity_),
        grid(grid_),
        displacements(displacements_),
        codes(codes_),
        parities(parities_) {}

};

EncodedGridPtr find_unique_encoded_grid(
    const Input& input,
    const Quantizer& quantizer,
    int D, // number of neighbors
    bool align_parity,
    int displacement,
    double duration
) {
    Timer timer;

    const int L = input.L;
    const int N = input.N;
    const int Q = quantizer.rate;
    const auto& pos = input.landing_pos;

    assert(N * (align_parity ? 2 : 1) <= (int)pow(Q, D));

    std::vector<int> powers({ 1 });
    while (powers.size() < D) {
        powers.push_back(powers.back() * Q);
    }

    std::vector<Pos> displacements;
    {
        // 6 2 5
        // 3 0 1
        // 7 4 8
        constexpr int dy[] = { 0, 0, -1, 0, 1, -1, -1, 1, 1 };
        constexpr int dx[] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
        std::vector<Pos> tmp, mtmp;
        while (true) {
            if (displacement >= L) {
                displacement--;
                continue;
            }
            tmp.clear();
            mtmp.clear();
            bool valid = true;
            for (int d = 0; d < D; d++) {
                Pos p(dy[d] * displacement, dx[d] * displacement);
                Pos mp(((p.y % L) + L) % L, ((p.x % L) + L) % L);
                if (std::count(mtmp.begin(), mtmp.end(), mp)) {
                    valid = false;
                    break;
                }
                tmp.push_back(p);
                mtmp.push_back(mp);
            }
            if (!valid) {
                displacement--;
                continue;
            }
            break;
        }
        displacements = tmp;
    }
    dump(displacements);

    auto grid = make_vector(0, L, L);
    auto grid_to_id = make_vector(std::vector<std::pair<int, int>>(), L, L);
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        for (int d = 0; d < D; d++) {
            auto [dy, dx] = displacements[d];
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            grid_to_id[ny][nx].emplace_back(i, d);
        }
    }

    Xorshift rnd;

    std::vector<Pos> cands;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            if (!grid_to_id[y][x].empty()) {
                cands.emplace_back(y, x);
                grid[y][x] = rnd.next_int(Q);
            }
        }
    }

    std::vector<int> codes(N);
    std::vector<bool> parities(N);
    std::vector<int> code_ctr((int)pow(Q, D));
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        int code = 0;
        for (int d = 0; d < D; d++) {
            auto [dy, dx] = displacements[d];
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            code += grid[ny][nx] * powers[d];
            parities[i] = parities[i] ^ (grid[ny][nx] & 1);
        }
        codes[i] = code;
        code_ctr[code]++;
    }

    int cost = 0;
    for (int c : code_ctr) {
        int x = std::max(0, c - 1);
        cost += x * x;
    }
    if (align_parity) {
        for (auto p : parities) {
            cost += p; // odd: penalty
        }
    }

    auto pop = [&](int value) {
        assert(code_ctr[value]);
        int px = std::max(0, code_ctr[value] - 1);
        cost -= px * px;
        code_ctr[value]--;
        int nx = std::max(0, code_ctr[value] - 1);
        cost += nx * nx;
    };

    auto push = [&](int value) {
        int px = std::max(0, code_ctr[value] - 1);
        cost -= px * px;
        code_ctr[value]++;
        int nx = std::max(0, code_ctr[value] - 1);
        cost += nx * nx;
    };

    // cands から 1 つ点を選び、その点の値を [0,Q) でランダム変更
    auto change = [&](int idx, int value) {
        int pcost = cost;
        assert(idx < cands.size());
        auto [y, x] = cands[idx];
        assert(grid[y][x] != value);
        int diff = value - grid[y][x];
        bool parity_changed = align_parity & (grid[y][x] ^ value) & 1;
        for (auto [i, d] : grid_to_id[y][x]) {
            pop(codes[i]);
            codes[i] += diff * powers[d];
            push(codes[i]);
            cost += parity_changed ? (parities[i] ? -1 : 1) : 0;
            parities[i] = parity_changed ? !parities[i] : parities[i];
        }
        grid[y][x] = value;
        return cost - pcost;
    };

    auto swap = [&](int idx1, int idx2) {
        assert(idx1 != idx2);
        auto [y1, x1] = cands[idx1];
        auto [y2, x2] = cands[idx2];
        int v1 = grid[y1][x1], v2 = grid[y2][x2];
        assert(v1 != v2);
        int diff = 0;
        diff += change(idx1, v2);
        diff += change(idx2, v1);
        return diff;
    };

    int loop = 0;
    while (timer.elapsed_ms() < duration && cost) {
        loop++;
        if (!(loop & 0xFFFFF)) dump(loop, cost);
        if (rnd.next_int(2)) {
            int idx = rnd.next_int(cands.size());
            auto [y, x] = cands[idx];
            int pvalue = grid[y][x];
            int nvalue;
            do {
                nvalue = rnd.next_int(Q);
            } while (pvalue == nvalue);
            int diff = change(idx, nvalue);
            double temp = 0.2;
            double prob = exp(-diff / temp);
            if (rnd.next_double() > prob) change(idx, pvalue);
        }
        else {
            int idx1, idx2;
            do {
                idx1 = rnd.next_int(cands.size());
                idx2 = rnd.next_int(cands.size());
            } while (idx1 == idx2);
            auto [y1, x1] = cands[idx1];
            auto [y2, x2] = cands[idx2];
            if (grid[y1][x1] == grid[y2][x2]) continue;
            int diff = swap(idx1, idx2);
            double temp = 0.2;
            double prob = exp(-diff / temp);
            if (rnd.next_double() > prob) swap(idx1, idx2);
        }
    }

    if (cost) {
        dump(cost);
        return nullptr;
    }
    return std::make_shared<EncodedGrid>(D, align_parity, grid, displacements, codes, parities);
}

unsigned popcount(unsigned bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}

// 6 2 5
// 3 0 1
// 7 4 8
constexpr int dy[] = { 0, 0, -1, 0, 1, -1, -1, 1, 1 };
constexpr int dx[] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };

void calc_error() {
    // 標準偏差 S の場合に、温度 t のセルを温度 t-d 以下 or t+d 以上と判定する確率
    int S = 25;
    std::mt19937_64 engine(0);
    std::normal_distribution<> dist_norm(0, S);

    int t1 = 500, d = 45, nsample = 1000000, ntrial = 3;
    std::vector<int> hist(1001);
    for (int i = 0; i < nsample; i++) {
        double xsum = 0.0;
        for (int j = 0; j < ntrial; j++) {
            int x = t1 + (int)round(dist_norm(engine));
            assert(0 <= x && x <= 1000);
            xsum += x;
        }
        hist[(int)round(xsum / ntrial)]++;
    }

    int wrong = nsample;
    for (int x = t1 - d + 1; x <= t1 + d - 1; x++) wrong -= hist[x];
    dump(wrong, nsample, double(wrong) / nsample);
}

struct Sampler {

    const int mu;
    const int sigma;
    std::mt19937_64 engine;
    std::normal_distribution<> rng;

    Sampler(int mu_, int sigma_, int seed = 0)
        : mu(mu_), sigma(sigma_), engine(seed), rng(mu, sigma) {}

    int sample() {
        return std::clamp((int)round(rng(engine)), 0, 1000);
    }

    std::vector<int> sample(int n) {
        std::vector<int> res;
        for (int i = 0; i < n; i++) {
            res.push_back(sample());
        }
        return res;
    }

};

std::vector<int> create_hist(Sampler& sampler, int num_iter) {
    std::vector<int> hist(1001);
    for (int n = 0; n < num_iter; n++) {
        hist[sampler.sample()]++;
    }
    return hist;
}

std::vector<double> create_pdf(Sampler& sampler, int num_iter) {
    auto hist = create_hist(sampler, num_iter);
    std::vector<double> pdf(hist.size());
    for (int i = 0; i < (int)pdf.size(); i++) {
        pdf[i] = (double)hist[i] / num_iter;
    }
    return pdf;
}

cv::Mat_<cv::Vec3b> create_pdf_img(const std::vector<double>& pdf) {
    const int N = pdf.size();
    cv::Mat_<cv::Vec3b> img(N, N, cv::Vec3b(255, 255, 255));
    for (int col = 0; col < N; col++) {
        int y = (int)round(pdf[col] * (N - 1));
        int iy = (N - 1) - y;
        img.at<cv::Vec3b>(iy, col) = cv::Vec3b(0, 0, 0);
    }
    return img;
}

double compute_log_likelihood(const std::vector<double>& pdf, const std::vector<int>& data) {
    double log_likelihood = 0.0;
    for (int x : data) {
        log_likelihood += log(pdf[x] + 1e-10);
    }
    return log_likelihood;
}

double compute_mean(const std::vector<int>& data) {
    return double(std::accumulate(data.begin(), data.end(), 0)) / data.size();
}

double compute_expectation(const std::vector<double>& pdf) {
    double e = 0.0;
    for (int x = 0; x <= 1000; x++) {
        e += x * pdf[x];
    }
    return e;
}

double compute_variance(const std::vector<double>& pdf) {
    double e = compute_expectation(pdf);
    double var = 0.0;
    for (int x = 0; x <= 1000; x++) {
        var += pdf[x] * (x - e) * (x - e);
    }
    return var;
}

void experiment1() {

    // 平均値を用いた分布の推定の方が結果がよくなる…？

    Sampler sampler1(0, 100), sampler2(100, 100), sampler3(200, 100);

    auto pdf1 = create_pdf(sampler1, 1000000);
    auto pdf2 = create_pdf(sampler2, 1000000);
    auto pdf3 = create_pdf(sampler3, 1000000);

    double e1 = compute_expectation(pdf1);
    double e2 = compute_expectation(pdf2);
    double e3 = compute_expectation(pdf3);
    dump(e1, e2, e3);

    for (int num_sample = 1; num_sample <= 50; num_sample++) {
        int num_trial = 100000;
        int correct = 0, correct_mean = 0, correct_mean2 = 0;
        for (int i = 0; i < num_trial; i++) {
            auto data2 = sampler2.sample(num_sample);

            auto ll1 = compute_log_likelihood(pdf1, data2);
            auto ll2 = compute_log_likelihood(pdf2, data2);
            auto ll3 = compute_log_likelihood(pdf3, data2);
            correct += ll2 > ll1 && ll2 > ll3;

            double mean = compute_mean(data2);
            double diff1 = abs(0 - mean), diff2 = abs(100 - mean), diff3 = abs(200 - mean);
            correct_mean += diff2 < diff1 && diff2 < diff3;

            double xdiff1 = abs(e1 - mean), xdiff2 = abs(e2 - mean), xdiff3 = abs(e3 - mean);
            correct_mean2 += xdiff2 < xdiff1 && xdiff2 < xdiff3;
        }
        double acc = (double)correct / num_trial;
        double acc_mean = (double)correct_mean / num_trial;
        double acc_mean2 = (double)correct_mean2 / num_trial;
        dump(num_sample, acc, acc_mean, acc_mean2, pow(acc, 7));
    }

}

void central_limit_theorem() {

    int S = 900;

    Sampler sampler(0, S);
    auto pdf = create_pdf(sampler, 1000000);
    auto E = compute_expectation(pdf);
    auto V = compute_variance(pdf);
    dump(E, V, sqrt(V));

    int num_sample = 12;
    int num_trial = 1000000;
    int hist[1001] = {};
    {
        int ctr = 0;
        double sum = 0;
        double sqsum = 0;
        for (int t = 0; t < num_trial; t++) {
            auto data = sampler.sample(num_sample);
            double mean = compute_mean(data);
            ctr++;
            sum += mean;
            sqsum += mean * mean;
            hist[(int)round(mean)]++;
        }
        double mean = sum / ctr;
        double var = sqsum / ctr - mean * mean;
        dump(mean, var, V / num_sample);
    }

    int correct = std::accumulate(hist, hist + 501, 0);
    dump(correct, num_trial, (double)correct / num_trial, pow((double)correct / num_trial, 7));

    dump(hist);

}

void error_correction(int N, int D, double E) {
    
    // D bit からなる整数 N 個
    // 各 bit が確率 E で flip する

    assert(N <= (1 << D));

    std::mt19937_64 engine(4);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    std::vector<int> A(N);
    std::iota(A.begin(), A.end(), 0);
    std::shuffle(A.begin(), A.end(), engine);

    auto generate = [&](int n) {
        int m = A[n];
        for (int b = 0; b < D; b++) {
            if (dist(engine) < E) {
                m ^= (1 << b);
            }
        }
        return m;
    };

    auto generate_all = [&]() {
        std::vector<int> B(N);
        for (int n = 0; n < N; n++) B[n] = generate(n);
        return B;
    };

    std::vector<std::vector<int>> data;
    data.push_back(generate_all());

    dump(A);
    dump(data.back());

}

void parity_test() {
    for (int Q = 2; Q <= 10; Q++) {
        for (int D = 2; D <= 9; D++) {
            int N = (int)pow(Q, D);
            if (N > 10000) continue;
            int odd = 0, even = 0;
            for (int n = 0; n < N; n++) {
                int m = n;
                int sum = 0;
                for (int d = 0; d < D; d++) {
                    sum += m % Q;
                    m /= Q;
                }
                ((sum & 1) ? odd : even)++;
            }
            dump(Q, D, N, odd, even);
        }
    }
}


void temperature_annealing() {

    Timer timer;
    Xorshift rnd;

    const int L = 50;
    const int N = 60;
    auto pos = choose_positions(L, N);
    auto fixed = make_vector(false, L, L);
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        for (int d = 0; d < 8; d++) {
            int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
            fixed[ny][nx] = true;
        }
    }
    
    auto temperature = make_vector(0, L, L);

    int fixed_count = 0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            if (fixed[y][x]) {
                fixed_count++;
                temperature[y][x] = rnd.next_int(2) ? 0 : 1000;
            }
        }
    }

    auto calc_point_cost = [&](int i, int j, int value) {
        int u = temperature[i == 0 ? L - 1 : i - 1][j], d = temperature[i == L - 1 ? 0 : i + 1][j];
        int l = temperature[i][j == 0 ? L - 1 : j - 1], r = temperature[i][j == L - 1 ? 0 : j + 1];
        return (value - u) * (value - u) + (value - d) * (value - d) + (value - l) * (value - l) + (value - r) * (value - r);
    };

    int64_t cost = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            cost += calc_point_cost(i, j, temperature[i][j]);
        }
    }
    cost /= 2;
    dump(cost);

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 3500;
    while ((now_time = timer.elapsed_ms()) < end_time) {
        int i = rnd.next_int(L), j = rnd.next_int(L);
        if (fixed[i][j]) continue;
        loop++;
        int old_value = temperature[i][j];
        int new_value = old_value + (rnd.next_int(2) ? 1 : -1);
        new_value = std::clamp(new_value, 0, 1000);
        int diff = calc_point_cost(i, j, new_value) - calc_point_cost(i, j, old_value);
        double temp = get_temp(1.0, 0.0, now_time - start_time, end_time - start_time);
        double prob = exp(-diff / temp);
        if (rnd.next_double() < prob) {
            //if (diff < 0) {
            temperature[i][j] = new_value;
            cost += diff;
        }
        if (!(loop & 0x7FFFFF)) {
            dump(loop, cost);
        }
    }
    dump(loop, cost);

}

void temperature_fast() {

    constexpr int dy[] = { 0, 0, -1, 0, 1, -1, -1, 1, 1 };
    constexpr int dx[] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
    constexpr int K = 1;

    Timer timer;
    Xorshift rnd;

    const int L = 50;
    const int N = 60;
    auto pos = choose_positions(L, N);
    auto fixed = make_vector(false, L, L);
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        for (int d = 0; d < 8; d++) {
            int ny = (y + dy[d] * K + L) % L, nx = (x + dx[d] * K + L) % L;
            fixed[ny][nx] = true;
        }
    }

    auto temperature = make_vector(0, L, L);

    int fixed_count = 0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            if (fixed[y][x]) {
                fixed_count++;
                temperature[y][x] = rnd.next_int(2) ? 0 : 1000;
            }
        }
    }

    dump(fixed_count);

    auto calc_point_cost = [&](int i, int j, int value) {
        int u = temperature[i == 0 ? L - 1 : i - 1][j], d = temperature[i == L - 1 ? 0 : i + 1][j];
        int l = temperature[i][j == 0 ? L - 1 : j - 1], r = temperature[i][j == L - 1 ? 0 : j + 1];
        return (value - u) * (value - u) + (value - d) * (value - d) + (value - l) * (value - l) + (value - r) * (value - r);
    };

    auto adjacent_mean = [&](int i, int j) {
        int u = temperature[i == 0 ? L - 1 : i - 1][j], d = temperature[i == L - 1 ? 0 : i + 1][j];
        int l = temperature[i][j == 0 ? L - 1 : j - 1], r = temperature[i][j == L - 1 ? 0 : j + 1];
        return (int)round((r + u + l + d) / 4.0);
    };

    auto calc_cost = [&]() {
        int64_t cost = 0;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                cost += calc_point_cost(i, j, temperature[i][j]);
            }
        }
        return cost / 2;
    };

    int64_t cost = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            cost += calc_point_cost(i, j, temperature[i][j]);
        }
    }
    cost /= 2;
    dump(cost);
    dump(calc_cost());

    dump(timer.elapsed_ms());
    for (int trial = 0; trial < 150; trial++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                if (fixed[i][j]) continue;
                temperature[i][j] = adjacent_mean(i, j);
            }
        }
    }
    dump(timer.elapsed_ms());

    dump(calc_cost());

}



void temperature_fast_double(int L, int N, std::vector<Pos>& pos, std::vector<Pos>& displacements) {

    Timer timer;
    Xorshift rnd;

    std::vector<int> bits(1 << (int)(displacements.size() - 1));
    std::iota(bits.begin(), bits.end(), 0);
    for (int& x : bits) x *= 2;
    dump(bits);

    std::sort(bits.begin(), bits.end(), [](int a, int b) {
        return popcount(a) < popcount(b);
        });

    dump(bits);

    int fixed_count = 0;
    auto fixed = make_vector(false, L, L);
    auto temperature = make_vector(0.0, L, L);

    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        int bit = bits[i];
        for (int b = 0; b < displacements.size(); b++) {
            int dy = displacements[b].y, dx = displacements[b].x;
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            fixed_count += !fixed[ny][nx];
            fixed[ny][nx] = true;
            temperature[y][x] = (bit >> b & 1) * 1000.0;
        }
    }

    dump(fixed_count);
    for (const auto& v : temperature) std::cerr << v << '\n';

    auto calc_point_cost = [&](int i, int j, int value) {
        auto u = temperature[i == 0 ? L - 1 : i - 1][j], d = temperature[i == L - 1 ? 0 : i + 1][j];
        auto l = temperature[i][j == 0 ? L - 1 : j - 1], r = temperature[i][j == L - 1 ? 0 : j + 1];
        return (value - u) * (value - u) + (value - d) * (value - d) + (value - l) * (value - l) + (value - r) * (value - r);
    };

    auto adjacent_mean = [&](int i, int j) {
        auto u = temperature[i == 0 ? L - 1 : i - 1][j], d = temperature[i == L - 1 ? 0 : i + 1][j];
        auto l = temperature[i][j == 0 ? L - 1 : j - 1], r = temperature[i][j == L - 1 ? 0 : j + 1];
        return (r + u + l + d) / 4.0;
    };

    auto calc_cost = [&]() {
        auto cost = 0.0;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                cost += calc_point_cost(i, j, temperature[i][j]);
            }
        }
        return cost / 2;
    };

    auto cost = calc_cost();
    dump(cost);

    dump(timer.elapsed_ms());
    for (int trial = 0; trial < 300; trial++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                if (fixed[i][j]) continue;
                temperature[i][j] = adjacent_mean(i, j);
            }
        }
        auto ncost = calc_cost();
        //dump(trial, ncost - cost);
        cost = ncost;
    }
    dump(timer.elapsed_ms());
    dump((int64_t)cost);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            temperature[i][j] = round(temperature[i][j]);
        }
    }

    dump((int64_t)calc_cost());

}

std::vector<Pos> choose_displacements(int L, const std::vector<Pos>& pos, int num_neighbors, int width) {
    const int N = pos.size();
    auto fixed = make_vector(false, L, L);
    int fixed_count = 0;

    std::vector<Pos> displacements;

    auto set = [&](int dy, int dx) {
        for (auto [y, x] : pos) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            fixed_count += !fixed[ny][nx];
            fixed[ny][nx] = true;
        }
        displacements.emplace_back(dy, dx);
    };

    auto count = [&](int dy, int dx) {
        int res = 0;
        for (auto [y, x] : pos) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            res += fixed[ny][nx];
        }
        return res;
    };

    if (true) {
        set(0, 0);
        dump(fixed_count);

        for (int i = 1; i < num_neighbors; i++) {
            int max_count = -1, max_dy = -1, max_dx = -1;
            for (int dy = -width; dy <= width; dy++) {
                for (int dx = -width; dx <= width; dx++) {
                    if (std::count(displacements.begin(), displacements.end(), Pos(dy, dx))) continue;
                    int c = count(dy, dx);
                    if (chmax(max_count, c)) {
                        max_dy = dy;
                        max_dx = dx;
                    }
                }
            }
            set(max_dy, max_dx);
            dump(fixed_count);
        }

        dump(displacements);
    }
    else {
        for (int d = 0; d < 8; d++) {
            set(dy[d], dx[d]);
        }
        dump(fixed_count);
    }

    return displacements;
}

struct GridSearcherDFS {

    const int L;
    const int N;
    const int Q;
    const int D;
    const std::vector<Pos> positions;
    const std::vector<Pos> displacements;

    std::vector<int> position_order;
    std::vector<int> code_order;
    std::vector<bool> code_used;

    std::vector<std::vector<int>> grid;
    std::vector<std::vector<int>> grid_ctr;

    bool found;

    GridSearcherDFS(int L_, int Q_, const std::vector<Pos>& positions_, const std::vector<Pos>& displacements_)
        : L(L_), N(positions_.size()), Q(Q_), D(displacements_.size()), positions(positions_), displacements(displacements_) {}

    int code_sum(int code) const {
        int res = 0;
        for (int d = 0; d < D; d++) {
            res += code % Q;
            code /= Q;
        }
        return res;
    }

    std::vector<int> to_array(int code) const {
        std::vector<int> arr;
        for (int d = 0; d < D; d++) {
            arr.push_back(code % Q);
            code /= Q;
        }
        return arr;
    }

    bool can_embed(int position_id, int code) const {
        const auto& [y, x] = positions[position_id];
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            int val = code % Q;
            code /= Q;
            if (grid[ny][nx] != -1 && grid[ny][nx] != val) return false;
        }
        return true;
    }

    void embed(int position_id, int code) {
        code_used[code] = true;
        const auto& [y, x] = positions[position_id];
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            int val = code % Q;
            code /= Q;
            grid[ny][nx] = val;
            grid_ctr[ny][nx]++;
        }
    }

    void undo(int position_id, int code) {
        code_used[code] = false;
        const auto& [y, x] = positions[position_id];
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            int val = code % Q;
            code /= Q;
            grid_ctr[ny][nx]--;
            if (!grid_ctr[ny][nx]) grid[ny][nx] = -1;
        }
    }

    void dfs(int idx) {
        if (idx == N) {
            found = true;
            return;
        }
        int position_id = position_order[idx];
        //dump(idx, position_id);
        for (int code : code_order) {
            if (code_used[code]) continue;
            if (can_embed(position_id, code)) {
                embed(position_id, code);
                dfs(idx + 1);
                if (found) return;
                undo(position_id, code);
            }
        }
    }

    void run() {
        // 周囲への影響度の高い位置から選ぶ
        auto overlap = make_vector(0, L, L);
        for (const auto& [y, x] : positions) {
            for (const auto& [dy, dx] : displacements) {
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                overlap[ny][nx]++;
            }
        }

        std::vector<int> position_priority(N);
        for (int i = 0; i < N; i++) {
            auto [y, x] = positions[i];
            for (const auto& [dy, dx] : displacements) {
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                position_priority[i] += overlap[ny][nx];
            }
        }

        position_order.resize(N);
        std::iota(position_order.begin(), position_order.end(), 0);
        std::sort(position_order.begin(), position_order.end(), [&](int i, int j) {
            return position_priority[i] > position_priority[j];
            });

        code_order.clear();
        int code_size = (int)pow(Q, D);
        {
            for (int code = 0; code < code_size; code++) {
                if (code_sum(code) % 2 == 0) {
                    code_order.push_back(code);
                }
            }
        }
        //code_order.resize((int)pow(Q, D));
        //std::iota(code_order.begin(), code_order.end(), 0);
        std::sort(code_order.begin(), code_order.end(), [this](int i, int j) {
            auto pi = code_sum(i), pj = code_sum(j);
            return pi == pj ? i < j : pi < pj;
            });

        code_used.resize(code_size);

        grid = make_vector(-1, L, L);
        grid_ctr = make_vector(0, L, L);

        found = false;

        dfs(0);

        for (const auto& v : grid) std::cerr << v << '\n';

        {
            std::set<int> code_set;
            for (int position_id : position_order) {
                auto [y, x] = positions[position_id];
                int code = 0;
                for (const auto& [dy, dx] : displacements) {
                    int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                    assert(grid[ny][nx] != -1);
                    code *= Q;
                    code += grid[ny][nx];
                }
                dump(position_id, code);
                assert(!code_set.count(code));
                code_set.insert(code);
            }
        }
    }

};

struct GridSearcherCodeBaseAnnealing {

    const int L;
    const int N;
    const int Q;
    const int D;
    const std::vector<Pos> positions;
    const std::vector<Pos> displacements;

    Xorshift rnd;

    std::vector<int> codes;
    std::vector<std::vector<std::vector<int>>> grid_ctr;

    int cost;

    GridSearcherCodeBaseAnnealing(int L_, int Q_, const std::vector<Pos>& positions_, const std::vector<Pos>& displacements_)
        : L(L_), N(positions_.size()), Q(Q_), D(displacements_.size()), positions(positions_), displacements(displacements_) {}

    int calc_grid_cost(int y, int x) const {
        const auto& g = grid_ctr[y][x];
        return std::accumulate(g.begin(), g.end(), 0) - *std::max_element(g.begin(), g.end());
    }

    void pop(int idx) {
        int code = codes[idx];
        codes[idx] = -1;
        const auto& [y, x] = positions[idx];
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            cost -= calc_grid_cost(ny, nx);
            grid_ctr[ny][nx][code % Q]--;
            cost += calc_grid_cost(ny, nx);
            code /= Q;
        }
    }

    void push(int idx, int code) {
        assert(codes[idx] == -1);
        codes[idx] = code;
        const auto& [y, x] = positions[idx];
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            cost -= calc_grid_cost(ny, nx);
            grid_ctr[ny][nx][code % Q]++;
            cost += calc_grid_cost(ny, nx);
            code /= Q;
        }
    }

    int swap(int idx1, int idx2) {
        assert(idx1 != idx2);
        if (idx1 > idx2) std::swap(idx1, idx2);
        int pcost = cost;
        int code1 = codes[idx1], code2 = codes[idx2];
        if (N <= idx2) {
            pop(idx1);
            push(idx1, code2);
            codes[idx2] = code1;
        }
        else {
            pop(idx1);
            pop(idx2);
            push(idx1, code2);
            push(idx2, code1);
        }
        return cost - pcost;
    }

    void reset() {
        shuffle_vector(codes, rnd);
        cost = 0;
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                grid_ctr[y][x].assign(Q, 0);
            }
        }
        for (int i = 0; i < N; i++) {
            int code = codes[i];
            const auto [y, x] = positions[i];
            for (const auto [dy, dx] : displacements) {
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                grid_ctr[ny][nx][code % Q]++;
                code /= Q;
            }
        }
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                cost += calc_grid_cost(y, x);
            }
        }
    }

    int choose_invalid_idx(Xorshift& rnd) const {
        std::vector<int> cands;
        for (int i = 0; i < N; i++) {
            auto [y, x] = positions[i];
            bool invalid = false;
            for (const auto [dy, dx] : displacements) {
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                if (calc_grid_cost(ny, nx)) {
                    invalid = true;
                    break;
                }
            }
            if (invalid) cands.push_back(i);
        }
        return cands[rnd.next_int(cands.size())];
    }

    void run() {

        const int M = (int)pow(Q, D);
        dump(M);
        codes.resize(M);
        std::iota(codes.begin(), codes.end(), 0);

        grid_ctr = make_vector(std::vector<int>(Q), L, L);

        for (int i = 0; i < N; i++) {
            int code = codes[i];
            const auto [y, x] = positions[i];
            for (const auto [dy, dx] : displacements) {
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                grid_ctr[ny][nx][code % Q]++;
                code /= Q;
            }
        }

        cost = 0;
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                cost += calc_grid_cost(y, x);
            }
        }

        
        int loop = 0;
        int min_cost = cost;
        int no_change = 0;
        while (cost) {
            loop++;
            int idx1 = rnd.next_int(N), idx2;
            do {
                idx2 = rnd.next_int(M);
            } while (idx1 == idx2);
            int diff = swap(idx1, idx2);
            if (no_change >= 100000) {
                no_change = 0;
                //reset();
                min_cost = cost;
                continue;
            }
            if (diff > 0) swap(idx1, idx2);
            if (chmin(min_cost, cost)) {
                no_change = 0;
            }
            else {
                no_change++;
            }
            if (!(loop & 0xFFFF)) dump(loop, cost);
        }
        dump(cost);

    }

};

struct GridAnnealer {

    const int L;
    const int N;
    const std::vector<Pos>& pos;
    const std::vector<Pos>& displacements;

    const int bit_rate;
    const int num_neighbors;
    const bool align_parity;

    Xorshift rnd;

    std::vector<int> powers;

    Grid<int> grid;
    Grid<std::vector<std::pair<int, int>>> grid_to_id;
    std::vector<Pos> cands;
    std::vector<int> codes;
    std::vector<bool> parities;
    std::vector<int> code_ctr;

    int cost;

    GridAnnealer(
        int L_, const std::vector<Pos>& pos_, const std::vector<Pos>& displacements_,
        int bit_rate_, bool align_parity_
    ) :
        L(L_), N(pos_.size()), pos(pos_), displacements(displacements_),
        bit_rate(bit_rate_), num_neighbors(displacements.size()), align_parity(align_parity_)
    {
        assert(N * (align_parity ? 2 : 1) <= (int)pow(bit_rate, num_neighbors));
        initialize();
    }

    void initialize() {

        powers.assign(1, 1);
        while (powers.size() < num_neighbors) {
            powers.push_back(powers.back() * bit_rate);
        }

        grid = make_vector(0, L, L);
        grid_to_id = make_vector(std::vector<std::pair<int, int>>(), L, L);
        for (int i = 0; i < N; i++) {
            auto [y, x] = pos[i];
            for (int d = 0; d < num_neighbors; d++) {
                auto [dy, dx] = displacements[d];
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                grid_to_id[ny][nx].emplace_back(i, d);
            }
        }

        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                if (!grid_to_id[y][x].empty()) {
                    cands.emplace_back(y, x);
                    //grid[y][x] = rnd.next_int(bit_rate);
                }
            }
        }

        codes.assign(N, 0);
        parities.assign(N, false);
        code_ctr.assign((int)pow(bit_rate, num_neighbors), 0);
        for (int i = 0; i < N; i++) {
            auto [y, x] = pos[i];
            int code = 0;
            for (int d = 0; d < num_neighbors; d++) {
                auto [dy, dx] = displacements[d];
                int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
                code += grid[ny][nx] * powers[d];
                parities[i] = parities[i] ^ (grid[ny][nx] & 1);
            }
            codes[i] = code;
            code_ctr[code]++;
        }

        cost = 0;
        for (int c : code_ctr) {
            int x = std::max(0, c - 1);
            cost += x * x;
        }
        if (align_parity) {
            for (auto p : parities) {
                cost += p; // odd: penalty
            }
        }

    }

    void pop(int value) {
        assert(code_ctr[value]);
        int px = std::max(0, code_ctr[value] - 1);
        cost -= px * px;
        code_ctr[value]--;
        int nx = std::max(0, code_ctr[value] - 1);
        cost += nx * nx;
    }

    void push(int value) {
        int px = std::max(0, code_ctr[value] - 1);
        cost -= px * px;
        code_ctr[value]++;
        int nx = std::max(0, code_ctr[value] - 1);
        cost += nx * nx;
    }

    int change(int idx, int value) {
        int pcost = cost;
        assert(idx < cands.size());
        auto [y, x] = cands[idx];
        assert(grid[y][x] != value);
        int diff = value - grid[y][x];
        bool parity_changed = align_parity & (grid[y][x] ^ value) & 1;
        for (auto [i, d] : grid_to_id[y][x]) {
            pop(codes[i]);
            codes[i] += diff * powers[d];
            push(codes[i]);
            cost += parity_changed ? (parities[i] ? -1 : 1) : 0;
            parities[i] = parity_changed ? !parities[i] : parities[i];
        }
        grid[y][x] = value;
        return cost - pcost;
    }

    int swap(int idx1, int idx2) {
        assert(idx1 != idx2);
        auto [y1, x1] = cands[idx1];
        auto [y2, x2] = cands[idx2];
        int v1 = grid[y1][x1], v2 = grid[y2][x2];
        assert(v1 != v2);
        int diff = 0;
        diff += change(idx1, v2);
        diff += change(idx2, v1);
        return diff;
    }

    bool run(double duration) {
        Timer timer;
        int loop = 0;
        while (timer.elapsed_ms() < duration) {
            loop++;
            //if (!(loop & 0xFFFFF)) dump(loop, cost);
            if (rnd.next_int(2)) {
                int idx = rnd.next_int(cands.size());
                auto [y, x] = cands[idx];
                int pvalue = grid[y][x];
                int nvalue;
                do {
                    nvalue = rnd.next_int(bit_rate);
                } while (pvalue == nvalue);
                int diff = change(idx, nvalue);
                double temp = 0.2;
                double prob = exp(-diff / temp);
                if (rnd.next_double() > prob) change(idx, pvalue);
            }
            else {
                int idx1, idx2;
                do {
                    idx1 = rnd.next_int(cands.size());
                    idx2 = rnd.next_int(cands.size());
                } while (idx1 == idx2);
                auto [y1, x1] = cands[idx1];
                auto [y2, x2] = cands[idx2];
                if (grid[y1][x1] == grid[y2][x2]) continue;
                int diff = swap(idx1, idx2);
                double temp = 0.2;
                double prob = exp(-diff / temp);
                if (rnd.next_double() > prob) swap(idx1, idx2);
            }
            if (!cost) break;
        }
        //dump(cost, timer.elapsed_ms(), loop, bit_rate, num_neighbors);
        return cost == 0;
    }

};

int64_t compute_placement_cost(
    Grid<int> grid,
    const std::vector<Pos>& pos,
    const std::vector<Pos>& displacements,
    int intercept,
    int slope
) {
    Timer timer;
    const int L = grid.size();
    const int N = pos.size();
    const int D = displacements.size();
    auto fixed = make_vector(false, L, L);
    for (const auto& [y, x] : pos) {
        for (const auto& [dy, dx] : displacements) {
            int ny = (y + dy + L) % L, nx = (x + dx + L) % L;
            fixed[ny][nx] = true;
        }
    }
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            grid[y][x] = intercept + grid[y][x] * slope;
        }
    }
    auto adjacent_mean = [&](int y, int x) {
        int u = grid[y == 0 ? L - 1 : y - 1][x], d = grid[y == L - 1 ? 0 : y + 1][x];
        int l = grid[y][x == 0 ? L - 1 : x - 1], r = grid[y][x == L - 1 ? 0 : x + 1];
        return (int)round((r + u + l + d) / 4.0);
    };
    auto compute_cost = [&]() {
        int64_t cost = 0;
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                int v = grid[y][x], r = grid[y][x == L - 1 ? 0 : x + 1], d = grid[y == L - 1 ? 0 : y + 1][x];
                cost += (v - r) * (v - r) + (v - d) * (v - d);
            }
        }
        return cost;
    };
    auto cost = compute_cost();
    while (true) {
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                if (fixed[y][x]) continue;
                grid[y][x] = adjacent_mean(y, x);
            }
        }
        auto ncost = compute_cost();
        if (cost == ncost) break;
        else cost = ncost;
    }
    return cost;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    std::string input_file("../../tools_win/in/0006.txt");
    Input input(input_file);

    const int L = input.L;
    auto pos = input.landing_pos;
    auto displacements = choose_displacements(L, pos, 8, 14);
    //std::vector<Pos> displacements;
    //for (int d = 0; d < 8; d++) displacements.emplace_back(dy[d] * 4, dx[d] * 4);

    int intercept = 143, slope = 858 - 143;

    GridAnnealer ga(L, pos, displacements, 2, true);
    int64_t min_cost = INT64_MAX;
    for (int t = 0; t < 100; t++) {
        ga.initialize();
        if (!ga.run(500.0)) continue;
        auto cost = compute_placement_cost(ga.grid, ga.pos, ga.displacements, intercept, slope);
        if (chmin(min_cost, cost)) {
            dump(t, cost);
        }
    }

    return 0;
}