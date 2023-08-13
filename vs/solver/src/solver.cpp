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
#define ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#ifdef ENABLE_DUMP
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<']'<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
#else
#define dump(...) void(0);
#endif
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
    Pos(int y = 0, int x = 0) : y(y), x(x) {}
};
std::istream& operator>>(std::istream& in, Pos& pos) {
    in >> pos.y >> pos.x;
    return in;
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

struct Metrics {
    int64_t score;
    int wrong;
    int64_t placement_cost;
    int64_t measurement_cost;
    int measurement_count;
    std::string stringify() const {
        return format(
            "score=%10lld, wrong=%3d, placement_cost=%10lld, measurement_cost=%10lld, measurement_count=%5d",
            score, wrong, placement_cost, measurement_cost, measurement_count
            );
    }
};

template<typename T>
using Grid = std::vector<std::vector<T>>;

struct Judge;
using JudgePtr = std::shared_ptr<Judge>;
struct Judge {

    virtual const Input& get_input() const = 0;
    virtual void set_temperature(const Grid<int>& temperature) = 0;
    virtual int measure(int i, int y, int x) = 0;
    virtual double measure(int i, int y, int x, int iter) = 0;
    virtual void answer(const std::vector<int>& estimate) = 0;
    virtual std::optional<Metrics> get_metrics() const = 0;

};

struct ServerJudge;
using ServerJudgePtr = std::shared_ptr<ServerJudge>;
struct ServerJudge : Judge {

    const Input input;

    ServerJudge() : input(std::cin) {}

    const Input& get_input() const { return input; }

    void set_temperature(const Grid<int>& temperature) override {
        for (const auto& row : temperature) {
            for (int i = 0; i < row.size(); i++) {
                std::cout << row[i] << (i == row.size() - 1 ? "\n" : " ");
            }
        }
        std::cout.flush();
    }

    int measure(int i, int y, int x) override {
        std::cout << i << " " << y << " " << x << std::endl; // endl does flush
        int v;
        std::cin >> v;
        if (v == -1) {
            std::cerr << "something went wrong. i=" << i << " y=" << y << " x=" << x << std::endl;
            exit(1);
        }
        return v;
    }

    double measure(int i, int y, int x, int iter) override {
        double sum = 0.0;
        for (int trial = 0; trial < iter; trial++) {
            sum += measure(i, y, x);
        }
        return sum / iter;
    }

    void answer(const std::vector<int>& estimate) override {
        std::cout << "-1 -1 -1" << std::endl;
        for (int e : estimate) {
            std::cout << e << std::endl;
        }
    }

    std::optional<Metrics> get_metrics() const override {
        return std::nullopt;
    }

};

struct FileJudge;
using FileJudgePtr = std::shared_ptr<FileJudge>;
struct FileJudge : Judge {

    const std::string input_file;
    const std::string output_file;

    const Input input;
    std::vector<int> A;
    std::vector<int> F;

    std::vector<std::tuple<int, int, int>> measurements;

    int turn = 0;
    Grid<int> temperature;
    std::vector<int> estimate;

    int64_t placement_cost = 0;
    int64_t measurement_cost = 0;

    int wrong = 0;
    int64_t score = 0;

    FileJudge(const std::string& input_file_, const std::string& output_file_)
        : input_file(input_file_), output_file(output_file_), input(input_file) {
        std::ifstream in(input_file);
        { Input tmp(in); } // dry load
        A.resize(input.N);
        F.resize(10000);
        in >> A >> F;
        in.close();
    }

    const Input& get_input() const { return input; }

    void set_temperature(const Grid<int>& temperature_) override {
        temperature = temperature_;
        const int L = input.L;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int s = temperature[i][j], t = temperature[i][(j + 1) % L], u = temperature[(i + 1) % L][j];
                placement_cost += (s - t) * (s - t) + (s - u) * (s - u);
            }
        }
    }

    int measure(int i, int y, int x) override {
        if (turn >= 10000) {
            std::cerr << "something went wrong. i=" << i << " y=" << y << " x=" << x << std::endl;
            exit(1);
        }
        measurements.emplace_back(i, y, x);
        measurement_cost += 100 * (10 + abs(y) + abs(x));
        int ty = (input.landing_pos[A[i]].y + input.L + y) % input.L;
        int tx = (input.landing_pos[A[i]].x + input.L + x) % input.L;
        return std::clamp(temperature[ty][tx] + F[turn++], 0, 1000);
    }

    double measure(int i, int y, int x, int iter) override {
        double sum = 0.0;
        for (int trial = 0; trial < iter; trial++) {
            sum += measure(i, y, x);
        }
        return sum / iter;
    }

    void answer(const std::vector<int>& estimate_) override {
        estimate = estimate_;
        wrong = 0;
        for (int i = 0; i < input.N; i++) {
            wrong += A[i] != estimate[i];
        }
        score = (int64_t)ceil(1e14 * pow(0.8, wrong) / (1e5 + placement_cost + measurement_cost));
        dump(A);
        dump(estimate);
        // dump(placement_cost, measurement_cost, wrong, score);
    }

    std::optional<Metrics> get_metrics() const override {
        return Metrics{ score, wrong, placement_cost, measurement_cost, turn };
    }

    ~FileJudge() {
        std::ofstream out(output_file);
        for (const auto& row : temperature) {
            for (int i = 0; i < row.size(); i++) {
                out << row[i] << (i == row.size() - 1 ? "\n" : " ");
            }
        }
        for (const auto& [i, y, x] : measurements) {
            out << i << ' ' << y << ' ' << x << '\n';
        }
        out << "-1 -1 -1\n";
        for (int e : estimate) {
            out << e << '\n';
        }
        out.close();
    }

};

struct LocalJudge;
using LocalJudgePtr = std::shared_ptr<LocalJudge>;
struct LocalJudge : Judge {

    std::mt19937_64 engine;
    const Input input;

    std::vector<int> A;
    std::vector<int> F;

    std::vector<std::tuple<int, int, int>> measurements;

    int turn = 0;
    Grid<int> temperature;
    std::vector<int> estimate;

    int64_t placement_cost = 0;
    int64_t measurement_cost = 0;

    int wrong = 0;
    int64_t score = 0;

    LocalJudge(int seed = 0, int L = -1, int N = -1, int S = -1) : engine(seed), input(engine, L, N, S) {
        A.resize(input.N);
        std::iota(A.begin(), A.end(), 0);
        std::shuffle(A.begin(), A.end(), engine);
        std::normal_distribution<> dist_norm(0.0, S);
        F.resize(10000);
        for (int i = 0; i < 10000; i++) {
            F[i] = (int)round(dist_norm(engine));
        }
    }

    const Input& get_input() const { return input; }

    void set_temperature(const Grid<int>& temperature_) override {
        temperature = temperature_;
        const int L = input.L;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int s = temperature[i][j], t = temperature[i][(j + 1) % L], u = temperature[(i + 1) % L][j];
                placement_cost += (s - t) * (s - t) + (s - u) * (s - u);
            }
        }
    }

    int measure(int i, int y, int x) override {
        if (turn >= 10000) {
            std::cerr << "something went wrong. i=" << i << " y=" << y << " x=" << x << std::endl;
            exit(1);
        }
        measurements.emplace_back(i, y, x);
        measurement_cost += 100 * (10 + abs(y) + abs(x));
        int ty = (input.landing_pos[A[i]].y + input.L + y) % input.L;
        int tx = (input.landing_pos[A[i]].x + input.L + x) % input.L;
        return std::clamp(temperature[ty][tx] + F[turn++], 0, 1000);
    }

    double measure(int i, int y, int x, int iter) override {
        double sum = 0.0;
        for (int trial = 0; trial < iter; trial++) {
            sum += measure(i, y, x);
        }
        return sum / iter;
    }

    void answer(const std::vector<int>& estimate_) override {
        estimate = estimate_;
        wrong = 0;
        for (int i = 0; i < input.N; i++) {
            wrong += A[i] != estimate[i];
        }
        score = (int64_t)ceil(1e14 * pow(0.8, wrong) / (1e5 + placement_cost + measurement_cost));
        // dump(placement_cost, measurement_cost, wrong, score);
    }

    std::optional<Metrics> get_metrics() const override {
        return Metrics{ score, wrong, placement_cost, measurement_cost, turn };
    }

};



constexpr int params_opt[31][4] = {
    {-1, -1, -1, -1},
    {10, 500, 7, 1},
    {3, 500, 28, 1},
    {2, 500, 48, 2},
    {3, 500, 50, 4},
    {2, 460, 80, 4},
    {2, 450, 100, 6},
    {2, 442, 115, 7},
    {2, 430, 140, 9},
    {2, 425, 150, 12},
    {2, 418, 165, 15},
    {2, 405, 190, 20},
    {2, 390, 220, 25},
    {2, 375, 250, 25},
    {2, 360, 280, 25},
    {2, 340, 320, 30},
    {2, 315, 370, 30},
    {2, 290, 420, 30},
    {2, 260, 480, 30},
    {2, 235, 530, 30},
    {2, 200, 600, 30},
    {2, 163, 675, 30},
    {2, 125, 750, 30},
    {2, 100, 800, 30},
    {2, 50, 900, 30},
    {2, 25, 950, 30},
    {2, 0, 1000, 30},
    {2, 0, 1000, 30},
    {2, 0, 1000, 30},
    {2, 0, 1000, 30},
    {2, 0, 1000, 30},
};

struct Params;
using ParamsPtr = std::shared_ptr<Params>;
struct Params {
    const int num_quantize;
    const int intercept;
    const int slope;
    const int num_trial;
    Params(int num_quantize_, int intercept_, int slope_, int num_trial_)
        : num_quantize(num_quantize_), intercept(intercept_), slope(slope_), num_trial(num_trial_) {}
};

ParamsPtr load_params(const Input& input) {
    for (int sqrtS = 1; sqrtS <= 30; sqrtS++) {
        if (input.S == sqrtS * sqrtS) {
            assert(params_opt[sqrtS][0] != -1);
            return std::make_shared<Params>(
                params_opt[sqrtS][0],
                params_opt[sqrtS][1],
                params_opt[sqrtS][2],
                params_opt[sqrtS][3]
            );
        }
    }
    assert(false);
    return nullptr;
}

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

struct FindGridResult {
    bool success = false;
    int num_neighbors = -1;
    bool align_parity = false;
    Grid<int> grid;
    std::vector<int> codes;
    std::vector<bool> parities;
};

// 6 2 5
// 3 0 1
// 7 4 8
constexpr int dy[] = { 0, 0, -1, 0, 1, -1, -1, 1, 1 };
constexpr int dx[] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };

FindGridResult find_unique_encoded_grid(
    const std::vector<Pos>& pos,
    int L, // grid size
    int Q, // number of quantization
    int D, // number of neighbors
    bool align_parity,
    double duration
) {
    Timer timer;

    FindGridResult res;

    const int N = pos.size();

    assert(N * (align_parity ? 2 : 1) <= (int)pow(Q, D));

    std::vector<int> powers({ 1 });
    while (powers.size() < D) {
        powers.push_back(powers.back() * Q);
    }

    auto grid = make_vector(0, L, L);
    auto grid_to_id = make_vector(std::vector<std::pair<int, int>>(), L, L);
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        for (int d = 0; d < D; d++) {
            int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
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
            int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
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
        //if (!(loop & 0xFFFFF)) dump(loop, cost);
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
    if (cost) return res;

    res.success = true;
    res.num_neighbors = D;
    res.align_parity = align_parity;
    res.grid = grid;
    res.codes = codes;
    res.parities = parities;

    return res;
}

FindGridResult manage_to_find_unique_encoded_grid(
    const std::vector<Pos>& pos,
    int L, // grid size
    int Q, // number of quantization
    double duration
) {

    const int N = pos.size();

    if (true) { // align parity
        int D = 0, NN = 1;
        while (NN < N * 2) {
            D++;
            NN *= Q;
        }
        int DMAX = std::min(D + 1, 9);
        while (D <= DMAX) {
            dump(L, Q, D, "parity");
            auto res = find_unique_encoded_grid(pos, L, Q, D, true, duration);
            if (res.success) return res;
            D++;
        }
    }

    {
        int D = 0, NN = 1;
        while (NN < N) {
            D++;
            NN *= Q;
        }
        while (true) {
            dump(L, Q, D);
            auto res = find_unique_encoded_grid(pos, L, Q, D, false, duration);
            if (res.success) return res;
            D++;
        }
    }

    return FindGridResult();
}

Grid<int> create_temperature(
    Timer& timer,
    Xorshift& rnd,
    const Input& input,
    const FindGridResult& find_grid_result,
    const Quantizer& quantizer
) {
    const int L = input.L;
    const int N = input.N;
    const int num_neighbors = find_grid_result.num_neighbors;

    auto temperature = find_grid_result.grid;

    auto fixed = make_vector(false, L, L);
    for (int i = 0; i < N; i++) {
        auto [y, x] = input.landing_pos[i];
        for (int d = 0; d < num_neighbors; d++) {
            int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
            fixed[ny][nx] = true;
        }
    }

    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            temperature[y][x] = quantizer.to_value(fixed[y][x] ? temperature[y][x] : 0);
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
        double temp = get_temp(10.0, 0.0, now_time - start_time, end_time - start_time);
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
    return temperature;
}

std::vector<int> predict(
    JudgePtr judge,
    const Input& input,
    const ParamsPtr& params,
    const FindGridResult& find_grid_result,
    const Quantizer& quantizer,
    const Grid<int>& temperature
) {
    
    const int L = input.L;
    const int N = input.N;
    const auto& landing_pos = input.landing_pos;
    const int num_neighbors = find_grid_result.num_neighbors;
    const bool align_parity = find_grid_result.align_parity;

    int num_trial = params->num_trial;
    std::vector<int> estimate(N);

    {
        int K = num_neighbors * N;
        chmin(num_trial, (align_parity ? 9000 : 9500) / K);
    }

    auto check_parity = [&](int i_in, const std::vector<int>& ctrs, const std::vector<int>& sums) {
        bool parity = find_grid_result.parities[i_in];
        int index_sum = 0;
        for (int d = 0; d < num_neighbors; d++) {
            index_sum += quantizer.to_index((double)sums[d] / ctrs[d]);
        }
        return parity == (index_sum & 1);
    };

    auto check_code_match = [&](int i_in, const std::vector<int>& ctrs, const std::vector<int>& sums) {
        auto& codes = find_grid_result.codes;
        int code = 0;
        for (int d = num_neighbors - 1; d >= 0; d--) {
            code *= quantizer.rate;
            code += quantizer.to_index((double)sums[d] / ctrs[d]);
        }
        bool found = std::count(codes.begin(), codes.end(), code);
        return found;
    };

    int additional_trial = 0;
    auto sample = [&](int i_in, std::vector<int>& ctrs, std::vector<int>& sums, bool additional) {
        for (int d = 0; d < num_neighbors; d++) {
            additional_trial += additional;
            ctrs[d]++;
            sums[d] += judge->measure(i_in, dy[d], dx[d]);
        }
    };

    for (int i_in = 0; i_in < N; i_in++) {

        std::vector<int> ctrs(num_neighbors);
        std::vector<int> sums(num_neighbors);

        for (int t = 0; t < num_trial; t++) {
            sample(i_in, ctrs, sums, false);
        }

        if (align_parity) {
            if (!check_parity(i_in, ctrs, sums)) {
                dump("parity check failed", i_in);
            }
            while (!check_parity(i_in, ctrs, sums)) {
                sample(i_in, ctrs, sums, true);
            }
        }

        if (!check_code_match(i_in, ctrs, sums)) {
            dump("code match failed", i_in);
        }
        while (!check_code_match(i_in, ctrs, sums)) {
            sample(i_in, ctrs, sums, true);
        }

        int best_hit = -1, best_id = -1;
        for (int i_out = 0; i_out < N; i_out++) {
            auto [y, x] = landing_pos[i_out];
            int hit = 0;
            for (int d = 0; d < num_neighbors; d++) {
                int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
                hit += quantizer.quantize((double)sums[d] / ctrs[d]) == temperature[ny][nx];
            }
            if (chmax(best_hit, hit)) {
                best_id = i_out;
            }
        }
        dump(i_in, best_hit);
        estimate[i_in] = best_id;
    }
    dump(additional_trial);

    dump(estimate);
    {
        auto e(estimate);
        std::sort(e.begin(), e.end());
        dump(e);
    }
    return estimate;
}

std::optional<Metrics> solve(JudgePtr judge) {

    Timer timer;
    Xorshift rnd;
    
    const auto& input = judge->get_input();
    const auto& params = load_params(input);
    Quantizer quantizer(params->num_quantize, params->intercept, params->slope);

    auto find_grid_result = manage_to_find_unique_encoded_grid(input.landing_pos, input.L, quantizer.rate, 500.0); // max ~2sec
    assert(find_grid_result.success);

    auto temperature = create_temperature(timer, rnd, input, find_grid_result, quantizer);
    judge->set_temperature(temperature);

    auto estimate = predict(judge, input, params, find_grid_result, quantizer, temperature);
    judge->answer(estimate);

    return judge->get_metrics();
}



void batch_execution() {

    int num_seeds = 50;

    // grid_search
    std::vector<Metrics> metrics_list(num_seeds);

    int progress = 0;
    int64_t score_sum = 0;
    int wrong_sum = 0;
    int64_t placement_sum = 0;
    int64_t measurement_sum = 0;
    int count_sum = 0;
    int64_t min_score = INT64_MAX, max_score = INT64_MIN;

#pragma omp parallel for num_threads(10)
    for (int seed = 0; seed < num_seeds; seed++) {
        auto judge = std::make_shared<LocalJudge>(seed, -1, -1, 900);
        auto metrics_opt = solve(judge);
#pragma omp critical(crit_sct)
        if (metrics_opt) {
            auto metrics = *metrics_opt;
            progress++;
            score_sum += metrics.score;
            wrong_sum += metrics.wrong;
            placement_sum += metrics.placement_cost;
            measurement_sum += metrics.measurement_cost;
            count_sum += metrics.measurement_count;
            chmin(min_score, metrics.score);
            chmax(max_score, metrics.score);
            std::cerr << format(
                "\rprogress=%3d/%3d, avg_score=%13.2f, avg_wrong=%5.2f, avg_placement=%13.2f, avg_measurement=%13.2f, avg_count=%7.2f, min=%11lld, max=%11lld",
                progress, num_seeds, (double)score_sum / progress, (double)wrong_sum / progress, (double)placement_sum / progress, (double)measurement_sum / progress, (double)count_sum / progress, min_score, max_score
            );
            metrics_list[seed] = metrics;
        }
    }
    std::cerr << '\n';
}



int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    //batch_execution();
    //exit(0);

    JudgePtr judge;

    if (true) {
        judge = std::make_shared<ServerJudge>();
    }
    else if(true) {
        std::string input_file("../../tools_win/in/0073.txt");
        std::string output_file("../../tools_win/out/0073.txt");
        judge = std::make_shared<FileJudge>(input_file, output_file);
    }
    else {
        judge = std::make_shared<LocalJudge>(0, 30, 80, 100);
    }

    auto metrics_opt = solve(judge);

    if (metrics_opt) {
        auto [score, wrong, placement_cost, measurement_cost, measurement_count] = *metrics_opt;
        dump(score, wrong, placement_cost, measurement_cost, measurement_count);
    }

    return 0;
}