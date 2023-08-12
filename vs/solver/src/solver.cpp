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

constexpr int param_s_to_interval[31] = {
    -1,
    3, 8, 8, 16, 16, 16, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};
constexpr int param_s_to_num_trial[31] = {
    -1,
    3, 8, 32, 64, 128, 256, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};

struct Pos {
    int y, x;
    Pos(int y = 0, int x = 0) : y(y), x(x) {}
};

struct Input {
    int L, N, S;
    std::vector<Pos> landing_pos;
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

struct Judge;
using JudgePtr = std::shared_ptr<Judge>;
struct Judge {

    virtual Input get_input() const = 0;
    virtual void set_temperature(const std::vector<std::vector<int>>& temperature) = 0;
    virtual int measure(int i, int y, int x) = 0;
    virtual double measure(int i, int y, int x, int iter) = 0;
    virtual void answer(const std::vector<int>& estimate) = 0;
    virtual std::optional<Metrics> get_metrics() const = 0;

};

struct ServerJudge;
using ServerJudgePtr = std::shared_ptr<ServerJudge>;
struct ServerJudge : Judge {

    Input get_input() const {
        int L, N, S;
        std::vector<Pos> landing_pos;
        std::cin >> L >> N >> S;
        landing_pos.resize(N);
        for (int i = 0; i < N; i++) {
            std::cin >> landing_pos[i].y >> landing_pos[i].x;
        }
        return { L, N, S, landing_pos };
    }

    void set_temperature(const std::vector<std::vector<int>>& temperature) override {
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

    Input input;
    std::vector<int> A;
    std::vector<int> F;

    std::vector<std::tuple<int, int, int>> measurements;

    int turn = 0;
    std::vector<std::vector<int>> temperature;
    std::vector<int> estimate;

    int64_t placement_cost = 0;
    int64_t measurement_cost = 0;

    int wrong = 0;
    int64_t score = 0;

    FileJudge(const std::string& input_file_, const std::string& output_file_)
        : input_file(input_file_), output_file(output_file_) {
        std::ifstream in(input_file);
        in >> input.L >> input.N >> input.S;
        input.landing_pos.resize(input.N);
        for (int i = 0; i < input.N; i++) {
            in >> input.landing_pos[i].y >> input.landing_pos[i].x;
        }
        A.resize(input.N);
        F.resize(10000);
        in >> A >> F;
        in.close();
    }

    Input get_input() const { return input; }

    void set_temperature(const std::vector<std::vector<int>>& temperature_) override {
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

    Input input;
    std::vector<int> A;
    std::vector<int> F;

    std::vector<std::tuple<int, int, int>> measurements;

    int turn = 0;
    std::vector<std::vector<int>> temperature;
    std::vector<int> estimate;

    int64_t placement_cost = 0;
    int64_t measurement_cost = 0;

    int wrong = 0;
    int64_t score = 0;

    LocalJudge(int seed = 0, int L = -1, int N = -1, int S = -1) {
        std::mt19937_64 engine(seed);
        std::uniform_int_distribution<> dist_int;

        if (L == -1) L = dist_int(engine) % 41 + 10;
        if (N == -1) N = dist_int(engine) % 41 + 60;
        if (S == -1) {
            S = dist_int(engine) % 30 + 1;
            S *= S;
        }
        input.L = L;
        input.N = N;
        input.S = S;

        std::vector<Pos> cands;
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                cands.emplace_back(y, x);
            }
        }
        std::shuffle(cands.begin(), cands.end(), engine);
        input.landing_pos = std::vector<Pos>(cands.begin(), cands.begin() + N);

        A.resize(input.N);
        std::iota(A.begin(), A.end(), 0);
        std::shuffle(A.begin(), A.end(), engine);

        std::normal_distribution<> dist_norm(0.0, S);
        F.resize(10000);
        for (int i = 0; i < 10000; i++) {
            F[i] = (int)round(dist_norm(engine));
        }
    }

    Input get_input() const { return input; }

    void set_temperature(const std::vector<std::vector<int>>& temperature_) override {
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

struct Solver {

    static constexpr int dy[] = { 0, 0, -1, -1, -1, 0, 1, 1, 1 };
    static constexpr int dx[] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };

    Timer timer;

    JudgePtr judge;
    const Input input;
    const int L;
    const int N;
    const int S;
    const std::vector<Pos> landing_pos;

    const int num_quantize;
    const int num_neighbors;
    const int intercept;
    const int slope;
    const int num_trial;

    Solver(JudgePtr judge_, int num_quantize_, int num_neighbors_, int intercept_, int slope_, int num_trial_) :
        judge(judge_), input(judge->get_input()),
        L(input.L), N(input.N), S(input.S), landing_pos(input.landing_pos),
        num_quantize(num_quantize_), num_neighbors(num_neighbors_),
        intercept(intercept_), slope(slope_), num_trial(num_trial_) {}

    void solve() {
        const auto temperature = create_temperature();
        judge->set_temperature(temperature);
        const auto estimate = predict(temperature);
        judge->answer(estimate);
        auto metrics_opt = judge->get_metrics();
        if (metrics_opt) {
            auto [score, wrong, placement_cost, measurement_cost, measurement_count] = *metrics_opt;
            // dump(score, wrong, placement_cost, measurement_cost, measurement_count);
        }
    }

    std::vector<std::vector<int>> create_temperature() {

        // 中央・右・上・左・下のマスの値を見る
        // 3 値であれば、3^5=243 >= 100 なので特定可能

        std::vector<int> stride({ 1 });
        while (stride.size() < num_neighbors) {
            stride.push_back(stride.back() * num_quantize);
        }
        auto temperature = make_vector(0, L, L); // 0,1,2 の 3 値を取る
        auto fixed = make_vector(false, L, L);
        for (int i = 0; i < N; i++) {
            auto [y, x] = landing_pos[i];
            for (int d = 0; d < num_neighbors; d++) {
                int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
                fixed[ny][nx] = true;
            }
        }
        std::vector<Pos> cands;
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                if (fixed[y][x]) {
                    cands.emplace_back(y, x);
                }
            }
        }
        // まずはランダムに値を割り振る
        Xorshift rnd;
        for (auto [y, x] : cands) {
            temperature[y][x] = rnd.next_int(num_quantize);
        }
        // code が unique になるまでセルの値を変更する
        while (true) {
            { // random trans
                int idx = rnd.next_int(cands.size());
                auto [y, x] = cands[idx];
                temperature[y][x] = rnd.next_int(num_quantize);
            }
            // 各 landing_pos の符号を求める
            std::vector<int> codes(N);
            for (int i = 0; i < N; i++) {
                auto [y, x] = landing_pos[i];
                for (int d = 0; d < num_neighbors; d++) {
                    int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
                    codes[i] += temperature[ny][nx] * stride[d];
                }
            }
            std::vector<int> code_ctr((int)pow(num_quantize, num_neighbors));
            bool duplicate = false;
            for (int c : codes) {
                if (code_ctr[c]) {
                    duplicate = true;
                    break;
                }
                code_ctr[c]++;
            }
            if (duplicate) continue;
            dump(codes);
            dump(code_ctr);
            for (const auto& v : temperature) std::cerr << v << '\n';
            break;
        }
        // 0->500-32, 1->500, 2->500+32 と変換

        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                if (fixed[y][x]) {
                    temperature[y][x] = intercept + temperature[y][x] * slope;
                }
                else {
                    temperature[y][x] = intercept;
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

    std::vector<int> predict(const std::vector<std::vector<int>>& temperature) {
        std::vector<int> estimate(N);

        std::vector<int> qval(num_quantize);
        for (int q = 0; q < num_quantize; q++) {
            qval[q] = intercept + slope * q;
        }
        dump(qval);

        for (int i_in = 0; i_in < N; i_in++) {
            std::vector<int> temps(num_neighbors);
            for (int d = 0; d < num_neighbors; d++) {
                auto measured_value = judge->measure(i_in, dy[d], dx[d], num_trial);
                double min_diff = 1e9, min_q = -1;
                for (int q = 0; q < num_quantize; q++) {
                    if (chmin(min_diff, abs(measured_value - qval[q]))) {
                        min_q = q;
                    }
                }
                temps[d] = qval[min_q];
            }
            dump(i_in, temps);
            int best_hit = -1, best_id = -1;
            for (int i_out = 0; i_out < N; i_out++) {
                auto [y, x] = landing_pos[i_out];
                int hit = 0;
                for (int d = 0; d < num_neighbors; d++) {
                    int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
                    hit += temps[d] == temperature[ny][nx];
                }
                if (chmax(best_hit, hit)) {
                    best_id = i_out;
                }
            }
            estimate[i_in] = best_id;
        }
        dump(estimate);
        return estimate;
    }

    std::optional<Metrics> get_metrics() const {
        return judge->get_metrics();
    }

};

//void batch_execution() {
//
//    int num_seeds = 50;
//
//    // grid_search
//    int intervals[] = { 1,2,4,8,16 };
//    int num_trials[] = { 1,2,4,8,16,32,64,128,256 };
//    for (int interval : intervals) {
//        for (int num_trial : num_trials) {
//            std::vector<Metrics> metrics_list(num_seeds);
//
//            int progress = 0;
//            int64_t score_sum = 0;
//            int64_t min_score = INT64_MAX, max_score = INT64_MIN;
//
//#pragma omp parallel for num_threads(6)
//            for (int seed = 0; seed < num_seeds; seed++) {
//                //std::string input_file(format("../../tools_win/in/%04d.txt", seed));
//                //std::string output_file(format("../../tools_win/out/%04d.txt", seed));
//                //auto judge = std::make_shared<FileJudge>(input_file, output_file);
//                auto judge = std::make_shared<LocalJudge>(seed, -1, -1, 4);
//                Solver solver(judge);
//                //solver.set_params(interval, num_trial);
//                solver.set_params_opt();
//                solver.solve();
//#pragma omp critical(crit_sct)
//                {
//                    auto metrics = *solver.get_metrics();
//                    progress++;
//                    score_sum += metrics.score;
//                    chmin(min_score, metrics.score);
//                    chmax(max_score, metrics.score);
//                    std::cerr << format("\rprogress=%3d/%3d, avg=%13.2f, min=%11lld, max=%11lld, interval=%3d, num_trial=%3d", progress, num_seeds, (double)score_sum / progress, min_score, max_score, interval, num_trial);
//                    metrics_list[seed] = metrics;
//                }
//            }
//            std::cerr << '\n';
//            //std::cerr << format("\ninterval=%3d, num_trial=%3d, avg=%13.2f, min=%11lld, max=%11lld\n", interval, num_trial, (double)score_sum / progress, min_score, max_score);
//            //std::cerr << '\n';
//            //for (int seed = 0; seed < num_seeds; seed++) {
//            //    std::cerr << format("seed=%3d, ", seed) << metrics_list[seed] << '\n';
//            //}
//        }
//    }
//}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    //batch_execution();
    //exit(0);

    JudgePtr judge;

    if (false) {
        judge = std::make_shared<ServerJudge>();
    }
    else if(true) {
        std::string input_file("../../tools_win/in/0000.txt");
        std::string output_file("../../tools_win/out/0000.txt");
        judge = std::make_shared<FileJudge>(input_file, output_file);
    }
    else {
        judge = std::make_shared<LocalJudge>(0, 10, 60, 1);
    }

    Solver solver(judge, 2, 8, 0, 1000, 20);
    solver.solve();
    std::cerr << *solver.get_metrics() << '\n';

    return 0;
}