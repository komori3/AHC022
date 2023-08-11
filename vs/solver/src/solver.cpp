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
};

struct Input {
    int L, N, S;
    std::vector<Pos> landing_pos;
};

struct Judge;
using JudgePtr = std::shared_ptr<Judge>;
struct Judge {

    virtual Input get_input() const = 0;
    virtual void set_temperature(const std::vector<std::vector<int>>& temperature) = 0;
    virtual int measure(int i, int y, int x) = 0;
    virtual double measure(int i, int y, int x, int iter) = 0;
    virtual void answer(const std::vector<int>& estimate) = 0;

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

};

struct FileJudge;
using FileJudgePtr = std::shared_ptr<FileJudge>;
struct FileJudge : Judge {

    std::ifstream in;
    std::ofstream out;

    Input input;
    std::vector<int> A;
    std::vector<int> F;

    int turn = 0;
    std::vector<std::vector<int>> temperature;
    std::vector<int> estimate;

    int64_t placement_cost = 0;
    int64_t measurement_cost = 0;

    FileJudge(const std::string& input_file, const std::string& output_file) : in(input_file), out(output_file) {
        in >> input.L >> input.N >> input.S;
        input.landing_pos.resize(input.N);
        for (int i = 0; i < input.N; i++) {
            in >> input.landing_pos[i].y >> input.landing_pos[i].x;
        }
        A.resize(input.N);
        F.resize(10000);
        in >> A >> F;
        dump(A);
        dump(F);
    }

    Input get_input() const { return input; }

    void set_temperature(const std::vector<std::vector<int>>& temperature_) override {
        temperature = temperature_;
        for (const auto& row : temperature) {
            for (int i = 0; i < row.size(); i++) {
                out << row[i] << (i == row.size() - 1 ? "\n" : " ");
            }
        }
        out.flush();

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
        out << i << " " << y << " " << x << '\n';
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
        out << "-1 -1 -1\n";
        for (int e : estimate) {
            out << e << '\n';
        }
        int wrong = 0;
        for (int i = 0; i < input.N; i++) wrong += A[i] != estimate[i];
        dump(placement_cost, measurement_cost, wrong);
        int64_t score = (int64_t)ceil(1e14 * pow(0.8, wrong) / (1e5 + placement_cost + measurement_cost));
        dump(score);
    }

};

struct Solver {

    Timer timer;

    const int L;
    const int N;
    const int S;
    const std::vector<Pos> landing_pos;
    JudgePtr judge;

    Solver(const Input& input, JudgePtr judge_) 
        : L(input.L), N(input.N), S(input.S), landing_pos(input.landing_pos), judge(judge_) {}

    void solve() {
        const auto temperature = create_temperature();
        judge->set_temperature(temperature);
        const auto estimate = predict(temperature);
        judge->answer(estimate);
    }

    std::vector<std::vector<int>> create_temperature() {

        // 中央から遠いものから割当
        std::vector<std::tuple<int, int, int>> tmp;
        for (int i = 0; i < N; i++) {
            tmp.emplace_back(i, landing_pos[i].y, landing_pos[i].x);
        }
        int M = L / 2;
        std::sort(tmp.begin(), tmp.end(), [&](const auto& a, const auto& b) {
            auto [ai, ay, ax] = a;
            auto [bi, by, bx] = b;
            int da = (ay - M) * (ay - M) + (ax - M) * (ax - M);
            int db = (by - M) * (by - M) + (bx - M) * (bx - M);
            return da > db;
            });

        auto temperature = make_vector(0, L, L);
        auto fixed = make_vector(false, L, L);
        int C = (int)floor(1000 / N);
        // set the temperature to i * 10 for i-th position
        for (int k = 0; k < N; k++) {
            int i = std::get<0>(tmp[k]);
            temperature[landing_pos[i].y][landing_pos[i].x] = k * C;
            fixed[landing_pos[i].y][landing_pos[i].x] = true;
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

        Xorshift rnd;
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

        return temperature;
    }

    std::vector<int> predict(const std::vector<std::vector<int>>& temperature) {
        std::vector<int> estimate(N);
        int T = 10000 / N;
        for (int i_in = 0; i_in < N; i_in++) {
            auto measured_value = judge->measure(i_in, 0, 0, T);
            dump(i_in, measured_value);
            // answer the position with the temperature closest to the measured value
            double min_diff = 9999;
            for (int i_out = 0; i_out < N; i_out++) {
                const Pos& pos = landing_pos[i_out];
                double diff = abs(temperature[pos.y][pos.x] - measured_value);
                if (diff < min_diff) {
                    min_diff = diff;
                    estimate[i_in] = i_out;
                }
            }
        }
        return estimate;
    }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    JudgePtr judge;

    if (false) {
        judge = std::make_shared<ServerJudge>();
    }
    else {
        std::string input_file("../../tools_win/in/0000.txt");
        std::string output_file("../../tools_win/out/0000.txt");
        judge = std::make_shared<FileJudge>(input_file, output_file);
    }

    auto input = judge->get_input();

    Solver solver(input, judge);
    solver.solve();

    return 0;
}