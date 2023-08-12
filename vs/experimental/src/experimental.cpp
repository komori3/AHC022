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



void calc_error() {
    // 標準偏差 S の場合に、温度 t のセルを温度 t-d 以下 or t+d 以上と判定する確率
    int S = 4;
    std::mt19937_64 engine(0);
    std::normal_distribution<> dist_norm(0, S);

    int t1 = 100, d = 16, nsample = 1000000, ntrial = 1;
    std::vector<int> hist(200);
    for (int i = 0; i < nsample; i++) {
        double xsum = 0.0;
        for (int j = 0; j < ntrial; j++) {
            int x = t1 + (int)round(dist_norm(engine));
            assert(0 <= x && x < 200);
            xsum += x;
        }
        hist[(int)round(xsum / ntrial)]++;
    }

    int wrong = nsample;
    for (int x = t1 - d + 1; x <= t1 + d - 1; x++) wrong -= hist[x];
    dump(wrong, nsample, double(wrong) / nsample);
}



struct Pos {
    int y, x;
    Pos(int y = -1, int x = -1) : y(y), x(x) {}
    std::string stringify() const {
        return "(" + std::to_string(y) + ", " + std::to_string(x) + ")";
    }
    bool operator<(const Pos& rhs) const {
        return y == rhs.y ? x < rhs.x : y < rhs.y;
    }
};

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

void find_unique_encoding_grid(
    const std::vector<Pos>& pos,
    int L, // grid size
    int Q, // number of quantization
    int D  // number of neighbors
) {
    // 6 2 5
    // 3 0 1
    // 7 4 8
    constexpr int dy[] = { 0, 0, -1, 0, 1, -1, -1, 1, 1 };
    constexpr int dx[] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };

    const int N = pos.size();

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
    std::vector<int> code_ctr((int)pow(Q, D));
    for (int i = 0; i < N; i++) {
        auto [y, x] = pos[i];
        int code = 0;
        for (int d = 0; d < D; d++) {
            int ny = (y + dy[d] + L) % L, nx = (x + dx[d] + L) % L;
            code += grid[ny][nx] * powers[d];
        }
        codes[i] = code;
        code_ctr[code]++;
    }

    int cost = 0;
    for (int c : code_ctr) {
        int x = std::max(0, c - 1);
        cost += x * x;
    }
    dump(code_ctr);
    dump(cost);

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
        for (auto [i, d] : grid_to_id[y][x]) {
            pop(codes[i]);
            codes[i] += diff * powers[d];
            push(codes[i]);
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
    while (cost) {
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
    dump(loop);
    dump(codes);
    dump(code_ctr);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    int L = 10, N = 64, Q = 4, D = 3;
    auto pos = choose_positions(L, N, 4);
    find_unique_encoding_grid(pos, L, Q, D);

    return 0;
}