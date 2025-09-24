#include <cmath>
#include <iostream>
#include <numbers>
#include "GeneticAlgorithm.hpp"

using namespace ga;

template <std::size_t N>
using Range = std::array<std::pair<double, double>, N>;

int main() {
    {
        constexpr auto X = 1; // 一元函数
        auto f = [](std::array<double, X> x) -> double {
            return x[0] * x[0]; // 一元函数， x^2
        };

        // 一元函数， x 的范围为 [-5, 5]
        GeneticAlgorithm<10, X, f> test1(false, 0.05, 0.8, Range<X>{{{-5, 5}}});

        test1.solve(100);
        std::cout << "test1 best solution: " << test1 << std::endl;

        // 取0的时候最小
    }

    {
        constexpr auto X = 1; // 一元函数
        auto f = [](std::array<double, X> x) -> double {
            return 5 * x[0] * std::exp(x[0]); // 一元函数， x^2
        };

        // 一元函数， x 的范围为 [-5, 0]
        GeneticAlgorithm<10, X, f> test2(false, 0.05, 0.8, Range<X>{{{-5, 0}}});

        test2.solve(100);
        std::cout << "test2 best solution: " << test2 << std::endl;

        // 最小值大概-1.84， 取1最小
    }

    {
        constexpr auto X = 5;                            // 5元函数
        auto f = [](std::array<double, X> x) -> double { // 多峰函数
            const int A = 10;

            double res = 0;
            for (std::size_t i = 0; i < X; ++i) {
                res += x[i] * x[i] - A * std::cos(2 * std::numbers::pi * x[i]);
            }
            return res + A * X;
        };

        GeneticAlgorithm<10, X, f> test3(
            false, 0.05, 0.8, Range<X>{{{-5.12, 5.12}, {-5.12, 5.12}, {-5.12, 5.12}, {-5.12, 5.12}, {-5.12, 5.12}}});

        test3.solve(100);
        std::cout << "test3 best solution: " << test3 << std::endl;

        // 最小值大概0， 取0最小
    }

    {
        constexpr auto X = 2;                            // 2元函数
        auto f = [](std::array<double, X> x) -> double { // Rosenbrock 函数
            return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
        };

        GeneticAlgorithm<100, X, f> test4(false, 0.1, 0.8, Range<X>{{{-2.048, 2.048}, {-2.048, 2.048}}});

        test4.solve(1000, SelectionType::Tournament, CrossoverType::SinglePoint, MutationType::Gaussian, 2, 0.001);
        std::cout << "test4 best solution: " << test4 << std::endl; // 似乎很难调 ， 变异率大一点似乎好一点
        // F(1,1)取最小
    }

    {
        constexpr auto X = 2;                            // 2元函数
        auto f = [](std::array<double, X> x) -> double { // Himmelblau 函数
            return (x[0] * x[0] + x[1] - 11) * (x[0] * x[0] + x[1] - 11) +
                   (x[0] + x[1] * x[1] - 7) * (x[0] + x[1] * x[1] - 7);
        };

        GeneticAlgorithm<100, X, f> test4(false, 0.1, 0.8, Range<X>{{{-6, 6}, {-6, 6}}});

        test4.solve(1000, SelectionType::Tournament, CrossoverType::SinglePoint, MutationType::Gaussian, 2, 0.001);
        std::cout << "test4 best solution: " << test4 << std::endl;
        // 有好几个最小值点
        // 最小值是0
    }

    {
        // Schwefel 函数
        constexpr auto X = 10;                           // 10元函数
        auto f = [](std::array<double, X> x) -> double { // Schwefel 函数
            double res = 0;
            for (std::size_t i = 0; i < X; ++i) {
                res -= x[i] * std::sin(std::sqrt(std::abs(x[i])));
            }
            return res;
        };

        GeneticAlgorithm<100, X, f> test5(false,
                                          0.05,
                                          0.8,
                                          Range<X>{{{-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500},
                                                    {-500, 500}}});

        test5.solve(1000, SelectionType::Tournament, CrossoverType::SinglePoint, MutationType::Gaussian, 4, 0.1);
        std::cout << "test5 best solution: " << test5 << std::endl;
        // 最小值是x_i=420.9687
        // 最小值是F(x_i)=-418.9829*10
    }
    {
        // Eggholder
        constexpr auto X = 2;                            // 2元函数
        auto f = [](std::array<double, X> x) -> double { // Eggholder 函数
            return -(x[1] + 47) * std::sin(std::sqrt(std::abs(x[0] / 2 + (x[1] + 47)))) -
                   x[0] * std::sin(std::sqrt(std::abs(x[0] - (x[1] + 47))));
        };

        GeneticAlgorithm<100, X, f> test6(false, 0.05, 0.8, Range<X>{{{-512, 512}, {-512, 512}}});

        test6.solve(1000, SelectionType::Tournament, CrossoverType::SinglePoint, MutationType::Gaussian, 4, 0.1);
        std::cout << "test6 best solution: " << test6 << std::endl;
        // 最小值是x_i=512, x_j=404.2319
        // 最小值是F(x_i,x_j)=-959.6407
    }

    return 0;
}
