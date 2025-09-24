#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

namespace ga {
    enum class SelectionType { RouletteWheel, Tournament };        // 选择算子: 轮盘赌 锦标赛
    enum class CrossoverType { SinglePoint, TwoPoint, Algorithm }; // 交叉算子：单点交叉 两点交叉 算术交叉
    enum class MutationType { Uniform, Gaussian };                 // 变异算子：均匀变异 高斯变异

    template <std::size_t PopulationSize, std::size_t ChromosomeSize, auto fitness_function>
    class GeneticAlgorithm {
    private:
        struct Individual {
            double fitness; // 个体的适应度大小
            std::array<double, ChromosomeSize> chromosome;
            auto get(std::size_t i) const {
                return chromosome[i];
            }
            auto &get(std::size_t i) {
                return chromosome[i];
            }
            // 个体：有ChromosomeSize个染色体，每个染色体的取值范围为range[i]
        };

    public:
        GeneticAlgorithm(bool maximization,
                         float mutation_rate,
                         float crossover_rate,
                         std::array<std::pair<double, double>, ChromosomeSize> range)
        : _maximization(maximization),
          _mutation_rate(mutation_rate),
          _crossover_rate(crossover_rate),
          _uniform_dist(0.0, 1.0),
          range(range) {
            init_population();
        }

        Individual get_best_individual() {
            sort_population();
            return _population.front();
        }

        Individual solve(std::size_t max_generations,
                         SelectionType selection_type = SelectionType::Tournament,
                         CrossoverType crossover_type = CrossoverType::SinglePoint,
                         MutationType mutation_type = MutationType::Gaussian,
                         std::size_t tournament_size = 2,
                         double sigma = 0.1) {
            for (std::size_t i = 0; i < max_generations; ++i) {
                evolve(selection_type, crossover_type, mutation_type, tournament_size, sigma);
            }
            return get_best_individual();
        }

        friend std::ostream &operator<<(std::ostream &ostream, GeneticAlgorithm &self) {
            auto best = self.get_best_individual();
            ostream << "fitness: " << (self._maximization ? best.fitness : -best.fitness) << std::endl;
            ostream << "chromosome: ";
            for (auto c : best.chromosome) {
                ostream << c << ' ';
            }
            return ostream << std::endl;
        }

    private:
        bool _maximization;    // 最大化问题
        float _mutation_rate;  // 变异率
        float _crossover_rate; // 交叉率
        std::mt19937 _rng;
        std::uniform_real_distribution<> _uniform_dist; // 0-1随机分布

        std::array<Individual, PopulationSize> _population;          // 个体组成的种群
        std::array<std::pair<double, double>, ChromosomeSize> range; // 每一个染色体的取值范围

        void update_fitness(Individual &ind) {
            ind.fitness = _maximization ? fitness_function(ind.chromosome) : -fitness_function(ind.chromosome);
            // 让其都是最大化
            // 计算个体的适应度
        }

        void sort_population() {
            std::sort(_population.begin(), _population.end(), [this](const Individual &a, const Individual &b) {
                return a.fitness > b.fitness;
            }); // 按适应度排序
        }

        void init_population() {
            for (std::size_t i = 0; i < PopulationSize; ++i) {
                Individual ind;
                for (std::size_t j = 0; j < ChromosomeSize; ++j) {
                    std::uniform_real_distribution<double> rng(range[j].first, range[j].second);
                    ind.get(j) = rng(_rng); // 随机初始化染色体
                }
                update_fitness(ind);
                _population[i] = ind;
            }
            sort_population();
        }

        Individual roulette_wheel_selection() { // 轮盘赌选择, 这里C++用discrete_distribution就很方便
            std::vector<double> weights;
            auto min_fitness = _population.back().fitness;
            for (const auto &ind : _population) {
                weights.push_back(ind.fitness - min_fitness + 1);
            }
            std::discrete_distribution<> rng(weights.begin(), weights.end()); // 正数
            return _population[rng(_rng)];
        }

        Individual tournament_selection(std::size_t tournament_size) { // 锦标赛选择
            std::uniform_int_distribution<> rng(0, PopulationSize - 1);
            Individual best = _population[rng(_rng)];
            for (std::size_t i = 0; i < tournament_size; ++i) {
                auto ind = _population[rng(_rng)];
                if (ind.fitness > best.fitness) {
                    best = ind;
                }
            }
            return best;
        }

        Individual selection(SelectionType selection_type, std::size_t tournament_size) {
            switch (selection_type) {
                case SelectionType::RouletteWheel:
                    return roulette_wheel_selection();
                case SelectionType::Tournament:
                    return tournament_selection(tournament_size);
                default:
                    return _population[0];
            }
        }

        std::pair<Individual, Individual> crossover_algorithm(const Individual &a,
                                                              const Individual &b) { // 染色体交叉 算数交叉
            Individual child1 = a, child2 = b;
            if (_uniform_dist(_rng) < _crossover_rate) {
                auto alpha = _uniform_dist(_rng);
                for (std::size_t i = 0; i < ChromosomeSize; ++i) {
                    child1.get(i) = alpha * a.get(i) + (1 - alpha) * b.get(i);
                    child2.get(i) = (1 - alpha) * a.get(i) + alpha * b.get(i);
                }
            }
            return {child1, child2};
        }

        std::pair<Individual, Individual> crossover_single_point(const Individual &a,
                                                                 const Individual &b) { // 染色体交叉 单点交叉
            Individual child1 = a, child2 = b;
            if (_uniform_dist(_rng) < _crossover_rate) {
                for (std::size_t i = 0; i < ChromosomeSize; ++i) {
                    if (_uniform_dist(_rng) < 0.5) {
                        std::swap(child1.get(i), child2.get(i));
                    }
                }
            }
            return {child1, child2};
        }

        std::pair<Individual, Individual> crossover_two_point(const Individual &a,
                                                              const Individual &b) { // 染色体交叉 两点交叉
            Individual child1 = a, child2 = b;
            if (_uniform_dist(_rng) < _crossover_rate) {
                for (std::size_t i = 0; i < ChromosomeSize; ++i) {
                    std::uniform_int_distribution<> pointDist(0, ChromosomeSize - 1);
                    std::size_t point1 = pointDist(_rng);
                    std::size_t point2 = pointDist(_rng);

                    while (point1 == point2) {
                        point2 = pointDist(_rng);
                    }

                    if (point1 > point2) {
                        std::swap(point1, point2);
                    }

                    for (std::size_t i = point1; i <= point2; ++i) {
                        std::swap(child1.get(i), child2.get(i));
                    }
                }
            }
            return {child1, child2};
        }

        std::pair<Individual, Individual> crossover(const Individual &a,
                                                    const Individual &b,
                                                    CrossoverType crossover_type) { // 染色体交叉 选择交叉类型
            switch (crossover_type) {
                case CrossoverType::SinglePoint:
                    return crossover_single_point(a, b);
                case CrossoverType::TwoPoint:
                    return crossover_two_point(a, b);
                case CrossoverType::Algorithm:
                    return crossover_algorithm(a, b);
                default:
                    return {a, b};
            }
        }

        void mutation_uniform(Individual &ind) { // 均匀变异
            for (std::size_t i = 0; i < ChromosomeSize; ++i) {
                if (_uniform_dist(_rng) < _mutation_rate) {
                    std::uniform_real_distribution<> rng(range[i].first, range[i].second);
                    ind.get(i) = rng(_rng); // 随机变异
                }
            }
        }

        void mutation_gaussian(Individual &ind, double sigma) { // 高斯变异
            for (std::size_t i = 0; i < ChromosomeSize; ++i) {
                if (_uniform_dist(_rng) < _mutation_rate) {
                    std::normal_distribution<> rng(0, sigma * (range[i].second - range[i].first));
                    ind.get(i) += rng(_rng); // 随机变异
                    ind.get(i) = std::clamp(ind.get(i), range[i].first, range[i].second);
                }
            }
        }

        void mutation(Individual &ind, MutationType mutation_type, double sigma) { // 变异
            switch (mutation_type) {
                case MutationType::Uniform:
                    mutation_uniform(ind);
                    break;
                case MutationType::Gaussian:
                    mutation_gaussian(ind, sigma);
                    break;
                default:
                    break;
            }
        }

        void evolve(SelectionType selection_type,
                    CrossoverType crossover_type,
                    MutationType mutation_type,
                    std::size_t tournament_size = 2,
                    double sigma = 0.1) {
            std::array<Individual, PopulationSize> new_population;
            for (std::size_t i = 0; i < PopulationSize; ++i) {
                Individual parent1 = selection(selection_type, tournament_size);
                Individual parent2 = selection(selection_type, tournament_size);
                auto [child1, child2] = crossover(parent1, parent2, crossover_type);
                mutation(child1, mutation_type, sigma);
                mutation(child2, mutation_type, sigma);
                update_fitness(child1);
                update_fitness(child2);
                new_population[i] = child1;
                if (i + 1 < PopulationSize) {
                    new_population[i + 1] = child2;
                }
            }
            _population = new_population;
        }
    };
} // namespace ga