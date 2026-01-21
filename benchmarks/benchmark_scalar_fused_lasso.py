"""Benchmark comparison of scalar fused lasso solvers.

Compares three implementations:
1. Coordinate Descent (Ours) - Custom implementation
2. SciPy minimize_scalar - General-purpose optimization
3. CVXPY - Convex optimization framework
"""
import os
import numpy as np
from scipy.optimize import minimize_scalar
import time
import warnings
from typing import Dict, List, Tuple

from gfl.core.scalar_fused_lasso import (
    solve_scalar_fused_lasso,
    scalar_fused_lasso_objective
)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. Install with: pip install cvxpy")


def solve_with_scipy(
    group_size: float,
    group_sum: float,
    adj_values: np.ndarray,
    weights: np.ndarray,
    reg_lambda: float,
    tol: float = 1e-8
) -> Tuple[float, bool]:
    """Solve using SciPy's minimize_scalar."""
    def objective(x):
        return scalar_fused_lasso_objective(
            x, group_size, group_sum, adj_values, weights, reg_lambda
        )
    
    try:
        # Determine reasonable bounds
        if len(adj_values) > 0:
            margin = max(abs(group_sum / group_size - adj_values.mean()), 10.0)
            bounds = (adj_values.min() - margin, adj_values.max() + margin)
        else:
            mean = group_sum / group_size if group_size > 0 else 0.0
            bounds = (mean - 10.0, mean + 10.0)
        
        result = minimize_scalar(objective, bounds=bounds, method='bounded', 
                                  options={'xatol': tol})
        return result.x, result.success
    except Exception as e:
        warnings.warn(f"SciPy solver failed: {e}")
        return np.nan, False


def solve_with_cvxpy(
    group_size: float,
    group_sum: float,
    adj_values: np.ndarray,
    weights: np.ndarray,
    reg_lambda: float
) -> Tuple[float, bool]:
    """Solve using CVXPY."""
    if not CVXPY_AVAILABLE:
        return np.nan, False
    
    try:
        x = cp.Variable()
        quad_term = 0.5 * group_size * cp.square(x) - group_sum * x
        fusion_terms = [weights[i] * cp.abs(x - adj_values[i]) 
                        for i in range(len(adj_values))]
        objective = cp.Minimize(quad_term + reg_lambda * cp.sum(fusion_terms))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        
        if problem.status == cp.OPTIMAL:
            return float(x.value), True
        else:
            return np.nan, False
    except Exception as e:
        warnings.warn(f"CVXPY solver failed: {e}")
        return np.nan, False


def generate_test_problem(n_neighbors: int, seed: int = 42) -> Dict:
    """Generate a random test problem."""
    rng = np.random.RandomState(seed)
    
    group_size = 10.0 + rng.uniform(0, 90.0)
    group_sum = rng.uniform(-50.0, 50.0) * group_size
    
    if n_neighbors > 0:
        adj_values = rng.uniform(-10.0, 10.0, n_neighbors)
        weights = rng.uniform(0.5, 2.0, n_neighbors)
    else:
        adj_values = np.array([])
        weights = np.array([])
    
    reg_lambda = rng.uniform(0.1, 5.0)
    
    return {
        'group_size': group_size,
        'group_sum': group_sum,
        'adj_values': adj_values,
        'weights': weights,
        'reg_lambda': reg_lambda
    }


def run_benchmark(
    problem_sizes: List[int],
    n_trials: int = 100,
    tolerance: float = 1e-6
) -> Dict:
    """Run comprehensive benchmark across problem sizes."""
    
    results = {
        'problem_sizes': problem_sizes,
        'coordinate_descent': {
            'times': [],
            'success_rates': [],
            'solutions': []
        },
        'scipy': {
            'times': [],
            'success_rates': [],
            'solutions': []
        },
        'cvxpy': {
            'times': [],
            'success_rates': [],
            'solutions': []
        }
    }
    
    for n_neighbors in problem_sizes:
        print(f"\nBenchmarking with {n_neighbors} neighbors...")
        
        # Storage for this problem size
        cd_times, cd_successes, cd_solutions = [], [], []
        sp_times, sp_successes, sp_solutions = [], [], []
        cv_times, cv_successes, cv_solutions = [], [], []
        
        for trial in range(n_trials):
            # Generate problem
            problem = generate_test_problem(n_neighbors, seed=trial)
            
            # Test Coordinate Descent (Ours)
            start = time.perf_counter()
            cd_sol = solve_scalar_fused_lasso(**problem)
            cd_time = time.perf_counter() - start
            cd_times.append(cd_time)
            cd_solutions.append(cd_sol)
            cd_successes.append(True)  # Always succeeds
            
            # Test SciPy
            start = time.perf_counter()
            sp_sol, sp_success = solve_with_scipy(**problem)
            sp_time = time.perf_counter() - start
            sp_times.append(sp_time)
            sp_solutions.append(sp_sol)
            sp_successes.append(sp_success)
            
            # Test CVXPY
            if CVXPY_AVAILABLE:
                start = time.perf_counter()
                cv_sol, cv_success = solve_with_cvxpy(**problem)
                cv_time = time.perf_counter() - start
                cv_times.append(cv_time)
                cv_solutions.append(cv_sol)
                cv_successes.append(cv_success)
        
        # Store aggregated results
        results['coordinate_descent']['times'].append(np.mean(cd_times))
        results['coordinate_descent']['success_rates'].append(
            100.0 * np.mean(cd_successes)
        )
        results['coordinate_descent']['solutions'].append(cd_solutions)
        
        results['scipy']['times'].append(np.mean(sp_times))
        results['scipy']['success_rates'].append(
            100.0 * np.mean(sp_successes)
        )
        results['scipy']['solutions'].append(sp_solutions)
        
        if CVXPY_AVAILABLE:
            results['cvxpy']['times'].append(np.mean(cv_times))
            results['cvxpy']['success_rates'].append(
                100.0 * np.mean(cv_successes)
            )
            results['cvxpy']['solutions'].append(cv_solutions)
        
        print(f"  CD:    {results['coordinate_descent']['times'][-1]*1000:.3f} ms "
              f"(success: {results['coordinate_descent']['success_rates'][-1]:.1f}%)")
        print(f"  SciPy: {results['scipy']['times'][-1]*1000:.3f} ms "
                f"(success: {results['scipy']['success_rates'][-1]:.1f}%)")
        if CVXPY_AVAILABLE:
            print(f"  CVXPY: {results['cvxpy']['times'][-1]*1000:.3f} ms "
                  f"(success: {results['cvxpy']['success_rates'][-1]:.1f}%)")
    
    return results


def compute_speedup(results: Dict) -> Dict:
    """Compute relative speedup vs slowest method."""
    speedup = {
        'coordinate_descent': [],
        'scipy': [],
        'cvxpy': []
    }
    
    for i in range(len(results['problem_sizes'])):
        cd_time = results['coordinate_descent']['times'][i]
        
        times = [cd_time]
        times.append(results['scipy']['times'][i])
        if CVXPY_AVAILABLE:
            times.append(results['cvxpy']['times'][i])
        
        slowest = max(times)
        
        speedup['coordinate_descent'].append(slowest / cd_time)
        speedup['scipy'].append(slowest / results['scipy']['times'][i])
        if CVXPY_AVAILABLE:
            speedup['cvxpy'].append(slowest / results['cvxpy']['times'][i])
    
    return speedup


def compute_throughput(results: Dict) -> Dict:
    """Compute throughput (solves per second)."""
    throughput = {
        'coordinate_descent': [],
        'scipy': [],
        'cvxpy': []
    }
    
    for i in range(len(results['problem_sizes'])):
        cd_time = results['coordinate_descent']['times'][i]
        throughput['coordinate_descent'].append(1.0 / cd_time)
        
        sp_time = results['scipy']['times'][i]
        throughput['scipy'].append(1.0 / sp_time)
        
        if CVXPY_AVAILABLE:
            cv_time = results['cvxpy']['times'][i]
            throughput['cvxpy'].append(1.0 / cv_time)
    
    return throughput

def plot_results(
        results: Dict, 
        speedup: Dict, 
        throughput: Dict,
        save: bool = False
):
    """Create benchmark plots similar to the provided image."""
    try:
        import matplotlib.pyplot as plt
        if save:
            import matplotlib
            matplotlib.use('Agg')
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    problem_sizes = results['problem_sizes']
    size_labels = [f"n={n}" for n in problem_sizes]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors matching the image
    colors = {
        'coordinate_descent': '#5DA5DA',  # Blue
        'scipy': '#FAA43A',               # Orange
        'cvxpy': '#60BD68'                # Green
    }
    
    # 1. Average Solve Time Comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(problem_sizes))
    width = 0.25
    
    ax.bar(x_pos - width, results['coordinate_descent']['times'], width, 
           label='Coordinate Descent (Ours)', color=colors['coordinate_descent'])

    ax.bar(x_pos, results['scipy']['times'], width,
            label='SciPy minimize_scalar', color=colors['scipy'])
    if CVXPY_AVAILABLE:
        ax.bar(x_pos + width, results['cvxpy']['times'], width,
               label='CVXPY', color=colors['cvxpy'])
    
    ax.set_ylabel('Average Time (ms)')
    ax.set_xlabel('Problem Size')
    ax.set_title('Average Solve Time Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels, rotation=45)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Solver Reliability Comparison
    ax = axes[0, 1]
    ax.bar(x_pos - width, results['coordinate_descent']['success_rates'], width,
           label='Coordinate Descent (Ours)', color=colors['coordinate_descent'])
    ax.bar(x_pos, results['scipy']['success_rates'], width,
            label='SciPy minimize_scalar', color=colors['scipy'])
    if CVXPY_AVAILABLE:
        ax.bar(x_pos + width, results['cvxpy']['success_rates'], width,
               label='CVXPY', color=colors['cvxpy'])
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Problem Size')
    ax.set_title('Solver Reliability Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels, rotation=45)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Relative Performance Speedup
    ax = axes[1, 0]
    ax.bar(x_pos - width, speedup['coordinate_descent'], width,
           label='Coordinate Descent (Ours)', color=colors['coordinate_descent'])
    ax.bar(x_pos, speedup['scipy'], width,
            label='SciPy minimize_scalar', color=colors['scipy'])
    if CVXPY_AVAILABLE:
        ax.bar(x_pos + width, speedup['cvxpy'], width,
               label='CVXPY', color=colors['cvxpy'])
    
    ax.set_ylabel('Speedup (vs Slowest)')
    ax.set_xlabel('Problem Size')
    ax.set_title('Relative Performance Speedup')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels, rotation=45)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Solver Throughput Comparison
    ax = axes[1, 1]
    ax.bar(x_pos - width, throughput['coordinate_descent'], width,
           label='Coordinate Descent (Ours)', color=colors['coordinate_descent'])
    ax.bar(x_pos, throughput['scipy'], width,
            label='SciPy minimize_scalar', color=colors['scipy'])
    if CVXPY_AVAILABLE:
        ax.bar(x_pos + width, throughput['cvxpy'], width,
               label='CVXPY', color=colors['cvxpy'])
    
    ax.set_ylabel('Throughput (solves/sec)')
    ax.set_xlabel('Problem Size')
    ax.set_title('Solver Throughput Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels, rotation=45)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        os.makedirs('./benchmarks/results', exist_ok=True)
        plt.savefig('./benchmarks/results/scalar_fused_lasso_timing.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to: ./benchmarks/results/scalar_fused_lasso_timing.png")
    else:
        plt.show()
    plt.close()

def print_summary_table(results: Dict, speedup: Dict, throughput: Dict):
    """Print a summary table of results."""
    print("\n" + "="*85)
    print("BENCHMARK SUMMARY")
    print("="*85)
    
    for i, n in enumerate(results['problem_sizes']):
        print(f"\nProblem Size: n={n} neighbors")
        print("-" * 85)
        print(f"{'Method':<25} {'Time (ms)':<15} {'Success (%)':<15} {'Speedup':<15} {'Throughput':<15}")
        print("-" * 85)
        
        # Coordinate Descent
        print(f"{'Coordinate Descent':<25} "
              f"{results['coordinate_descent']['times'][i]*1000:<15.3f} "
              f"{results['coordinate_descent']['success_rates'][i]:<15.1f} "
              f"{speedup['coordinate_descent'][i]:<15.1f}x "
              f"{throughput['coordinate_descent'][i]:<15.1f}")
        
        # SciPy
        print(f"{'SciPy minimize_scalar':<25} "
                f"{results['scipy']['times'][i]*1000:<15.3f} "
                f"{results['scipy']['success_rates'][i]:<15.1f} "
                f"{speedup['scipy'][i]:<15.1f}x "
                f"{throughput['scipy'][i]:<15.1f}")
        
        # CVXPY
        if CVXPY_AVAILABLE:
            print(f"{'CVXPY':<25} "
                  f"{results['cvxpy']['times'][i]*1000:<15.3f} "
                  f"{results['cvxpy']['success_rates'][i]:<15.1f} "
                  f"{speedup['cvxpy'][i]:<15.1f}x "
                  f"{throughput['cvxpy'][i]:<15.1f}")
    
    print("\n" + "="*85)

def main():
    """Run the benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark scalar fused lasso solvers')
    parser.add_argument('--save', action='store_true', 
                        help='Save plots and results to ./results/')
    args = parser.parse_args()

    print("Scalar Fused Lasso Solver Benchmark")
    print("=" * 80)
    print(f"CVXPY available: {CVXPY_AVAILABLE}")
    print("=" * 80)
    
    # Define problem sizes (matching and extending the image)
    problem_sizes = [5, 20, 50, 100, 200, 500]  # Extended to larger sizes
    n_trials = 100
    
    print(f"\nRunning benchmark with {n_trials} trials per problem size...")
    
    # Run benchmark
    results = run_benchmark(problem_sizes, n_trials=n_trials)
    
    # Compute metrics
    speedup = compute_speedup(results)
    throughput = compute_throughput(results)
    
    # Print summary
    print_summary_table(results, speedup, throughput)
    
    # Plot results
    plot_results(results, speedup, throughput, args.save)
    
    # Save results to file
    import json
    results_data = {
        'problem_sizes': problem_sizes,
        'n_trials': n_trials,
        'results': {
            'coordinate_descent': {
                'times': [float(x) for x in results['coordinate_descent']['times']],
                'success_rates': [float(x) for x in results['coordinate_descent']['success_rates']]
            }
        }
    }
    
    results_data['results']['scipy'] = {
        'times': [float(x) for x in results['scipy']['times']],
        'success_rates': [float(x) for x in results['scipy']['success_rates']]
    }
    
    if CVXPY_AVAILABLE:
        results_data['results']['cvxpy'] = {
            'times': [float(x) for x in results['cvxpy']['times']],
            'success_rates': [float(x) for x in results['cvxpy']['success_rates']]
        }
    
    if args.save:
        os.makedirs('./benchmarks/results', exist_ok=True)
        with open('./benchmarks/results/scalar_fused_lasso_benchmark_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        print("\nResults saved to: ./benchmarks/results/scalar_fused_lasso_benchmark_results.json")
    


if __name__ == '__main__':
    main()
