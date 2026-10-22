"""SAC 结果绘图入口（输出至 figures/sac/，读取 logs/sac/）。"""

from plot_results import build_arg_parser, run_plot


if __name__ == "__main__":
  run_plot(build_arg_parser(default_algorithm="sac").parse_args())
