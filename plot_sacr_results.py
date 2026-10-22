"""SACr 结果绘图入口（输出至 figures/sacr/，读取 logs/sacr/）。"""

from plot_results import build_arg_parser, run_plot


if __name__ == "__main__":
  run_plot(build_arg_parser(default_algorithm="sacr").parse_args())
