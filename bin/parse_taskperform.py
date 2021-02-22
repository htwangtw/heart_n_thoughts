from heart_n_thoughts.dataset import parse_taskperform

# execute from the top of dir
in_file = "task-nbackmindwandering_performance.tsv"
out_file = "task_performance.tsv"

performance =  parse_taskperform(in_file, out_file)