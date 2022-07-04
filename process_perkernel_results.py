# A script for generating plots and so forth from some of the traces in the
# results/ directory. Run "bash run_rocprof_experiments.sh" first.
import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy

def load_json(filename):
    data = None
    with open(filename) as f:
        data = json.loads(f.read())
    return data

def get_cdf_plots(measured_size, competitor_size):
    # 4 curves:
    #  - Unmanaged
    #  - 15 CUs
    #  - 30 Cus
    result_dir = "results_partitioning_data"
    filenames = [
        "%s/measured_%s_vs_4_%s_unpartitioned.json" % (result_dir,
            measured_size, competitor_size),
        "%s/measured_%s_vs_4_%s_15_cus.json" % (result_dir,
            measured_size, competitor_size),
        "%s/measured_%s_vs_4_%s_30_cus.json" % (result_dir,
            measured_size, competitor_size),
    ]
    labels = [
        "Unamanged",
        "Partitioned to 15 CUs",
        "Partitioned to 30 CUs",
    ]
    times = []
    for f in filenames:
        # Convert times to milliseconds
        jt = load_json(f)["job_times"]
        for i in range(len(jt)):
            jt[i] *= 1000.0
        times.append(jt)
    return plot_cdfs(times, labels, "%s vs %s competitors" % (measured_size,
        competitor_size))

def get_stats(all_times):
    """ Takes a list of filenames, and returns a dict with stats encompassing
    all of the job times in every one of the files. """
    all_times.sort()
    for i in range(len(all_times)):
        # Convert to ms
        all_times[i] *= 1000.0
    to_return = {
        "min": min(all_times),
        "max": max(all_times),
        "median": all_times[len(all_times) // 2],
        "mean": numpy.mean(all_times),
        "std_dev": numpy.std(all_times),
    }
    return to_return


def print_table():
    files = [
        ["isolated_unmanaged_med.json"],
        ["isolated_perkernel_med.json"],
        ["isolated_perjob_med.json"],
        ["2tasks_unmanaged_med_task0.json",
        "2tasks_unmanaged_med_task1.json"],
        ["2tasks_perkernel_med_task0.json",
        "2tasks_perkernel_med_task1.json"],
        ["2tasks_perjob_med_task0.json",
        "2tasks_perjob_med_task1.json"],
    ]
    has_competitor = [
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
    ]
    management = [
        "Unmanaged",
        "Per-Kernel Locking",
        "Per-Job Locking",
        "Unmanaged",
        "Per-Kernel Locking",
        "Per-Job Locking",
    ]
    print(r'''\begin{tabular}{|c|c|c c c c|}
\hline
With & \multirow{2}*{Management} & \multirow{2}*{Min (ms)} & \multirow{2}*{Max (ms)} & Arithmetic & Stdandard \\
Competitor? & & & & Mean (ms) & Deviation \\
\hline''')
    for i in range(len(files)):
        times = []
        for f in files[i]:
            data = load_json("results_perkernel/" + f)
            times.extend(data["job_times"])
        stats = get_stats(times)
        hc = has_competitor[i]
        m = management[i]
        line = "%s & %s & %.02f & %.02f & %.02f & %.02f \\\\" % (hc, m,
            stats["min"], stats["max"], stats["mean"], stats["std_dev"])
        print(line)
    print(r'\hline')
    print(r'\end{tabular}')

print_table()

