# A script for generating plots and so forth from some of the traces in the
# results/ directory. Run "bash run_rocprof_experiments.sh" first.
import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re

def convert_values_to_cdf(values):
    """Takes a 1-D list of values and converts it to a CDF representation. The
    CDF consists of a vector of times and a vector of percentages of 100."""
    if len(values) == 0:
        return [[], []]
    values.sort()
    total_size = float(len(values))
    current_min = values[0]
    count = 0.0
    data_list = [values[0]]
    ratio_list = [0.0]
    for v in values:
        count += 1.0
        if v > current_min:
            data_list.append(v)
            ratio_list.append((count / total_size) * 100.0)
            current_min = v
    data_list.append(values[-1])
    ratio_list.append(100)
    return [data_list, ratio_list]

all_styles = None
def get_line_styles():
    """Returns a list of line style possibilities, that includes more options
    than matplotlib's default set that includes only a few solid colors."""
    global all_styles
    if all_styles is not None:
        return all_styles
    line_styles = []
    line_styles.append({"color": "k", "linestyle": "-"})
    line_styles.append({"color": "red", "linestyle": "--", "markevery": 0.075,
        "markersize": 6, "marker": "x", "mew": 1.0})
    line_styles.append({"color": "blue", "linestyle": "-", "markevery": 0.075,
        "markersize": 5, "marker": "o"})
    line_styles.append({"color": "green", "linestyle": "--",
        "markevery": 0.075, "markersize": 6, "marker": ">"})
    line_styles.append({"color": "k", "linestyle": "-.", "markevery": 0.075,
        "markersize": 6, "marker": "*"})
    line_styles.append({"color": "grey", "linestyle": "--"})
    line_styles.append({"color": "k", "linestyle": "-",
        "dashes": [8, 4, 2, 4, 2, 4]})
    line_styles.append({"color": "grey", "linestyle": "-",
        "dashes": [8, 4, 2, 4, 2, 4]})
    line_styles.append({"color": "grey", "linestyle": "-."})
    all_styles = line_styles
    return all_styles

def add_plot_padding(axes):
    """Takes matplotlib axes, and adds some padding so that lines close to
    edges aren't obscured by tickmarks or the plot border."""
    y_limits = axes.get_ybound()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xbound()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0] - y_pad, y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
        x_range / 5.0))
    axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
        y_range / 10.0))

def plot_cdfs(all_times, labels, title):
    """ Takes a list of lists of kernel durations (in ms), and a list of
    labels, and plots all curves as CDFs. """
    style_cycler = itertools.cycle(get_line_styles())

    figure = plot.figure(num=title)
    axes = figure.add_subplot(1, 1, 1)
    # Make the axes track data exactly, we'll manually add padding later.
    axes.autoscale(enable=True, axis='both', tight=True)

    for i in range(len(all_times)):
        times = all_times[i]
        cdf = convert_values_to_cdf(times)
        axes.plot(cdf[0], cdf[1], lw=1.0, label=labels[i],
            **next(style_cycler))

    add_plot_padding(axes)
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("% <= X")

    legend = plot.legend(loc=3, ncol=2, bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        mode="expand", borderaxespad=0.0)
    #legend.set_draggable(True)
    return figure

def load_json(filename):
    data = None
    with open(filename) as f:
        data = json.loads(f.read())
    return data

def generate_plots():
    #filenames = glob.glob("4way_sharing_results/*.json")
    globs = [
        "4_competitors_*_8_batch_25_width*.json",
        "4_competitors_*_32_batch_50_width*.json",
        "4_competitors_*_64_batch_100_width*.json",
    ]
    filenames = []
    for g in globs:
        filenames.extend(list(glob.glob("4way_sharing_results/" + g)))
    experiments = {}
    for f in filenames:
        content = load_json(f)
        experiment_name = content["args"]["experiment_name"]
        if experiment_name == "":
            print("Skipping %s: no experiment name." % (f,))
            continue
        if experiment_name not in experiments:
            experiments[experiment_name] = {}
        experiment = experiments[experiment_name]
        scenario_name = content["args"]["scenario_name"]
        if scenario_name == "":
            print("Skipping %s: no scenario name." % (f, ))
            continue
        if scenario_name not in experiment:
            experiment[scenario_name] = []
        experiment[scenario_name].extend(content["job_times"])

    label_scenarios = {
        "4-Way Partitioning": '4-way sharing (partitioned)',
        "Exclusive Locking": 'Exclusive access',
        "2-Exclusion Locking": '2-way sharing (unpartitioned)',
        "Unmanaged": "Unmanaged",
        "2-Excl. w/ Partitioning": '2-way sharing (partitioned)',
    }
    label_list = [
        "Unmanaged",
        "Exclusive Locking",
        "2-Exclusion Locking",
        "2-Excl. w/ Partitioning",
        "4-Way Partitioning",
    ]
    figures = []
    for experiment_name in experiments:
        print("Processing experiment " + experiment_name + ":")
        experiment = experiments[experiment_name]
        scenario_names = list(experiment.keys())
        scenario_names.sort()
        times = []
        for label in label_list:
            scenario_name = label_scenarios[label]
            samples = experiment[label_scenarios[label]]
            print("  Scenario %s: %d samples" % (scenario_name, len(samples)))
            for i in range(len(samples)):
                # Convert to ms
                samples[i] *= 1000.0
            times.append(samples)
        figures.append(plot_cdfs(times, label_list, experiment_name))
    plot.show()

def get_stats(filenames):
    """ Takes a list of filenames, and returns a dict with stats encompassing
    all of the job times in every one of the files. """
    all_times = []
    blocking_times = []
    for f in filenames:
        content = load_json(f)
        all_times.extend(content["job_times"])
        bt = content["blocking_times"]
        # FIX A BUG: I was including some uninitialized blocking times in here,
        # so we will check each value and convert it to ms if it's valid.
        for i in range(len(bt)):
            t = bt[i]
            if t >= 2.0:
                continue
            blocking_times.append(t * 1000.0)
    all_times.sort()
    blocking_times.sort()
    # Convert to ms
    for i in range(len(all_times)):
        all_times[i] *= 1000.0
    mean_blocking = numpy.mean(blocking_times)
    mean_job = numpy.mean(all_times)
    to_return = {
        "min": min(all_times),
        "max": max(all_times),
        "median": all_times[len(all_times) // 2],
        "mean": mean_job,
        "std_dev": numpy.std(all_times),
        "mean_blocking": mean_blocking,
        "blocking_percent": (mean_blocking / mean_job) * 100.0,
    }
    return to_return

def generate_table_section(task_size):
    """ Takes a task size, "small", "med", or "large", and generates the
    portion of the table with lock-based info for it. """

    batch_width = {
        "small": (8, 25),
        "med": (32, 50),
        "large": (64, 100),
    }[task_size]
    row_labels = [
        "Unmanaged",
        "Exclusive Locking",
        "2-Exclusion Locking",
        "2-Excl. w/ Partitioning",
        "4-Way Partitioning",
    ]
    row_fnames = [
        "unmanaged",
        "exclusive",
        "2way_unpartitioned",
        "2way_partitioned",
        "4way_partitioned",
    ]
    for i in range(len(row_labels)):
        rl = row_labels[i]
        line = ""
        if i == 0:
            line += r'\multirow{5}*{\nn' + task_size + r'{}} & '
        else:
            line += " & "
        line += rl + " & "
        t = "4_competitors_%s_%d_batch_%d_width_task*.json" % (row_fnames[i],
            batch_width[0], batch_width[1])
        filenames = glob.glob("4way_sharing_results/" + t)
        stats = get_stats(filenames)
        line += "%.02f & %.02f & %.02f & %.02f \\\\" % (stats["min"],
            stats["max"], stats["mean"], stats["std_dev"])
        print(line)

# Table layout:
# - 3 sections: small, medium, and large tasks
#   - Row 1: Unmanaged
#   - Row 2: Exclusive locking
#   - Row 3: 2-Exclusion locking
#   - Row 4: 2-Exclusion locking and partitioning
#   - Row 5: 4-Way Partitioning
def generate_tables():
    sizes = ["small", "medium", "large"]
    print(r'''\begin{tabular}{|c|c|c c c c|}
\hline
Task & Management & \multirow{2}*{Min (ms)} & \multirow{2}*{Max (ms)} & Arithmetic & Standard \\
Sizes & Technique & & & Mean (ms) & Deviation \\
\hline''')
    generate_table_section("small")
    print(r'\hline')
    generate_table_section("med")
    print(r'\hline')
    generate_table_section("large")
    print(r'\hline')
    print(r'\end{tabular}')

# Show a ton of plots.
generate_plots()

generate_tables()

