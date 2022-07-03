# A script for generating plots and so forth from some of the traces in the
# results/ directory. Run "bash run_rocprof_experiments.sh" first.
import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy

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
    # The title isn't necessary in the paper.
    #figure.suptitle(title)
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

def get_cdf_plots(measured_size, competitor_size):
    # 4 curves:
    #  - Unmanaged
    #  - 15 CUs
    #  - 30 Cus
    result_dir = "results"
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

def print_table_section(measured_size, competitor_size):
    partitioning = ["15", "20", "30", "Unpartitioned"]
    for i in range(len(partitioning)):
        p = partitioning[i]
        filename = "results/measured_%s_" % (measured_size)
        if competitor_size == "None":
            filename += "isolated_"
        else:
            filename += "vs_4_%s_" % (competitor_size,)
        if p == "Unpartitioned":
            filename += "unpartitioned.json"
        else:
            filename += "%s_cus.json" % (p,)
        line = ""
        if i == 0:
            tmp = competitor_size
            if tmp != "None":
                tmp = r'\nn' + tmp + '{}'
            line += r'\multirow{4}*{' + tmp + "} & "
        else:
            line += " & "
        line += p + " & "
        content = load_json(filename)
        line += "%.03f & " % (content["min_job_time"] * 1000.0,)
        line += "%.03f & " % (content["max_job_time"] * 1000.0,)
        line += "%.03f & " % (content["mean_job_time"] * 1000.0,)
        line += "%.03f " % (content["job_time_std_dev"] * 1000.0,)
        line += r'\\'
        print(line)

def print_table(measured_size):
    print(r'''\begin{tabular}{|c|c|c c c c|}
\hline
\multirow{2}*{Competitor} & Partition & \multirow{2}*{Min (ms)} & \multirow{2}*{Max (ms)} & Arithmetic & Stdandard \\
 & Size (CUs) & & & Mean (ms) & Deviation \\
\hline''')
    competitor_sizes = ["None", "small", "med", "large"]
    for s in competitor_sizes:
        print_table_section(measured_size, s)
        print(r'\hline')
    print(r'\end{tabular}')

print("\nTable with medium measured task:\n")
print_table("med")

figures = []
figures.append(get_cdf_plots("med", "large"))
plot.show()

