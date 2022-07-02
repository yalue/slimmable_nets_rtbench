# A script for generating plots and so forth from some of the traces in the
# results/ directory. Run "bash run_rocprof_experiments.sh" first.
import json
import matplotlib.pyplot as plot
import numpy
import re
import itertools

# Used to extract kernel names from the "args" object in the trace JSON.
kernel_name_re = re.compile(r'kernel\((?:void )?([^()<>]*)[()<>]')

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
        y_range / 5.0))

def plot_cdfs(all_times, labels):
    """ Takes a list of lists of kernel durations (in ms), and a list of
    labels, and plots all curves as CDFs. """
    style_cycler = itertools.cycle(get_line_styles())

    figure = plot.figure()
    # The title isn't necessary in the paper.
    # figure.suptitle("Kernel Time Distributions")
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

def get_kernel_name(event):
    max_length = 50
    matches = kernel_name_re.search(event["args"]["args"])
    name = matches.group(1)
    if len(name) > max_length:
        name = name[0 : (max_length - 1)] + "..."
    return name

def get_kernels(events):
    """ Returns a list: [(kernel name, kernel duration), ...]. """
    kernels = []
    for event in events:
        if "LaunchKernel" not in event["name"]:
            continue
        name = get_kernel_name(event)
        duration = event["args"]["AvgDurationS"]
        kernels.append((name, duration))
    return kernels

def load_json(filename):
    data = None
    with open("results_fixed_issue/" + filename) as f:
        data = json.loads(f.read())
    return data

def get_times_ms(filename):
    """ Returns a list of kernel durations, in ms, from the specified JSON
    file. """
    events = load_json(filename)
    kernels = get_kernels(events["traceEvents"])
    times_ms = []
    for k in kernels:
        times_ms.append(k[1] * 1000.0)
    return times_ms

def plot_from_files(filenames, labels):
    """ Gets a CDF using the given filenames and labels. """
    times = []
    for f in filenames:
        times.append(get_times_ms(f))
    return plot_cdfs(times, labels)

def print_table_line(filename, batch_size, width_mult):
    """ Formats the four columns of the given result file as tabular data. """
    data = load_json(filename)
    events = data["traceEvents"]
    job_time_ms = data["job_info"]["mean_job_time"] * 1000.0
    kernels = get_kernels(events)
    total_kernel_ms = 0.0
    for k in kernels:
        total_kernel_ms += k[1] * 1000.0
    kernel_percentage = (total_kernel_ms / job_time_ms) * 100.0
    print("%d & %.2f & %.03f & %.03f & %.01f\\%% \\\\" % (batch_size,
        width_mult, total_kernel_ms, job_time_ms, kernel_percentage))

def table_from_files(filenames, batch_sizes, width_mults):
    """ Prints LaTeX tabular data containing the total kernel time and overall
    job time for each filename. """
    print(r'\begin{tabular}{|c|c|c|c|c|}')
    print(r'\hline')
    print(r'\multirow{2}*{Batch Size} & Width & Mean Total & Mean Job & \% Kernel \\')
    print(r' & Multiplier & Kernel Time (ms) & Time (ms) & Execution \\')
    print(r'\hline')
    for i in range(len(filenames)):
        print_table_line(filenames[i], batch_sizes[i], width_mults[i])
    print(r'\hline')
    print(r'\end{tabular}')

figs = []

filenames = [
    "full_width.json",
    "50_width.json",
    "25_width.json",
]
labels = [
    "Width Mult = 1.0",
    "Width Mult = 0.5",
    "Width Mult = 0.25",
]
figs.append(plot_from_files(filenames, labels))

filenames = [
    "16_batch.json",
    "full_width.json",
    "64_batch.json",
    "128_batch.json",
]
labels = [
    "Batch Size = 16",
    "Batch Size = 32",
    "Batch Size = 64",
    "Batch Size = 128",
]
figs.append(plot_from_files(filenames, labels))
plot.show()

filenames = [
    "8_batch.json",
    "16_batch.json",
    "full_width.json",
    "64_batch.json",
    "128_batch.json",
    "50_width_8_batch.json",
    "50_width_16_batch.json",
    "50_width.json",
    "50_width_64_batch.json",
    "50_width_128_batch.json",
    "25_width_8_batch.json",
    "25_width_16_batch.json",
    "25_width.json",
    "25_width_64_batch.json",
    "25_width_128_batch.json",
]
tmp1 = [8, 16, 32, 64, 128]
tmp2 = [1.0] * 5 + [0.5] * 5 + [0.25] * 5
table_from_files(filenames, tmp1 * 3, tmp2)

