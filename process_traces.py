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
    color_options = [
        "black",
        "cyan",
        "red",
        "magenta",
        "blue",
        "green",
        "y",
    ]
    # [Solid line, dashed line, dash-dot line, dotted line]
    dashes_options = [
        [1, 0],
        [3, 1, 3, 1],
        [3, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    marker_options = [
        None,
        "o",
        "v",
        "s",
        "*",
        "+",
        "D"
    ]
    # Build a combined list containing every style combination.
    all_styles = []
    for m in marker_options:
        for d in dashes_options:
            for c in color_options:
                to_add = {}
                if m is not None:
                    to_add["marker"] = m
                    to_add["markevery"] = 0.1
                to_add["c"] = c
                to_add["dashes"] = d
                all_styles.append(to_add)
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
    figure.suptitle("Kernel Time Distributions")
    axes = figure.add_subplot(1, 1, 1)
    # Make the axes track data exactly, we'll manually add padding later.
    axes.autoscale(enable=True, axis='both', tight=True)

    for i in range(len(all_times)):
        times = all_times[i]
        cdf = convert_values_to_cdf(times)
        axes.plot(cdf[0], cdf[1], lw=1.5, label=labels[i],
            **next(style_cycler))

    add_plot_padding(axes)
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("% <= X")

    # TODO: Put legend on the top of the plot
    legend = plot.legend()
    legend.set_draggable(True)
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
        duration = event["args"]["DurationS"]
        kernels.append((name, duration))
    return kernels

def get_times_ms(filename):
    """ Returns a list of kernel durations, in ms, from the specified JSON
    file. """
    data = None
    with open(filename) as f:
        data = json.loads(f.read())
    events = data["traceEvents"]
    kernels = get_kernels(events)
    times_ms = []
    for k in kernels:
        times_ms.append(k[1] * 1000.0)
    return times_ms

data = None
with open("rocprof_trace_filtered.json") as f:
    data = json.loads(f.read())
events = data["traceEvents"]

kernels = get_kernels(events)
times_ms = []
for k in kernels:
    print("%s,%.12f" % (k[0], k[1]))
    times_ms.append(k[1] * 1000.0)

bar_ranges = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
]

# The first "label" would sit on the y axis, so we'll add a dummy spot to keep
# the rest of the stuff centered.
bar_labels = [""]
bar_data = [0]

for i in range(len(bar_ranges)):
    range_end = bar_ranges[i]
    range_start = 0.0
    if i > 0:
        range_start = bar_ranges[i - 1]
    bar_labels.append("%.4f - %.4f" % (range_start, range_end))
    count = 0
    for t in times_ms:
        if t < range_start:
            continue
        if t >= range_end:
            continue
        count += 1
    print("Got %d kernels in range %s" % (count, bar_labels[-1]))
    bar_data.append(count)

max_range = bar_ranges[-1]
bar_labels.append(">= %.3f" % (max_range, ))
count = 0
for t in times_ms:
    if t < max_range:
        continue
    count += 1
bar_data.append(count)

figure = plot.figure()
figure.suptitle("Binned Kernel Times")
axes = figure.add_subplot(1, 1, 1)
xdata = numpy.arange(len(bar_data))
bar_container = axes.bar(xdata, bar_data)
axes.set_xlim(left=0, right=len(bar_data))
axes.set_xticklabels(bar_labels)
axes.set_xlabel("Kernel Durations (ms)")
axes.set_ylabel("Number of Kernels")
cdf_figure = plot_cdf(times_ms)
plot.show()

