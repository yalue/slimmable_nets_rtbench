require 'json'

# Returns the earliest time, in seconds, that's in the trace.
def get_earliest_time(events)
  earliest_time = 1.0e10
  events.each do |event|
    next if !event.include?("args")
    next if !event["args"].include?("BeginNs")
    t = event["args"]["BeginNs"].to_i / 1.0e9
    earliest_time = t if t < earliest_time
  end
  earliest_time
end

# Returns true if the given event should be included in the output JSON
def passes_filter(event)
  # Get rid of the "DataFlow" noise
  return false if event["cat"] == "DataFlow"
  # Get rid of junk related to old-style kernel launches, usually related to
  # MIOpen
  skip_names = ["hipGetDevice", "hipCtxSetCurrent", "hipSetDevice",
    "__hipPushCallConfiguration", "__hipPopCallConfiguration",
    "hipGetDeviceProperties", "hipGetDeviceCount"]
  return false if skip_names.include?(event["name"])
  true
end

# If the event includes timestamps, this adds numerical timestamps to the JSON,
# as an offset relative to the earliest time.
def update_arg_times(event, base_time, kernel_time_info)
  return if !event.include?("args")
  args = event["args"]
  return if !args.include?("BeginNs")
  start_s = args["BeginNs"].to_i.to_f / 1.0e9
  end_s = args["EndNs"].to_i.to_f / 1.0e9
  args["BeginS"] = start_s - base_time
  args["EndS"] = end_s - base_time

  if event["name"].include?("LaunchKernel")
    args_str = args["args"]
    name = kernel_name(args_str)
    dims = grid_dims(args_str)
    id = name + dims.to_s
    info = kernel_time_info[id]
    if !info
      puts "Previously unseen kernel " + id.to_s
      exit 1
    end
    avg_s = info["total_time"] / info["calls"].to_f
    args["AvgDurationS"] = avg_s
    args["grid_dim"] = info["dims"][0]
    args["block_dim"] = info["dims"][1]
  end
  event["args"] = args
end

# Returns true if the given event is a launch of FakeKernelA
def is_fake_kernel_a(event)
  return false if event["name"] != "hipLaunchKernel"
  if event["args"]["args"] =~ / kernel\(FakeKernelA/
    return true
  end
  false
end

# Returns true if the given event is a launch of FakeKernelB
def is_fake_kernel_b(event)
  return false if event["name"] != "hipLaunchKernel"
  if event["args"]["args"] =~ / kernel\(FakeKernelB/
    return true
  end
  return false
end

# Returns a kernel's name from its "args" string.
def kernel_name(str)
  if str !~ /\( kernel\(/
    puts "Bad kernel args string: " + str
    exit 1
  end
  # Chop "( kernel(" off the start
  current_index = 9
  close_parens_needed = 1
  while true
    char = str[current_index]
    if char == "("
      close_parens_needed += 1
    elsif char == ")"
      close_parens_needed -= 1
      break if close_parens_needed == 0
    end
    current_index += 1
  end
  str[9...current_index]
end

# Returns block and grid dimensions: [[gridX, gridY, gridZ],
# [blockX, blockY, blockZ]]
def grid_dims(str)
  grid = [0, 0, 0]
  block = [0, 0, 0]
  if str =~ / globalWorkSizeX\((\d+)\) globalWorkSizeY\((\d+)\) globalWorkSizeZ\((\d+)\) /
    grid[0] = $1.to_i
    grid[1] = $2.to_i
    grid[2] = $3.to_i
  end
  if str =~ / blockDimX\((\d+)\) blockDimY\((\d+)\) blockDimZ\((\d+)\) /
    block[0] = $1.to_i
    block[1] = $2.to_i
    block[2] = $3.to_i
  end
  return [grid, block]
end

# Updates the dict mapping kernel IDs -> time info.
def update_kernel_time_info(info, event)
  return if !event["name"].include?("LaunchKernel")
  args = event["args"]["args"]
  name = kernel_name(args)
  dims = grid_dims(args)
  id = name + dims.to_s
  if !info.include?(id)
    new_info = {"id"=>id, "dims"=>dims, "clean_name"=>name, "calls"=>0,
      "total_time"=>0.0}
    info[id] = new_info
  end
  info[id]["calls"] += 1
  info[id]["total_time"] += event["args"]["ActualDuration"]
end

# The durations listed in the hipLaunchKernel events are bogus.
def get_actual_kernel_durations(events)
  to_return = []
  seen_kernel_b = false
  events.each do |event|
    # Looking for is_fake_kernel_b will "align" us with the start of the events
    # with the correct DurationNs measurements.
    if !seen_kernel_b
      seen_kernel_b = is_fake_kernel_b(event)
      next
    end
    next if !event.include?("args")
    args = event["args"]
    next if !args.include?("queue-id")
    duration_s = args["DurationNs"].to_i.to_f / 1.0e9
    to_return << duration_s
  end
  to_return
end

# Copy kernel's actual durations from their separate events near the end of the
# trace.
def copy_kernel_durations(events)
  actual_durations = get_actual_kernel_durations(events)
  kernel_index = 0
  events.each do |event|
    next if !event.include?("name")
    next if !event["name"].include?("LaunchKernel")
    break if kernel_index > actual_durations.size
    event["args"]["ActualDuration"] = actual_durations[kernel_index]
    kernel_index += 1
  end
  if kernel_index != actual_durations.size
    puts "Got %d kernels, but %d durations!" % [kernel_index, actual_durations.size]
    exit 1
  end
  puts "Fixed kernel durations."
end

if ARGV.size != 3
  puts "Usage: ruby #{$0} <input file.json> <log JSON from pytorch script> <output_file.json>"
  exit 1
end

content = File.open(ARGV[0], "rb") {|f| JSON.parse(f.read)}
puts "Read #{ARGV[0]} OK."
puts "Got %d events in input file." % [content["traceEvents"].size]
copy_kernel_durations(content["traceEvents"])

base_time = get_earliest_time(content["traceEvents"])
puts "Base time " + base_time.to_s
filtered_content = {}
# Uncomment if we want this stuff.
# filtered_content["otherData"] = content["otherData"]
filtered_events = []

# We'll only keep events after the *second* FakeKernelA, but we'll start taking
# average kernel time after the first.
kernel_a_count = 0
kernel_time_info = {}

content["traceEvents"].each do |event|
  # Start by looking for the first FakeKernelA
  if kernel_a_count == 0
    kernel_a_count += 1 if is_fake_kernel_a(event)
    next
  end

  # Next, gather average performance info until the next FakeKernelA
  if kernel_a_count == 1
    if is_fake_kernel_a(event)
      kernel_a_count += 1
      next
    end
    update_kernel_time_info(kernel_time_info, event)
    next
  end

  # Finally, record events that pass the filter.
  next if !passes_filter(event)
  # We don't bother including FakeKernelB in the filtered output, either.
  break if is_fake_kernel_b(event)

  # We've seen fake_kernel_a and we're keeping this event.
  update_arg_times(event, base_time, kernel_time_info)
  filtered_events << event
end

filtered_content["traceEvents"] = filtered_events

job_content = File.open(ARGV[1], "rb") {|f| JSON.parse(f.read)}
filtered_content["job_info"] = job_content

File.open(ARGV[2], "wb") {|f| f.write(JSON.pretty_generate(filtered_content))}
puts "Wrote #{ARGV[2]} OK."
puts "Output #{filtered_events.size.to_s} events."

