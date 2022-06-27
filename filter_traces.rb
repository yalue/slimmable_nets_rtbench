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
def update_arg_times(event, base_time)
  return if !event.include?("args")
  args = event["args"]
  return if !args.include?("BeginNs")
  start_s = args["BeginNs"].to_i.to_f / 1.0e9
  end_s = args["EndNs"].to_i.to_f / 1.0e9
  args["BeginS"] = start_s - base_time
  args["EndS"] = end_s - base_time
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

if ARGV.size != 2
  puts "Usage: ruby #{$0} <input file.json> <output_file.json>"
  exit 1
end

content = File.open(ARGV[0], "rb") {|f| JSON.parse(f.read)}
puts "Read #{ARGV[0]} OK."
puts "Got %d events in input file." % [content["traceEvents"].size]

base_time = get_earliest_time(content["traceEvents"])
puts "Base time " + base_time.to_s
filtered_content = {}
filtered_content["otherData"] = content["otherData"]
filtered_events = []

# We'll only keep stuff found after FakeKernelA
kernel_a_found = false

content["traceEvents"].each do |event|
  if !kernel_a_found
    # We don't bother including FakeKernelA in the filtered output.
    kernel_a_found = is_fake_kernel_a(event)
    next
  end
  next if !passes_filter(event)
  # We don't bother including FakeKernelB in the filtered output, either.
  break if is_fake_kernel_b(event)

  # We've seen fake_kernel_a and we're keeping this event.
  update_arg_times(event, base_time)
  filtered_events << event
end
filtered_content["traceEvents"] = filtered_events

File.open(ARGV[1], "wb") {|f| f.write(JSON.pretty_generate(filtered_content))}
puts "Wrote #{ARGV[1]} OK."
puts "Output #{filtered_events.size.to_s} events."

