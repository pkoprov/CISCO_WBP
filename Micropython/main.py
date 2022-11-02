machine.freq(240000000) # overclock the CPU

# Turn LED off to indicate the setup is complete
time.sleep(1)
led.value(0)

# collect an ambient vibration
calibration_data = []
start = time.ticks_ms() / 1000
while time.ticks_ms() / 1000 - start < 1:
    calibration_data.append(m.acceleration[0])
    gc.collect()

# blink LED to indicate the finish of collecting the ambient vibration
for i in range(7):
    led.toggle()
    time.sleep(0.25)

# calculate the mean and SD of ambient vibration
mean = sum(calibration_data) / len(calibration_data)
sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5

start_flag = False
n = 0  # counter of datapoints below threshold

led.value(0)
# main loop
while True:
    # read the input
    ax, ay, az = m.acceleration
    # check if the values are exceeding the threshold
    if ax > mean + 5 * sd or ax < mean - 5 * sd and not start_flag:
        start_flag = True
        led.value(1)
        n = 0
    else:
        n += 1
        if n >= 100:
            start_flag = False
            led.value(0)

    if start_flag:
        # check if the values are the same
        if m.acceleration[0] == ax and m.acceleration[1] == ay and m.acceleration[2] == az:
            continue
        print(ax, ay, az)
