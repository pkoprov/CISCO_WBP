machine.freq(240000000)
calibration_data = []
time.sleep(1)
led.value(0)
start = time.ticks_ms() / 1000
while time.ticks_ms() / 1000 - start < 1:
    calibration_data.append(m.acceleration[0])
    gc.collect()

for i in range(7):
    led.toggle()
    time.sleep(0.25)

mean = sum(calibration_data) / len(calibration_data)
sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5

start_flag = False
n = 0  # counter of datapoints below threshold

led.value(0)
# main loop
while True:
    # read the input
    ax, ay, az = m.acceleration
    if ax > mean + 5 * sd or ax < mean - 5 * sd and not start_flag:
        #         start_time = time.ticks_ms()/1000
        start_flag = True
        led.value(1)
        n = 0
    else:
        n += 1
        if n >= 100:
            start_flag = False
            led.value(0)

    if start_flag:
        if m.acceleration[0] == ax and m.acceleration[1] == ay and m.acceleration[2] == az:
            continue
        print(ax, ay, az)
