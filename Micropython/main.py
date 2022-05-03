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

duration = float(sys.stdin.readline())

led.value(0)
# main loop
while True:
    # read the input
    ax, ay, az = m.acceleration
    if ax > mean + 5 * sd or ax < mean - 5 * sd:
        start_time = time.ticks_ms() / 1000
        led.value(1)

        while time.ticks_ms() / 1000 - start_time < duration:
            if m.acceleration[0] == ax and m.acceleration[1] == ay and m.acceleration[2] == az:
                continue
            ax, ay, az = m.acceleration
            print(ax, ay, az)
        led.value(0)
        time.sleep(2)