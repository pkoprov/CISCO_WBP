calibration_data = []
start = time.ticks_ms() / 1000
while time.ticks_ms() / 1000 - start < 1:
    calibration_data.append(m.acceleration[0])
    gc.collect()

mean = sum(calibration_data) / len(calibration_data)
sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5

print("Ready")
# main loop
while True:
    # read the input
    ax, ay, az = m.acceleration
    if ax > mean + 5 * sd or ax < mean - 5 * sd:
        try:
            print(ax, ay, az)
            # sys.stdout.write(struct.pack('3d', ax, ay, az) + '\n'.encode())
        except:
            print("Not working!")
            time.sleep(1)
