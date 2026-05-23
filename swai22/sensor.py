import modi_plus
import time
import math

bundle = modi_plus.MODIPlus()
bundle.modules

env = bundle.envs[0]
imu = bundle.imus[0]
display = bundle.displays[0]
motor = bundle.motors[0]

first_a_x = imu.acceleration_x
first_a_y = imu.acceleration_y
first_a_z = imu.acceleration_z

def get_sensor():
    temperature = env.temperature
    humidity = env.humidity

    v_x, v_y, v_z = 0, 0, 0

    last_time = time.time()

    cureent_a_x = imu.acceleration_x 
    current_a_y = imu.acceleration_y
    current_a_z = imu.acceleration_z

    time.sleep(0.2)
    current_time = time.time()
    dt = current_time - last_time

    v_x = abs(cureent_a_x - first_a_x) * dt
    v_y = abs(current_a_y - first_a_y) * dt
    v_z = abs(current_a_z - first_a_z) * dt

    v_total = int(math.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2))

    return (temperature, humidity, v_total)

def set_motor():
    display.text = f"스프링클러를가동합니다!!"
    motor.speed = 70
    time.sleep(3)
    motor.speed = 0

def display_info():
    text = f"            temp : {str(get_sensor()[0])}℃   humi : {str(get_sensor()[1])}%  wind : {str(get_sensor()[2])}m/s"
    display.text = text

# if __name__ == "__main__":
#     while True:
#         display_info()