import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, veh_mass,fuel_capacity,brake_deadband,decel_limit,accel_limit,
                wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        # TODO: Tune
        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1,max_lat_accel,max_steer_angle)

        kp = 0.2
        ki = 0.1
        kd = 0.05
        min_th = 0 #min throttle value
        max_th = 0.1
        self.throttle_controller = PID(kp,ki,kd,min_th,max_th)

        tau = 0.5 #cutoff frequency = 1/(2pi*tau)
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau,ts)

        self.veh_mass = veh_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_vel = 0.0


    def control(self, current_vel,dbw_enabled,linear_vel,ang_vel):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.,0.,0.
        
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel,ang_vel,current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error,sample_time)
        brake = 0

        if 0.==linear_vel and 0.1>current_vel:
            throttle = 0
            brake = 700

        elif throttle<0.1 and vel_error<0:
            throttle = 0
            decel = max(vel_error,self.decel_limit)
            brake = abs(decel)*self.veh_mass*self.wheel_radius #torque

        return throttle, brake, steering
