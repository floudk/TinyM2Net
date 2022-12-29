import time
import os
def read_sys_info(init_free_memory,logfile):
    
    power_path = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input"
    memory_useds = []
    cpu_energy = 0
    sleep_time_interval = 0.1
    last_time = time.time()
    current_time = time.time()
    try:
        while 1:
            with os.popen('free -m', 'r') as fd:
                lines = fd.readlines()
                line = lines[1].split()
                used_mem = init_free_memory - int(line[3])
                memory_useds.append(used_mem) 
            
            with open(power_path, 'r') as f:
            
                cpu_power = int(f.readline())
                cpu_energy += cpu_power*sleep_time_interval
            time.sleep(sleep_time_interval)
            
            current_time = time.time()
            
            if current_time-last_time>=1:
                print("Average Used Memory: {} M, CPU Energy: {} mJ".format(sum(memory_useds)/len(memory_useds), cpu_energy))
                last_time = current_time
    except KeyboardInterrupt:
        with open(logfile,'w') as f:
            f.write(str(memory_useds))

if __name__=="__main__":
    
    read_sys_info(2349, "./Memory_LOG/normalconvmodel.log")