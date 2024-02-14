
def is_raspberry_pi() -> bool:
    '''
    Check if the script is running on a Raspberry Pi.
    '''

    is_rpi = False  # Default to not RPi.

    try:
        with open('/proc/cpuinfo', 'r') as cpuinfo:
            for line in cpuinfo:
                if line.startswith('Model') and 'Raspberry' in line:
                    is_rpi = True

    except IOError:
        # /proc/cpuinfo is not accessible, not running on Raspberry Pi or Linux.
        is_rpi = False

    return is_rpi
